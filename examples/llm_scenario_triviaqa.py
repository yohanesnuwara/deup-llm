"""Scenario A: DEUP risk prediction for a frozen HF LLM on open-domain QA (TriviaQA).

Unlike the GSM8K example (single numeric answer, hard 0/1 exact-match loss), this
script targets *free-form textual* answers. Correctness is graded with a robust
**containment exact-match** (does any gold alias appear in the model's final
answer), and the DEUP error predictor g(x) is trained to regress that loss from
frozen-LLM confidence features. Repeated-sampling semantic entropy is enabled as
an extra feature.

Reasoning vs non-reasoning models  (IMPORTANT)
----------------------------------------------
Qwen3.x and reasoning models (e.g. ``deepreinforce-ai/Ornith-1.0-9B``) open the
assistant turn with a ``<think> ... </think>`` trace before the final answer. Two
consequences:

  * The grader strips the reasoning trace and scores only the post-``</think>``
    answer (``strip_reasoning``). A *dangling* (unclosed) ``<think>`` means the
    generation was truncated before the model finished thinking -> no usable
    answer -> scored wrong. This is correct behaviour, but it means a too-small
    token budget makes EVERY row wrong.

  * TriviaQA is factual recall, not multi-step reasoning. You almost always want
    thinking OFF: ``--thinking off`` (default) passes ``enable_thinking=False``
    through the chat template so no ``<think>`` block is opened, the model answers
    directly, and ``--max-new-tokens 64`` is plenty.

    Use ``--thinking on`` only to study reasoning-trace uncertainty, and then you
    MUST raise ``--max-new-tokens`` (>=512) so the think block can close.

The token-confidence features (log-probs, entropy, margins) are computed over the
*full* generation; when thinking is on, that includes the trace -- intentional.

Install extras:
    pip install -e ".[llm]"
    pip install datasets tqdm scipy

Example runs
------------
# Non-reasoning recall (recommended for TriviaQA), Qwen3.5-27B in 4-bit:
uv run python examples/llm_scenario_triviaqa.py \
    --model-id Qwen/Qwen3.5-27B --device-map auto --load-in-4bit \
    --max-memory-gpu 30GiB --max-memory-cpu 120GiB \
    --train-size 150 --test-size 50 --max-new-tokens 64 --semantic-samples 5

# Small non-reasoning model:
uv run python examples/llm_scenario_triviaqa.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct --device-map cuda \
    --train-size 100 --test-size 50 --max-new-tokens 64 --semantic-samples 3

# Study reasoning traces (thinking ON needs a big token budget):
uv run python examples/llm_scenario_triviaqa.py \
    --model-id Qwen/Qwen3.5-9B --device-map cuda --thinking on \
    --train-size 100 --test-size 50 --max-new-tokens 512 --semantic-samples 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from deup.domains.llm import (
    HFGenerationConfig,
    LLMDEUPRiskEstimator,
    normalize_text,
    token_f1_loss,
)

# Control character used to pack multiple gold aliases into a single reference
# string, so the whole thing still flows through the estimator's element-wise
# ``loss_fn(answer, reference)`` contract.
ALIAS_SEP = "\x1f"

# Matches a balanced <think>...</think> block (any casing, spanning newlines).
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def strip_reasoning(text: str) -> str:
    """Return only the final answer, dropping any <think> reasoning trace.

    Handles three cases:
      * balanced ``<think>...</think>`` blocks are removed;
      * a dangling open ``<think>`` with no close (generation truncated before the
        model finished thinking) means there is no usable answer -> everything from
        the tag onward is discarded;
      * plain text with no tags is returned unchanged.

    Some chat templates *open* the think block in the prompt, so the generated
    text starts mid-trace with no leading ``<think>`` tag but does contain a
    closing ``</think>``. Handle that by keeping only text after the last
    ``</think>`` when a close is present.
    """

    # If a closing tag exists, the real answer is whatever follows the last one.
    lower_full = text.lower()
    if "</think>" in lower_full:
        text = text[lower_full.rindex("</think>") + len("</think>"):]
    # Remove any remaining balanced blocks (defensive).
    text = _THINK_RE.sub(" ", text)
    lower = text.lower()
    if "<think>" in lower:  # truncated / unclosed reasoning block, no answer
        text = text[: lower.index("<think>")]
    text = text.strip()

    # Some chat templates open the <think> block in the *prompt*, so a truncated
    # generation arrives as a bare reasoning trace with NO tag at all (it starts
    # mid-thought). Detect that shape and treat it as "no answer": a leading
    # reasoning preamble with no closing tag anywhere in the raw generation.
    if "</think>" not in lower_full and _looks_like_reasoning_preamble(text):
        return ""
    return text


_REASONING_PREAMBLE_RE = re.compile(
    r"^\s*(thinking process|let me think|okay,? let'?s|first,? (?:i|let)|"
    r"the (?:user|question) (?:is )?ask|i need to|analyze the request|reasoning:)",
    flags=re.IGNORECASE,
)


def _looks_like_reasoning_preamble(text: str) -> bool:
    """Heuristic: text is an un-tagged reasoning trace, not a final answer.

    Fires only on the characteristic opening of a chain-of-thought preamble. Real
    trivia answers are short entities ("Kilimanjaro", "Japan") and never match.
    """

    return bool(_REASONING_PREAMBLE_RE.match(text))


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a contiguous token run inside ``haystack``."""

    if not needle:
        return False
    n, m = len(haystack), len(needle)
    return any(haystack[i : i + m] == needle for i in range(n - m + 1))


def alias_containment_loss(answer: str, reference: str) -> float:
    """0/1 loss: 0 if any normalized gold alias is contained in the final answer.

    This is the standard open-domain QA answer-string match. It is robust to the
    model wrapping the answer in a full sentence ("The answer was X.") and does not
    hand out spurious partial credit for incidental word overlap the way token-F1
    does. Reasoning traces are stripped before matching.
    """

    ans_tokens = normalize_text(strip_reasoning(answer)).split()
    aliases = [a for a in reference.split(ALIAS_SEP) if a]
    for alias in aliases:
        if _contains_subsequence(ans_tokens, normalize_text(alias).split()):
            return 0.0
    return 1.0


def alias_f1_loss(answer: str, reference: str) -> float:
    """Alias-aware ``1 - best token-F1`` loss (continuous), reasoning stripped."""

    ans = strip_reasoning(answer)
    aliases = [a for a in reference.split(ALIAS_SEP) if a]
    if not aliases:
        return 1.0
    return min(token_f1_loss(ans, alias) for alias in aliases)


GRADERS = {"containment": alias_containment_loss, "f1": alias_f1_loss}


def build_qa_prompt(tokenizer, question: str, *, thinking: bool) -> str:
    """Render a QA prompt, using the model's chat template when available.

    When ``thinking`` is False, pass ``enable_thinking=False`` so Qwen3.x models
    do not open a ``<think>`` block (they answer directly). Tokenizers that do not
    accept the flag (e.g. Qwen2.5) fall back gracefully.
    """

    instruction = (
        "Answer the trivia question with the shortest correct answer only "
        "(a name, entity, or phrase). Do not explain.\n\n"
        f"Question: {question}\nAnswer:"
    )
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": instruction}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            # Older/other tokenizers don't accept enable_thinking.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return instruction


def triviaqa_reference(answer_field: dict) -> str:
    """Pack the gold value + aliases into one ALIAS_SEP-delimited string."""

    golds = [answer_field.get("value", "")]
    golds.extend(answer_field.get("aliases", []) or [])
    seen: dict[str, None] = {}
    for g in golds:
        g = (g or "").strip()
        if g and g not in seen:
            seen[g] = None
    return ALIAS_SEP.join(seen.keys()) if seen else ""


def _resolve_device_map(requested: str | None) -> str | None:
    """Resolve requested device map, enforcing CUDA when explicitly requested."""

    if requested is None:
        return None

    req = requested.strip().lower()
    if req == "cuda":
        try:
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError(
                "Requested --device-map cuda, but CUDA is unavailable. "
                "Update your NVIDIA driver or install a PyTorch build compatible "
                "with your driver/runtime."
            )
        except Exception as exc:
            raise RuntimeError(
                "Requested --device-map cuda, but CUDA initialization failed. "
                "Update your NVIDIA driver or install a PyTorch build compatible "
                f"with your driver/runtime. Original error: {exc}"
            )

    return requested


def _resolve_torch_dtype(requested: str) -> str | torch.dtype:
    """Map CLI dtype name to a torch dtype accepted by from_pretrained."""

    name = requested.strip().lower()
    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset-config", default="rc.nocontext",
                        help="TriviaQA config, e.g. 'rc.nocontext' or 'unfiltered.nocontext'.")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="With --thinking off, 64 is plenty. With --thinking on, use >=512 "
                             "so the <think> block can close.")
    parser.add_argument("--thinking", choices=["off", "on"], default="off",
                        help="off (default): suppress <think> for direct answers (best for TriviaQA). "
                             "on: allow reasoning traces (requires large --max-new-tokens).")
    parser.add_argument("--grader", choices=list(GRADERS), default="containment",
                        help="containment = 0/1 answer-string match (recommended); "
                             "f1 = 1 - best token-F1 (continuous).")
    parser.add_argument("--semantic-samples", type=int, default=3,
                        help="Repeated samples per prompt for semantic-entropy features. "
                             "Each sample is a full generation -- expensive for large models.")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                        help="Weights dtype for model loading. Keep 'auto' unless you need to force one.")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Enable 8-bit bitsandbytes quantization to reduce VRAM.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Enable 4-bit bitsandbytes quantization (recommended for 27B on 32GB GPUs).")
    parser.add_argument("--max-memory-gpu", default=None,
                        help="Per-GPU memory cap, e.g. '30GiB'. Requires --device-map auto.")
    parser.add_argument("--max-memory-cpu", default=None,
                        help="CPU RAM cap for offload when using --device-map auto, e.g. '120GiB'.")
    parser.add_argument("--offload-folder", default=None,
                        help="Folder for disk offload when model shards do not fit in GPU/CPU RAM.")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--output", default="llm_deup_triviaqa_results.json")
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one quantization mode: --load-in-4bit or --load-in-8bit.")

    thinking = args.thinking == "on"
    if thinking and args.max_new_tokens < 256:
        print(f"[warning] --thinking on with --max-new-tokens {args.max_new_tokens}: the <think> "
              "block likely won't close and every answer will be truncated. Use >=512.")

    loss_fn = GRADERS[args.grader]

    ds = load_dataset("mandarjoshi/trivia_qa", args.dataset_config)
    train_split = ds["train"].select(range(args.train_size))
    val_key = "validation" if "validation" in ds else "test"
    test_split = ds[val_key].select(range(args.test_size))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    resolved_device_map = _resolve_device_map(args.device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    model_kwargs["low_cpu_mem_usage"] = True
    model_kwargs["torch_dtype"] = _resolve_torch_dtype(args.torch_dtype)

    if args.load_in_4bit or args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

    if args.max_memory_gpu or args.max_memory_cpu:
        if str(resolved_device_map).lower() != "auto":
            raise ValueError("--max-memory-gpu/--max-memory-cpu require --device-map auto.")
        max_memory: dict[int | str, str] = {}
        if args.max_memory_gpu is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("--max-memory-gpu was set but CUDA is unavailable.")
            for idx in range(torch.cuda.device_count()):
                max_memory[idx] = args.max_memory_gpu
        if args.max_memory_cpu is not None:
            max_memory["cpu"] = args.max_memory_cpu
        model_kwargs["max_memory"] = max_memory

    if args.offload_folder:
        model_kwargs["offload_folder"] = args.offload_folder

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA OOM while loading model. For large models (e.g. 27B) on a 32GB GPU, "
            "use quantized loading and auto device mapping, e.g.:\n"
            "  --device-map auto --load-in-4bit --max-memory-gpu 30GiB --max-memory-cpu 120GiB"
        ) from exc

    if resolved_device_map is None:
        model.eval()

    train_prompts = [build_qa_prompt(tokenizer, r["question"], thinking=thinking) for r in train_split]
    train_refs = [triviaqa_reference(r["answer"]) for r in train_split]
    test_prompts = [build_qa_prompt(tokenizer, r["question"], thinking=thinking) for r in test_split]
    test_refs = [triviaqa_reference(r["answer"]) for r in test_split]

    generation_config = HFGenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    # Qwen3.x recommended sampling for the semantic-entropy rollouts.
    sample_generation_config = HFGenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95,
    )

    deup = LLMDEUPRiskEstimator(
        model,
        tokenizer,
        generation_config=generation_config,
        sample_generation_config=sample_generation_config,
        n_semantic_samples=args.semantic_samples,
        # For 0/1 containment loss, no transform is cleaner; f1 benefits from log.
        target_transform="none" if args.grader == "containment" else "log",
    )

    print(f"Fitting DEUP error predictor on frozen-LLM {args.grader} losses "
          f"(thinking={'on' if thinking else 'off'})...")
    deup.fit(train_prompts, train_refs, loss_fn=loss_fn, show_progress=True)

    from tqdm.auto import tqdm

    risks, errors, rows = [], [], []
    empty_answers = 0
    for prompt, ref in tqdm(list(zip(test_prompts, test_refs)), desc="DEUP eval"):
        pred = deup.predict_one(prompt)
        final = strip_reasoning(pred.answer)
        if not final:
            empty_answers += 1
        err = loss_fn(pred.answer, ref)
        risks.append(pred.predicted_risk)
        errors.append(err)
        rows.append({
            "prompt": prompt,
            "gold_aliases": ref.split(ALIAS_SEP),
            "answer_raw": pred.answer,
            "answer_final": final,
            "loss": err,
            "predicted_risk": pred.predicted_risk,
            "epistemic_uncertainty": pred.epistemic_uncertainty,
            "semantic_entropy": pred.features.get("semantic_entropy"),
        })

    risk = np.asarray(risks, dtype=float)
    err = np.asarray(errors, dtype=float)
    incorrect = (err > 0.5).astype(int)  # 1 = wrong

    # --- Guardrail: catch the "model never answered" failure mode explicitly ---
    if empty_answers > 0.5 * len(rows):
        print(f"\n[WARNING] {empty_answers}/{len(rows)} answers were EMPTY after stripping "
              "the reasoning trace. The model was cut off before producing a final answer.")
        if thinking:
            print("          You have --thinking on: raise --max-new-tokens to >=512.")
        else:
            print("          Inspect 'answer_final' in the output JSON; if it still contains "
                  "'Thinking Process', this tokenizer ignored enable_thinking=False — "
                  "use --thinking on with --max-new-tokens 512 instead.")

    # risk_error_spearman is undefined if risk is constant; guard against the warning.
    if np.std(risk) == 0 or np.std(err) == 0:
        spearman = float("nan")
        print("[note] risk or error is constant across all rows -> Spearman undefined "
              "(this usually means every answer was scored the same, i.e. the run is degenerate).")
    else:
        spearman = float(spearmanr(risk, err).statistic)

    metrics = {
        "grader": args.grader,
        "thinking": "on" if thinking else "off",
        "n_empty_answers": empty_answers,
        "mean_loss": float(np.mean(err)),
        "accuracy": float(np.mean(1.0 - (err > 0.5))),
        "mean_predicted_risk": float(np.mean(risk)),
        "risk_error_spearman": spearman,
    }
    if len(np.unique(incorrect)) == 2:
        metrics["error_detection_auroc"] = float(roc_auc_score(incorrect, risk))
        metrics["error_detection_auprc"] = float(average_precision_score(incorrect, risk))
    else:
        metrics["error_detection_auroc"] = None
        metrics["error_detection_auprc"] = None
        print("[note] Only one outcome class present (all right or all wrong) -> "
              "AUROC/AUPRC undefined. For DEUP to learn, you need a MIX of right and wrong.")

    output = {"model_id": args.model_id, "dataset": f"trivia_qa/{args.dataset_config}",
              "metrics": metrics, "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()