"""Scenario A: DEUP risk prediction for a frozen HF LLM on open-domain QA (TriviaQA).

Unlike the GSM8K example (single numeric answer, hard 0/1 exact-match loss), this
script targets *free-form textual* answers. Correctness is graded with a robust
**containment exact-match** (does any gold alias appear in the model's final
answer), and the DEUP error predictor g(x) is trained to regress that loss from
frozen-LLM confidence features. Repeated-sampling semantic entropy is enabled as
an extra feature.

Reasoning-model support
-----------------------
Models like ``deepreinforce-ai/Ornith-1.0-9B`` (Qwen3.5-based) open the assistant
turn with a ``<think> ... </think>`` trace before the final answer. Two things
follow:

  * The grader must strip the reasoning trace and score only the post-``</think>``
    answer (``strip_reasoning``). Grading the raw generation would score the whole
    chain-of-thought.
  * ``--max-new-tokens`` must be large enough to *close* the think block, or the
    model never reaches its final answer. 32 is fine for a plain instruct model but
    will truncate a reasoning model mid-thought. Default here is 512.

The token-confidence features (log-probs, entropy, margins) are still computed
over the *full* generation, including the think trace -- that is intentional:
uncertainty expressed during reasoning is legitimate signal for g(x).

Why an instruct/reasoning model, not distilgpt2
-----------------------------------------------
DEUP's error predictor needs *variance* in the target: some answers right, some
wrong. A base LM that is wrong on everything gives g(x) nothing to learn.

Install extras:
    pip install -e ".[llm]"
    pip install datasets tqdm scipy

Research run (reasoning model):

uv run python examples/llm_scenario_triviaqa.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device-map cuda --train-size 100 --test-size 50 --max-new-tokens 64 --semantic-samples 3

uv run python examples/llm_scenario_triviaqa.py --model-id deepreinforce-ai/Ornith-1.0-9B --device-map cuda --train-size 100 --test-size 50 --max-new-tokens 512 --semantic-samples 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """

    text = _THINK_RE.sub(" ", text)
    lower = text.lower()
    if "<think>" in lower:  # truncated / unclosed reasoning block
        text = text[: lower.index("<think>")]
    return text.strip()


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


def build_qa_prompt(tokenizer, question: str) -> str:
    """Render a QA prompt, using the model's chat template when available."""

    instruction = (
        "Answer the trivia question with the shortest correct answer only "
        "(a name, entity, or phrase). Do not explain.\n\n"
        f"Question: {question}\nAnswer:"
    )
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="deepreinforce-ai/Ornith-1.0-9B")
    parser.add_argument("--dataset-config", default="rc.nocontext",
                        help="TriviaQA config, e.g. 'rc.nocontext' or 'unfiltered.nocontext'.")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Reasoning models need room to close <think>. Use >=256.")
    parser.add_argument("--grader", choices=list(GRADERS), default="containment",
                        help="containment = 0/1 answer-string match (recommended); "
                             "f1 = 1 - best token-F1 (continuous).")
    parser.add_argument("--semantic-samples", type=int, default=3,
                        help="Repeated samples per prompt for semantic-entropy features. "
                             "Each sample is a full generation -- expensive for reasoning models.")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--output", default="llm_deup_triviaqa_results.json")
    args = parser.parse_args()

    loss_fn = GRADERS[args.grader]

    ds = load_dataset("mandarjoshi/trivia_qa", args.dataset_config)
    train_split = ds["train"].select(range(args.train_size))
    val_key = "validation" if "validation" in ds else "test"
    test_split = ds[val_key].select(range(args.test_size))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.device_map is None:
        model.eval()

    train_prompts = [build_qa_prompt(tokenizer, r["question"]) for r in train_split]
    train_refs = [triviaqa_reference(r["answer"]) for r in train_split]
    test_prompts = [build_qa_prompt(tokenizer, r["question"]) for r in test_split]
    test_refs = [triviaqa_reference(r["answer"]) for r in test_split]

    generation_config = HFGenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    # Ornith / Qwen3.5 recommended sampling for the semantic-entropy rollouts.
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

    print(f"Fitting DEUP error predictor on frozen-LLM {args.grader} losses...")
    deup.fit(train_prompts, train_refs, loss_fn=loss_fn, show_progress=True)

    from tqdm.auto import tqdm

    risks, errors, rows = [], [], []
    for prompt, ref in tqdm(list(zip(test_prompts, test_refs)), desc="DEUP eval"):
        pred = deup.predict_one(prompt)
        err = loss_fn(pred.answer, ref)
        risks.append(pred.predicted_risk)
        errors.append(err)
        rows.append({
            "prompt": prompt,
            "gold_aliases": ref.split(ALIAS_SEP),
            "answer_raw": pred.answer,
            "answer_final": strip_reasoning(pred.answer),
            "loss": err,
            "predicted_risk": pred.predicted_risk,
            "epistemic_uncertainty": pred.epistemic_uncertainty,
            "semantic_entropy": pred.features.get("semantic_entropy"),
        })

    risk = np.asarray(risks, dtype=float)
    err = np.asarray(errors, dtype=float)
    incorrect = (err > 0.5).astype(int)  # 1 = wrong

    metrics = {
        "grader": args.grader,
        "mean_loss": float(np.mean(err)),
        "accuracy": float(np.mean(1.0 - (err > 0.5))),
        "mean_predicted_risk": float(np.mean(risk)),
        # Primary continuous metric: does predicted risk rank realized error?
        "risk_error_spearman": float(spearmanr(risk, err).statistic),
    }
    if len(np.unique(incorrect)) == 2:
        metrics["error_detection_auroc"] = float(roc_auc_score(incorrect, risk))
        metrics["error_detection_auprc"] = float(average_precision_score(incorrect, risk))
    else:
        metrics["error_detection_auroc"] = None
        metrics["error_detection_auprc"] = None

    output = {"model_id": args.model_id, "dataset": f"trivia_qa/{args.dataset_config}",
              "metrics": metrics, "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()