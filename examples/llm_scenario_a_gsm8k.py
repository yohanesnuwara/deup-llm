"""Scenario A: DEUP risk prediction for a frozen Hugging Face LLM on GSM8K.

This example is intentionally small and CPU-friendly in structure, but LLM inference
can still be slow. Start with a small model such as ``sshleifer/tiny-gpt2`` for a
smoke test, then switch to an instruction-tuned model for real experiments.

Install extras:
    pip install -e ".[llm]"
    pip install datasets tqdm

Example smoke test:
    python examples/llm_scenario_a_gsm8k.py \
        --model-id sshleifer/tiny-gpt2 \
        --train-size 8 \
        --test-size 4 \
        --max-new-tokens 32

Research-oriented run:
    python examples/llm_scenario_a_gsm8k.py \
        --model-id Qwen/Qwen2.5-0.5B-Instruct \
        --train-size 300 \
        --test-size 100 \
        --max-new-tokens 256 \
        --semantic-samples 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from deup.domains.llm import (
    HFGenerationConfig,
    LLMDEUPRiskEstimator,
    extract_last_number,
    numeric_exact_loss,
)


def build_gsm8k_prompt(question: str, *, number_only_output: bool = False) -> str:
    if number_only_output:
        return (
            "Solve the math problem. Output only the final numeric answer. "
            "No words, no explanation, no punctuation.\n\n"
            f"Problem: {question}\n"
            "Answer:"
        )
    return (
        "Solve the math problem. Show brief reasoning, then end with 'Answer: <number>'.\n\n"
        f"Problem: {question}\n"
        "Solution:"
    )


def gsm8k_reference(answer_field: str) -> str:
    # GSM8K stores the final answer after ####.
    if "####" in answer_field:
        return answer_field.split("####", 1)[1].strip()
    value = extract_last_number(answer_field)
    return value if value is not None else answer_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--semantic-samples", type=int, default=0)
    parser.add_argument(
        "--number-only-output",
        action="store_true",
        help="Use a strict prompt that asks for only a final number.",
    )
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--output", default="llm_deup_gsm8k_results.json")
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main")
    train_split = dataset["train"].select(range(args.train_size))
    test_split = dataset["test"].select(range(args.test_size))

    train_prompts = [
        build_gsm8k_prompt(row["question"], number_only_output=args.number_only_output)
        for row in train_split
    ]
    train_refs = [gsm8k_reference(row["answer"]) for row in train_split]
    test_prompts = [
        build_gsm8k_prompt(row["question"], number_only_output=args.number_only_output)
        for row in test_split
    ]
    test_refs = [gsm8k_reference(row["answer"]) for row in test_split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model_kwargs = {}
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.device_map is None:
        model.eval()

    generation_config = HFGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    sample_generation_config = HFGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    deup = LLMDEUPRiskEstimator(
        model,
        tokenizer,
        generation_config=generation_config,
        sample_generation_config=sample_generation_config,
        n_semantic_samples=args.semantic_samples,
        target_transform="none",
    )

    print("Fitting DEUP error predictor on frozen-LLM benchmark errors...")
    deup.fit(train_prompts, train_refs, loss_fn=numeric_exact_loss, show_progress=True)

    risks = []
    errors = []
    rows = []
    for prompt, ref in zip(test_prompts, test_refs):
        pred = deup.predict_one(prompt)
        err = numeric_exact_loss(pred.answer, ref)
        risks.append(pred.predicted_risk)
        errors.append(err)
        rows.append(
            {
                "prompt": prompt,
                "reference": ref,
                "answer": pred.answer,
                "error": err,
                "predicted_risk": pred.predicted_risk,
                "epistemic_uncertainty": pred.epistemic_uncertainty,
            }
        )

    y = np.asarray(errors, dtype=int)
    risk = np.asarray(risks, dtype=float)
    metrics = {
        "mean_error": float(np.mean(y)),
        "mean_predicted_risk": float(np.mean(risk)),
    }
    if len(np.unique(y)) == 2:
        metrics["error_detection_auroc"] = float(roc_auc_score(y, risk))
        metrics["error_detection_auprc"] = float(average_precision_score(y, risk))
    else:
        metrics["error_detection_auroc"] = None
        metrics["error_detection_auprc"] = None

    output = {"model_id": args.model_id, "metrics": metrics, "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
