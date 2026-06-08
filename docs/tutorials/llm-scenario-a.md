# Scenario A: frozen Hugging Face LLM + DEUP risk prediction

This tutorial shows the lightweight LLM adaptation of DEUP where the base LLM is
**frozen**. The LLM generates answers and token-level scores. DEUP then trains a
secondary error predictor on benchmark losses.

The estimated value is best interpreted as predicted task risk:

\[
\hat R(x)=g_\phi(z(x, \hat y, \text{logits}))
\]

If an aleatoric estimator \(\hat A(x)\) is available, the DEUP-style epistemic
signal is:

\[
\hat U_{epi}(x)=\max(0, \hat R(x)-\hat A(x)).
\]

Without \(\hat A(x)\), the score is a conservative risk proxy.

## Install

```bash
pip install -e ".[llm]"
pip install datasets tqdm
```

## Smoke test

```bash
python examples/llm_scenario_a_gsm8k.py \
  --model-id sshleifer/tiny-gpt2 \
  --train-size 8 \
  --test-size 4 \
  --max-new-tokens 32
```

## More realistic run

```bash
python examples/llm_scenario_a_gsm8k.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --train-size 300 \
  --test-size 100 \
  --max-new-tokens 256 \
  --semantic-samples 3
```

## Main API

```python
from deup.domains.llm import HFGenerationConfig, LLMDEUPRiskEstimator, numeric_exact_loss

estimator = LLMDEUPRiskEstimator(
    model,
    tokenizer,
    generation_config=HFGenerationConfig(max_new_tokens=128, do_sample=False),
    n_semantic_samples=3,
    target_transform="none",
)

estimator.fit(train_prompts, train_references, loss_fn=numeric_exact_loss)
pred = estimator.predict_one("Solve: 2 + 2. Answer:")

print(pred.answer)
print(pred.predicted_risk)
print(pred.epistemic_uncertainty)
```

## Features currently extracted

The default `LLMTokenFeatureExtractor` computes:

- mean generated-token log probability
- sequence NLL
- mean and maximum generation entropy
- top-1 probability
- top-1/top-2 margin
- generated-token probability rank
- prompt and answer length
- optional semantic entropy from repeated generations

These are **features for DEUP**, not epistemic uncertainty by themselves.
