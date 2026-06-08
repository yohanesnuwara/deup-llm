"""Scenario-A DEUP utilities for frozen Hugging Face causal language models.

This module adapts the package's DEUP core to LLMs without fine-tuning the base
model. A frozen LLM generates answers and token-level scores; those scores are
converted into a tabular feature vector ``z``. A DEUP error predictor is then
trained on observed benchmark losses ``l(reference, answer)``.

Important interpretation
------------------------
The fitted score is a predicted task risk ``g(z)``. It becomes a DEUP-style
estimate of epistemic uncertainty only after subtracting an aleatoric estimate
``a(x)``. When no aleatoric estimator is supplied, the score is the conservative
proxy described in Lahlou et al. (2023): ``u(x) = g(x)``.

The imports of ``torch`` and ``transformers`` are intentionally lazy so that the
base ``deup`` package remains usable without LLM dependencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
import string
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin

from deup.core.decompose import decompose_epistemic
from deup.core.error_estimator import ErrorEstimator

TextLossFn = Callable[[str, str], float]
FeatureDict = dict[str, float]


@dataclass(frozen=True)
class HFGenerationConfig:
    """Generation settings used by :class:`LLMDEUPRiskEstimator`.

    Parameters are a small, stable subset of Hugging Face ``generate`` arguments.
    Additional keyword arguments can still be passed directly to
    :meth:`LLMDEUPRiskEstimator.generate_with_scores`.
    """

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    num_return_sequences: int = 1


@dataclass(frozen=True)
class LLMGenerationResult:
    """Result returned by ``generate_with_scores``."""

    prompt: str
    answer: str
    scores: Sequence[Any]
    generated_ids: Any


@dataclass(frozen=True)
class LLMDEUPPrediction:
    """Inference result containing answer, predicted risk, and DEUP signal."""

    answer: str
    predicted_risk: float
    epistemic_uncertainty: float
    features: FeatureDict


def normalize_text(text: str) -> str:
    """Normalize free-form answers for exact-match style evaluation.

    This is intentionally lightweight and dependency-free. It removes punctuation,
    articles, and duplicate whitespace, then lowercases the answer.
    """

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def exact_match_loss(answer: str, reference: str) -> float:
    """Return ``0.0`` when normalized strings match, else ``1.0``."""

    return 0.0 if normalize_text(answer) == normalize_text(reference) else 1.0


def token_f1_loss(answer: str, reference: str) -> float:
    """Return ``1 - token F1`` for open QA-style answers."""

    pred = normalize_text(answer).split()
    gold = normalize_text(reference).split()
    if not pred and not gold:
        return 0.0
    if not pred or not gold:
        return 1.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 1.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(1.0 - f1)


def extract_last_number(text: str) -> str | None:
    """Extract the last integer/decimal in a generated answer.

    Useful for GSM8K-like tasks where the final numeric answer is often the last
    number. Commas are ignored. Returns ``None`` if no number is found.
    """

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numeric_exact_loss(answer: str, reference: str, *, atol: float = 1e-9) -> float:
    """Return 0/1 loss after comparing the last numeric values in two strings."""

    a = extract_last_number(answer)
    b = extract_last_number(reference)
    if a is None or b is None:
        return exact_match_loss(answer, reference)
    try:
        return 0.0 if math.isclose(float(a), float(b), abs_tol=atol, rel_tol=0.0) else 1.0
    except ValueError:
        return 1.0


def multiple_choice_loss(answer: str, reference: str) -> float:
    """0/1 loss for generated multiple-choice letters.

    The first standalone A/B/C/D/E letter in ``answer`` is compared with the first
    such letter in ``reference``. If no letter is found, falls back to exact match.
    """

    pattern = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)
    ans = pattern.search(answer)
    ref = pattern.search(reference)
    if ans is None or ref is None:
        return exact_match_loss(answer, reference)
    return 0.0 if ans.group(1).upper() == ref.group(1).upper() else 1.0


def semantic_entropy_from_answers(
    answers: Sequence[str],
    *,
    normalizer: Callable[[str], str] = normalize_text,
) -> FeatureDict:
    """Compute simple semantic-instability features from repeated generations.

    This dependency-free implementation clusters answers by normalized string.
    For numeric tasks, pass a normalizer that extracts final numbers. For a richer
    setup, replace this with an NLI or embedding-based clustering function.
    """

    if not answers:
        return {
            "semantic_entropy": 0.0,
            "semantic_num_clusters": 0.0,
            "semantic_top_cluster_frac": 0.0,
        }
    keys = [normalizer(a) for a in answers]
    counts = Counter(keys)
    total = float(len(keys))
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return {
        "semantic_entropy": float(entropy),
        "semantic_entropy_norm": float(entropy / max_entropy),
        "semantic_num_clusters": float(len(counts)),
        "semantic_top_cluster_frac": float(max(probs)),
    }


class LLMTokenFeatureExtractor:
    """Convert Hugging Face generation scores into numerical DEUP features."""

    def __init__(self, *, entropy_top_k: int | None = 512) -> None:
        self.entropy_top_k = entropy_top_k

    def transform(self, scores: Sequence[Any], generated_ids: Any) -> FeatureDict:
        """Return token-level features from ``generate(..., output_scores=True)``.

        Full-vocabulary entropy can be expensive for large vocabularies. By default
        this computes entropy on the top 512 logits and renormalizes over that set.
        Set ``entropy_top_k=None`` for exact full-vocabulary entropy.
        """

        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError('LLM utilities require torch: pip install "deup[llm]"') from exc

        if len(scores) == 0:
            return self._empty_features()

        ids = generated_ids.detach().cpu() if hasattr(generated_ids, "detach") else generated_ids
        token_logprobs: list[float] = []
        entropies: list[float] = []
        top1_probs: list[float] = []
        margins: list[float] = []
        ranks: list[float] = []

        n_steps = min(len(scores), len(ids))
        for t in range(n_steps):
            logits_t = scores[t]
            if logits_t.ndim == 2:
                logits_t = logits_t[0]
            token_id = int(ids[t])
            log_probs = F.log_softmax(logits_t, dim=-1)
            probs = torch.exp(log_probs)

            token_logprobs.append(float(log_probs[token_id].detach().cpu()))
            top2 = torch.topk(probs, k=2).values
            top1_probs.append(float(top2[0].detach().cpu()))
            margins.append(float((top2[0] - top2[1]).detach().cpu()))

            # 1-based rank of the generated token by model probability.
            token_prob = probs[token_id]
            rank = int(torch.sum(probs > token_prob).detach().cpu()) + 1
            ranks.append(float(rank))

            if self.entropy_top_k is None:
                entropy = -(probs * log_probs).sum()
            else:
                k = min(self.entropy_top_k, logits_t.shape[-1])
                vals = torch.topk(logits_t, k=k).values
                top_log_probs = F.log_softmax(vals, dim=-1)
                top_probs = torch.exp(top_log_probs)
                entropy = -(top_probs * top_log_probs).sum()
            entropies.append(float(entropy.detach().cpu()))

        if not token_logprobs:
            return self._empty_features()

        arr_logp = np.asarray(token_logprobs, dtype=float)
        arr_entropy = np.asarray(entropies, dtype=float)
        arr_margin = np.asarray(margins, dtype=float)
        arr_top1 = np.asarray(top1_probs, dtype=float)
        arr_rank = np.asarray(ranks, dtype=float)
        return {
            "mean_logprob": float(np.mean(arr_logp)),
            "min_logprob": float(np.min(arr_logp)),
            "sequence_nll": float(-np.mean(arr_logp)),
            "mean_entropy": float(np.mean(arr_entropy)),
            "max_entropy": float(np.max(arr_entropy)),
            "mean_top1_prob": float(np.mean(arr_top1)),
            "mean_margin": float(np.mean(arr_margin)),
            "min_margin": float(np.min(arr_margin)),
            "mean_generated_token_rank": float(np.mean(arr_rank)),
            "max_generated_token_rank": float(np.max(arr_rank)),
            "num_generated_tokens": float(len(arr_logp)),
        }

    @staticmethod
    def _empty_features() -> FeatureDict:
        return {
            "mean_logprob": 0.0,
            "min_logprob": 0.0,
            "sequence_nll": 0.0,
            "mean_entropy": 0.0,
            "max_entropy": 0.0,
            "mean_top1_prob": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "mean_generated_token_rank": 0.0,
            "max_generated_token_rank": 0.0,
            "num_generated_tokens": 0.0,
        }


class LLMDEUPRiskEstimator(BaseEstimator, RegressorMixin):
    """Scenario-A DEUP wrapper for a frozen Hugging Face causal LLM.

    Parameters
    ----------
    model, tokenizer:
        Hugging Face causal LM and tokenizer. The model is not fine-tuned.
    error_model:
        Optional sklearn-style regressor passed into :class:`ErrorEstimator`.
    token_feature_extractor:
        Extractor for logits/scores. Defaults to :class:`LLMTokenFeatureExtractor`.
    target_transform:
        Error target transform used by :class:`ErrorEstimator`.
    aleatoric_estimator:
        Optional callable ``aleatoric_estimator(prompt) -> float``. If absent,
        predicted risk is also returned as a conservative DEUP proxy.
    generation_config:
        Default generation settings for the main answer.
    sample_generation_config:
        Generation settings for repeated samples used by semantic entropy.
    n_semantic_samples:
        Number of repeated samples for semantic entropy. Set to 0 to disable.
    semantic_normalizer:
        Normalizer used to cluster repeated generations.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        error_model: Any = None,
        token_feature_extractor: LLMTokenFeatureExtractor | None = None,
        target_transform: Literal["none", "log", "asinh"] = "none",
        aleatoric_estimator: Callable[[str], float] | None = None,
        generation_config: HFGenerationConfig | None = None,
        sample_generation_config: HFGenerationConfig | None = None,
        n_semantic_samples: int = 0,
        semantic_normalizer: Callable[[str], str] = normalize_text,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.error_model = error_model
        self.token_feature_extractor = token_feature_extractor
        self.target_transform = target_transform
        self.aleatoric_estimator = aleatoric_estimator
        self.generation_config = generation_config
        self.sample_generation_config = sample_generation_config
        self.n_semantic_samples = n_semantic_samples
        self.semantic_normalizer = semantic_normalizer

    def generate_with_scores(
        self,
        prompt: str,
        *,
        config: HFGenerationConfig | None = None,
        **generate_kwargs: Any,
    ) -> LLMGenerationResult:
        """Generate one answer and return Hugging Face per-step scores."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError('LLM utilities require torch: pip install "deup[llm]"') from exc

        cfg = config or self.generation_config or HFGenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "num_return_sequences": cfg.num_return_sequences,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        kwargs.update(generate_kwargs)

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                kwargs.setdefault("pad_token_id", eos)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        sequence = outputs.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = sequence[prompt_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return LLMGenerationResult(
            prompt=prompt,
            answer=answer,
            scores=outputs.scores,
            generated_ids=generated_ids,
        )

    def extract_features(
        self,
        result: LLMGenerationResult,
        *,
        sampled_answers: Sequence[str] | None = None,
    ) -> FeatureDict:
        """Build one tabular DEUP feature vector from an LLM generation result."""

        extractor = self.token_feature_extractor or LLMTokenFeatureExtractor()
        feats = extractor.transform(result.scores, result.generated_ids)
        feats.update(
            {
                "prompt_chars": float(len(result.prompt)),
                "prompt_words": float(len(result.prompt.split())),
                "answer_chars": float(len(result.answer)),
                "answer_words": float(len(result.answer.split())),
            }
        )
        if sampled_answers is not None:
            feats.update(
                semantic_entropy_from_answers(
                    sampled_answers,
                    normalizer=self.semantic_normalizer,
                )
            )
        return feats

    @staticmethod
    def dicts_to_matrix(feature_dicts: Sequence[Mapping[str, float]]) -> tuple[npt.NDArray[Any], list[str]]:
        """Convert feature dictionaries to a stable numeric matrix."""

        names = sorted({k for row in feature_dicts for k in row.keys()})
        matrix = np.asarray([[float(row.get(name, 0.0)) for name in names] for row in feature_dicts])
        return matrix, names

    def make_feature_records(
        self,
        prompts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> tuple[list[FeatureDict], list[str]]:
        """Generate answers and feature dictionaries for prompts."""

        feature_dicts: list[FeatureDict] = []
        answers: list[str] = []
        iterator: Sequence[str] | Any = prompts
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(prompts, desc="LLM-DEUP feature extraction")
            except ImportError:
                iterator = prompts

        sample_cfg = self.sample_generation_config or HFGenerationConfig(
            max_new_tokens=(self.generation_config or HFGenerationConfig()).max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        for prompt in iterator:
            result = self.generate_with_scores(prompt)
            samples: list[str] | None = None
            if self.n_semantic_samples > 0:
                samples = [
                    self.generate_with_scores(prompt, config=sample_cfg).answer
                    for _ in range(self.n_semantic_samples)
                ]
            feature_dicts.append(self.extract_features(result, sampled_answers=samples))
            answers.append(result.answer)
        return feature_dicts, answers

    def fit(
        self,
        prompts: Sequence[str],
        references: Sequence[str],
        *,
        loss_fn: TextLossFn = exact_match_loss,
        show_progress: bool = False,
    ) -> LLMDEUPRiskEstimator:
        """Generate benchmark answers, compute losses, and fit the DEUP error predictor."""

        if len(prompts) != len(references):
            raise ValueError("prompts and references must have the same length")
        feature_dicts, answers = self.make_feature_records(prompts, show_progress=show_progress)
        errors = np.asarray(
            [float(loss_fn(answer, reference)) for answer, reference in zip(answers, references)],
            dtype=float,
        )
        X, names = self.dicts_to_matrix(feature_dicts)
        self.feature_names_ = names
        self.training_answers_ = answers
        self.training_errors_ = errors
        self.error_estimator_ = ErrorEstimator(
            model=self.error_model,
            features=None,
            target_transform=self.target_transform,
        )
        self.error_estimator_.fit(X, errors)
        return self

    def predict_risk_from_features(self, features: Mapping[str, float]) -> float:
        """Predict risk from an already-computed feature dictionary."""

        if not hasattr(self, "error_estimator_"):
            raise RuntimeError("fit must be called before predict")
        x = np.asarray([[float(features.get(name, 0.0)) for name in self.feature_names_]])
        return float(self.error_estimator_.predict(x)[0])

    def predict_one(self, prompt: str) -> LLMDEUPPrediction:
        """Generate one answer and return answer, predicted risk, and DEUP signal."""

        result = self.generate_with_scores(prompt)
        samples: list[str] | None = None
        if self.n_semantic_samples > 0:
            sample_cfg = self.sample_generation_config or HFGenerationConfig(
                max_new_tokens=(self.generation_config or HFGenerationConfig()).max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            samples = [
                self.generate_with_scores(prompt, config=sample_cfg).answer
                for _ in range(self.n_semantic_samples)
            ]
        features = self.extract_features(result, sampled_answers=samples)
        risk = self.predict_risk_from_features(features)
        aleatoric = 0.0 if self.aleatoric_estimator is None else float(self.aleatoric_estimator(prompt))
        epistemic = float(decompose_epistemic([risk], [aleatoric])[0])
        return LLMDEUPPrediction(
            answer=result.answer,
            predicted_risk=risk,
            epistemic_uncertainty=epistemic,
            features=features,
        )

    def predict(self, prompts: Sequence[str]) -> npt.NDArray[Any]:
        """Return predicted risks for a sequence of prompts.

        This sklearn-compatible method returns risk only. Use :meth:`predict_one` if
        you also need the generated answers and feature dictionaries.
        """

        return np.asarray([self.predict_one(prompt).predicted_risk for prompt in prompts], dtype=float)
