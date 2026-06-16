from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from deup.domains.llm import LLMDEUPRiskEstimator, LLMErrorDataset, LLMGenerationResult


class _DummyTokenExtractor:
    def transform(self, scores: Sequence[Any], generated_ids: Sequence[Any]) -> dict[str, float]:
        return {
            "num_scores": float(len(scores)),
            "num_generated_ids": float(len(generated_ids)),
        }


def _build_estimator() -> LLMDEUPRiskEstimator:
    return LLMDEUPRiskEstimator(
        model=object(),
        tokenizer=object(),
        token_feature_extractor=_DummyTokenExtractor(),
        target_transform="none",
    )


def test_collect_returns_cached_llm_error_dataset(
    monkeypatch: Any,
) -> None:
    estimator = _build_estimator()
    prompts = ["p0", "p1", "p2"]
    references = ["a0", "a1", "a2"]
    answers_by_prompt = {"p0": "a0", "p1": "wrong", "p2": "a2"}

    def _fake_generate(self: Any, prompt: str, **_: Any) -> LLMGenerationResult:
        answer = answers_by_prompt[prompt]
        generated_ids = [1, 2] if answer == "wrong" else [1]
        return LLMGenerationResult(
            prompt=prompt,
            answer=answer,
            scores=[0.0],
            generated_ids=generated_ids,
        )

    monkeypatch.setattr(LLMDEUPRiskEstimator, "generate_with_scores", _fake_generate)

    dataset = estimator.collect(prompts, references)

    assert isinstance(dataset, LLMErrorDataset)
    assert dataset.answers == ["a0", "wrong", "a2"]
    np.testing.assert_array_equal(dataset.errors, np.asarray([0.0, 1.0, 0.0]))
    assert dataset.feature_matrix.shape == (3, len(dataset.feature_names))
    assert set(dataset.feature_names) >= {
        "num_scores",
        "num_generated_ids",
        "prompt_chars",
        "prompt_words",
        "answer_chars",
        "answer_words",
    }


def test_fit_from_collected_reuses_cached_llm_pass(monkeypatch: Any) -> None:
    estimator = _build_estimator()
    prompts = ["p0", "p1", "p2", "p3"]
    references = ["a0", "a1", "a2", "a3"]
    answers_by_prompt = {"p0": "a0", "p1": "a1", "p2": "wrong", "p3": "a3"}
    call_count = 0

    def _fake_generate(self: Any, prompt: str, **_: Any) -> LLMGenerationResult:
        nonlocal call_count
        call_count += 1
        return LLMGenerationResult(
            prompt=prompt,
            answer=answers_by_prompt[prompt],
            scores=[0.0],
            generated_ids=[1],
        )

    monkeypatch.setattr(LLMDEUPRiskEstimator, "generate_with_scores", _fake_generate)

    dataset = estimator.collect(prompts, references)
    assert call_count == len(prompts)

    estimator.fit_from_collected(dataset)
    assert call_count == len(prompts)
    assert estimator.collected_dataset_ is dataset
    assert estimator.training_answers_ == dataset.answers
    np.testing.assert_allclose(estimator.training_feature_matrix_, dataset.feature_matrix)


def test_fit_delegates_to_collect_and_fit_from_collected(monkeypatch: Any) -> None:
    estimator = _build_estimator()
    prompts = ["p0", "p1"]
    references = ["a0", "a1"]

    collected = LLMErrorDataset(
        feature_dicts=[{"f": 1.0}, {"f": 2.0}],
        feature_matrix=np.asarray([[1.0], [2.0]], dtype=float),
        feature_names=["f"],
        answers=["a0", "a1"],
        errors=np.asarray([0.0, 1.0], dtype=float),
    )

    calls: list[str] = []

    def _fake_collect(self: Any, *args: Any, **kwargs: Any) -> LLMErrorDataset:
        calls.append("collect")
        assert args[0] == prompts
        assert args[1] == references
        assert kwargs["show_progress"] is False
        return collected

    def _fake_fit_from_collected(self: Any, dataset: LLMErrorDataset) -> LLMDEUPRiskEstimator:
        calls.append("fit_from_collected")
        assert dataset is collected
        return self

    monkeypatch.setattr(LLMDEUPRiskEstimator, "collect", _fake_collect)
    monkeypatch.setattr(LLMDEUPRiskEstimator, "fit_from_collected", _fake_fit_from_collected)

    returned = estimator.fit(prompts, references)

    assert returned is estimator
    assert calls == ["collect", "fit_from_collected"]
