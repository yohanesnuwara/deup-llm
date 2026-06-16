"""Domain-specific DEUP presets.

Each module is a thin configuration layer over the core estimators — no duplicated
orchestration logic. See ``ARCHITECTURE.md`` use-case map.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from deup.domains.llm import (
        HFGenerationConfig,
        LLMDEUPRiskEstimator,
        LLMErrorDataset,
        LLMTokenFeatureExtractor,
    )
except Exception:  # pragma: no cover - optional LLM dependencies may be absent
    pass
else:
    __all__ += [
        "HFGenerationConfig",
        "LLMDEUPRiskEstimator",
        "LLMErrorDataset",
        "LLMTokenFeatureExtractor",
    ]
