"""Run all deup benchmarks and write a combined summary."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import DEFAULT_SEED, write_json  # noqa: E402


def main() -> None:
    modules = [
        "benchmarks.run_regression_benchmark",
        "benchmarks.run_n_sweep",
        "benchmarks.run_cifar_proxy",
        "benchmarks.run_finance_walkforward",
    ]
    for mod_name in modules:
        print(f"\n>>> {mod_name}")
        importlib.import_module(mod_name).main()

    summary = {
        "seed": DEFAULT_SEED,
        "benchmarks": [
            "regression_benchmark.json",
            "n_sweep.json",
            "cifar_proxy.json",
            "finance_walkforward.json",
        ],
        "figures": ["n_sweep.png"],
    }
    path = write_json("summary.json", summary)
    print(f"\n=== All benchmarks complete ===\nWrote {path}")


if __name__ == "__main__":
    main()
