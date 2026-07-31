"""Compare per-KPI benchmark rounds without coupling to an agent implementation."""

from __future__ import annotations

import csv
from pathlib import Path

MAJOR_IMPROVEMENT_EXCLUSIVE = 0.03


def _load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline KPI quality gates not found: {path}")
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                rows[row["kpi"]] = {
                    "recall": float(row["recall"]),
                    "precision": float(row["precision"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def compare_kpi_quality(
    current: list[dict],
    *,
    baseline_dir: Path,
) -> list[dict]:
    """Return auditable deltas and flag only >3pp gains in both metrics."""
    baseline = _load_baseline(baseline_dir / "per_kpi_quality_gates.csv")
    comparisons = []
    for row in current:
        previous = baseline.get(row["kpi"])
        recall = row.get("recall")
        precision = row.get("precision")
        if previous is None or recall is None or precision is None:
            comparisons.append(
                {
                    "kpi": row["kpi"],
                    "baseline_recall": previous.get("recall") if previous else None,
                    "current_recall": recall,
                    "recall_delta": None,
                    "baseline_precision": previous.get("precision") if previous else None,
                    "current_precision": precision,
                    "precision_delta": None,
                    "major_improvement": False,
                }
            )
            continue
        recall_delta = float(recall) - previous["recall"]
        precision_delta = float(precision) - previous["precision"]
        comparisons.append(
            {
                "kpi": row["kpi"],
                "baseline_recall": previous["recall"],
                "current_recall": recall,
                "recall_delta": recall_delta,
                "baseline_precision": previous["precision"],
                "current_precision": precision,
                "precision_delta": precision_delta,
                "major_improvement": (
                    recall_delta > MAJOR_IMPROVEMENT_EXCLUSIVE
                    and precision_delta > MAJOR_IMPROVEMENT_EXCLUSIVE
                ),
            }
        )
    return comparisons
