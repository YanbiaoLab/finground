import csv
from pathlib import Path

from finground.benchmark.kpi_improvement import compare_kpi_quality
from finground.benchmark.multi_kpi_scorer import (
    MIN_PRECISION_EXCLUSIVE,
    MIN_RECALL_EXCLUSIVE,
    _metrics,
)


def test_precision_counts_extra_predictions() -> None:
    metrics = _metrics(
        [
            {"status": "matched", "rel_error": 0.0},
            {"status": "extra", "rel_error": None},
        ]
    )

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5


def test_quality_thresholds_are_strict_user_requirements() -> None:
    assert MIN_RECALL_EXCLUSIVE == 0.95
    assert MIN_PRECISION_EXCLUSIVE == 0.98


def test_major_improvement_requires_both_metrics_above_three_points(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with (baseline / "per_kpi_quality_gates.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["kpi", "recall", "precision"])
        writer.writeheader()
        writer.writerows(
            [
                {"kpi": "revenue", "recall": 0.80, "precision": 0.90},
                {"kpi": "inventory", "recall": 0.80, "precision": 0.90},
            ]
        )

    comparisons = compare_kpi_quality(
        [
            {"kpi": "revenue", "recall": 0.84, "precision": 0.94},
            {"kpi": "inventory", "recall": 0.84, "precision": 0.92},
        ],
        baseline_dir=baseline,
    )

    assert comparisons[0]["major_improvement"] is True
    assert comparisons[1]["major_improvement"] is False
