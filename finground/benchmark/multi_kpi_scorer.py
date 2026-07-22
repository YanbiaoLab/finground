"""Score Multi-KPI predictions directly against wide Parquet ground truth."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from finground.benchmark.parquet import iter_parquet_rows
from finground.kpis import KPI_KEYS
from finground.models import ReportExtraction

GROUND_TRUTH_COLUMNS = (
    "ticker",
    "exchange",
    "company_name",
    "industry",
    "year",
    *KPI_KEYS,
)


def _load_run_scope(output_dir: Path) -> set[str] | None:
    metadata_path = output_dir / "run_meta.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run metadata in {metadata_path}: {error}") from error
    report_ids = metadata.get("report_ids")
    if not isinstance(report_ids, list) or not all(
        isinstance(report_id, str) for report_id in report_ids
    ):
        raise ValueError(f"run metadata does not contain a valid report_ids list: {metadata_path}")
    return set(report_ids)


def _load_predictions(
    raw_dir: Path,
    report_scope: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Multi-KPI raw output directory not found: {raw_dir}")
    predictions: list[dict] = []
    runs: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid prediction JSON in {path}: {error}") from error
        report_name = record.get("report_name")
        if report_scope is not None and report_name not in report_scope:
            continue
        run = {
            "ticker": record.get("ticker"),
            "year": record.get("year"),
            "exchange": record.get("exchange"),
            "status": record.get("status"),
            "model": record.get("model"),
            "report_name": report_name,
        }
        runs.append(run)
        if record.get("status") != "ok":
            continue
        try:
            extraction = ReportExtraction.model_validate(record.get("extraction"))
        except ValidationError:
            run["status"] = "invalid"
            continue
        predictions.extend(
            {
                "ticker": str(record.get("ticker")),
                "year": item.fiscal_year,
                "kpi": item.kpi,
                "pred_value": item.value,
                "reporting_currency": extraction.reporting_currency,
                "model": record.get("model"),
                "report_name": record.get("report_name"),
            }
            for item in extraction.kpis
        )
    if not runs:
        raise ValueError(f"no prediction JSON files found in {raw_dir}")
    return predictions, runs


def _load_ground_truth(
    parquet_path: Path,
    allowed_pairs: set[tuple[str, int]],
) -> dict[tuple[str, int, str], dict]:
    ground_truth: dict[tuple[str, int, str], dict] = {}
    for row in iter_parquet_rows(
        parquet_path,
        GROUND_TRUTH_COLUMNS,
        batch_size=4096,
    ):
        pair = (str(row["ticker"]), int(row["year"]))
        if pair not in allowed_pairs:
            continue
        metadata = {
            "exchange": str(row["exchange"] or ""),
            "company_name": str(row["company_name"] or ""),
            "industry": str(row["industry"] or ""),
        }
        for kpi in KPI_KEYS:
            value = row[kpi]
            if value is not None:
                ground_truth[(pair[0], pair[1], kpi)] = metadata | {"value": float(value)}
    return ground_truth


def _classify(
    ground_truth: float | None,
    prediction: float | None,
    *,
    tolerance: float,
    zero_eps: float,
) -> tuple[str, float | None]:
    if ground_truth is None:
        return "extra", None
    if prediction is None:
        return "missing", None
    if abs(ground_truth) < zero_eps:
        return ("matched", 0.0) if abs(prediction) < zero_eps else ("wrong", None)
    relative_error = (prediction - ground_truth) / abs(ground_truth)
    return (
        ("matched", relative_error)
        if abs(relative_error) <= tolerance
        else ("wrong", relative_error)
    )


def _metrics(rows: list[dict]) -> dict:
    counts = Counter(row["status"] for row in rows)
    matched = counts["matched"]
    wrong = counts["wrong"]
    missing = counts["missing"]
    extra = counts["extra"]
    ground_truth_total = matched + wrong + missing
    prediction_total = matched + wrong + extra
    relative_errors = [
        abs(row["rel_error"])
        for row in rows
        if row["rel_error"] is not None and row["status"] in {"matched", "wrong"}
    ]
    return {
        "n_gt": ground_truth_total,
        "n_pred": prediction_total,
        "matched": matched,
        "wrong": wrong,
        "missing": missing,
        "extra": extra,
        "recall": matched / ground_truth_total if ground_truth_total else None,
        "precision": matched / (matched + wrong) if matched + wrong else None,
        "median_abs_rel_error": (statistics.median(relative_errors) if relative_errors else None),
    }


def _aggregate(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            grouped[str(value)].append(row)
    return [{key: value} | _metrics(items) for value, items in grouped.items()]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_table(rows: list[dict], key: str) -> str:
    if not rows:
        return "_(no data)_"
    columns = [
        key,
        "n_gt",
        "n_pred",
        "matched",
        "wrong",
        "missing",
        "extra",
        "recall",
        "precision",
        "median_abs_rel_error",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    lines.extend(
        ("| " + " | ".join(_format_metric(row.get(column)) for column in columns) + " |")
        for row in rows
    )
    return "\n".join(lines)


def _render_summary(
    *,
    overall: dict,
    runs: list[dict],
    output_dir: Path,
    parquet_path: Path,
    tolerance: float,
    aggregations: dict[str, list[dict]],
) -> str:
    run_counts = Counter(str(run["status"]) for run in runs)
    lines = [
        "# Multi-KPI extraction benchmark",
        "",
        f"- Predictions: `{output_dir / 'raw'}`",
        f"- Ground truth: `{parquet_path}`",
        f"- Match tolerance: ±{tolerance:.2%}",
        f"- Reports loaded: {len(runs)}",
        "- Report statuses: "
        + ", ".join(f"{key}={value}" for key, value in sorted(run_counts.items())),
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {_format_metric(value)} |" for key, value in overall.items())
    for title, key in (
        ("Per KPI", "kpi"),
        ("Per fiscal year", "year"),
        ("Per exchange", "exchange"),
        ("Per industry", "industry"),
    ):
        lines.extend(["", f"## {title}", "", _metrics_table(aggregations[key], key)])
    return "\n".join(lines) + "\n"


def score_multi_kpi(
    *,
    output_dir: Path,
    parquet_path: Path,
    tolerance: float = 0.01,
    zero_eps: float = 0.5,
) -> dict:
    """Score one Multi-KPI run using projected ground-truth columns from Parquet."""
    if min(tolerance, zero_eps) < 0:
        raise ValueError("score tolerances must be non-negative")
    report_scope = _load_run_scope(output_dir)
    predictions, runs = _load_predictions(output_dir / "raw", report_scope)
    allowed_pairs = {
        (str(run["ticker"]), int(run["year"]))
        for run in runs
        if run["ticker"] is not None and run["year"] is not None
    }
    ground_truth = _load_ground_truth(parquet_path, allowed_pairs)

    prediction_index: dict[tuple[str, int, str], dict] = {}
    for prediction in predictions:
        pair = (prediction["ticker"], prediction["year"])
        if pair in allowed_pairs:
            key = (pair[0], pair[1], prediction["kpi"])
            prediction_index.setdefault(key, prediction)

    rows: list[dict] = []
    for ticker, year, kpi in sorted(set(ground_truth) | set(prediction_index)):
        truth = ground_truth.get((ticker, year, kpi))
        prediction = prediction_index.get((ticker, year, kpi))
        truth_value = truth["value"] if truth else None
        prediction_value = prediction["pred_value"] if prediction else None
        status, relative_error = _classify(
            truth_value,
            prediction_value,
            tolerance=tolerance,
            zero_eps=zero_eps,
        )
        rows.append(
            {
                "ticker": ticker,
                "year": year,
                "kpi": kpi,
                "status": status,
                "gt_value": truth_value,
                "pred_value": prediction_value,
                "rel_error": relative_error,
                "exchange": truth.get("exchange") if truth else None,
                "industry": truth.get("industry") if truth else None,
                "company_name": truth.get("company_name") if truth else None,
                "reporting_currency": (
                    prediction.get("reporting_currency") if prediction else None
                ),
                "model": prediction.get("model") if prediction else None,
                "report_name": prediction.get("report_name") if prediction else None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = [
        "ticker",
        "year",
        "kpi",
        "status",
        "gt_value",
        "pred_value",
        "rel_error",
        "exchange",
        "industry",
        "company_name",
        "reporting_currency",
        "model",
        "report_name",
    ]
    _write_csv(output_dir / "predictions_long.csv", rows, prediction_fields)

    metric_fields = [
        "n_gt",
        "n_pred",
        "matched",
        "wrong",
        "missing",
        "extra",
        "recall",
        "precision",
        "median_abs_rel_error",
    ]
    aggregations: dict[str, list[dict]] = {}
    for key, filename in (
        ("kpi", "per_kpi_metrics.csv"),
        ("year", "per_year_metrics.csv"),
        ("exchange", "per_exchange_metrics.csv"),
        ("industry", "per_industry_metrics.csv"),
    ):
        aggregated = sorted(_aggregate(rows, key), key=lambda row: row[key])
        aggregations[key] = aggregated
        _write_csv(output_dir / filename, aggregated, [key, *metric_fields])

    overall = _metrics(rows)
    (output_dir / "summary.md").write_text(
        _render_summary(
            overall=overall,
            runs=runs,
            output_dir=output_dir,
            parquet_path=parquet_path,
            tolerance=tolerance,
            aggregations=aggregations,
        ),
        encoding="utf-8",
    )
    return overall | {
        "reports_scored": len(runs),
        "output_dir": str(output_dir),
    }
