"""Score Needle predictions directly against KPI-QA Parquet ground truth."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from finground.benchmark.parquet import iter_parquet_rows
from finground.kpis import PER_SHARE_KPIS, SHARE_COUNT_KPIS

GROUND_TRUTH_COLUMNS = (
    "query_id",
    "ticker",
    "kpi",
    "year",
    "value",
    "source",
    "company_name",
    "exchange",
    "industry",
)
SCALE_FACTORS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


@dataclass(frozen=True, slots=True)
class NeedleGroundTruth:
    query_id: str
    ticker: str
    kpi: str
    year: int
    value: float
    source: str
    company_name: str
    exchange: str
    industry: str


def _unit_class(kpi: str) -> str:
    if kpi in PER_SHARE_KPIS:
        return "per_share"
    if kpi in SHARE_COUNT_KPIS:
        return "shares"
    return "monetary"


def _load_responses(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Needle responses not found: {path}")
    responses: dict[str, dict] = {}
    with path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
            query_id = record.get("query_id")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"missing query_id on {path}:{line_number}")
            responses[query_id] = record
    return responses


def _load_ground_truth(parquet_path: Path) -> dict[str, NeedleGroundTruth]:
    ground_truth: dict[str, NeedleGroundTruth] = {}
    for row in iter_parquet_rows(
        parquet_path,
        GROUND_TRUTH_COLUMNS,
        batch_size=4096,
    ):
        if row["value"] is None:
            continue
        query_id = str(row["query_id"])
        ground_truth[query_id] = NeedleGroundTruth(
            query_id=query_id,
            ticker=str(row["ticker"]),
            kpi=str(row["kpi"]),
            year=int(row["year"]),
            value=float(row["value"]),
            source=str(row["source"] or ""),
            company_name=str(row["company_name"] or ""),
            exchange=str(row["exchange"] or ""),
            industry=str(row["industry"] or ""),
        )
    return ground_truth


def _within(prediction: float, target: float, tolerance: float, zero_eps: float) -> bool:
    if abs(target) < zero_eps:
        return abs(prediction) < zero_eps
    return abs(prediction - target) / abs(target) <= tolerance


def _classify(
    record: dict,
    ground_truth: NeedleGroundTruth | None,
    *,
    tolerance: float,
    zero_eps: float,
) -> tuple[str, float | None, float | None]:
    status = record.get("status")
    if status == "skipped_too_long":
        return "skipped", None, None
    if status in {"failed", "error"}:
        return "no_response", None, None
    prediction = record.get("value")
    if not record.get("found") or prediction is None:
        return "not_found", None, None
    if ground_truth is None:
        return "wrong", None, None
    prediction = float(prediction)
    ratio = prediction / ground_truth.value if ground_truth.value != 0 else None
    relative_error = (
        (prediction - ground_truth.value) / abs(ground_truth.value)
        if ground_truth.value != 0
        else None
    )
    outcome = "matched" if _within(prediction, ground_truth.value, tolerance, zero_eps) else "wrong"
    return outcome, relative_error, ratio


def _wrong_bucket(
    record: dict,
    ground_truth: NeedleGroundTruth,
    ground_truth_index: dict[str, NeedleGroundTruth],
    ratio: float | None,
    *,
    tolerance: float,
    zero_eps: float,
) -> str:
    prediction = record.get("value")
    if prediction is None:
        return "other"
    prediction = float(prediction)
    for offset in (-1, 1, -2, 2):
        adjacent = ground_truth_index.get(
            f"{ground_truth.ticker}_{ground_truth.kpi}_{ground_truth.year + offset}"
        )
        if adjacent and _within(prediction, adjacent.value, tolerance, zero_eps):
            return f"year_shift({offset:+d})"
    if ground_truth.value != 0 and _within(
        prediction,
        -ground_truth.value,
        tolerance,
        zero_eps,
    ):
        return "sign_error"
    if ground_truth.value != 0:
        for factor in SCALE_FACTORS:
            if _within(
                prediction,
                ground_truth.value * factor,
                tolerance,
                zero_eps,
            ):
                exponent = round(math.log10(factor))
                return f"scale_error(x1e{exponent:+d})"
    if ratio is not None and 0.5 <= abs(ratio) <= 2:
        return "scope_factor"
    return "other"


def _metrics(items: list[dict]) -> dict:
    counts = Counter(item["outcome"] for item in items)
    matched = counts["matched"]
    wrong = counts["wrong"]
    not_found = counts["not_found"]
    evaluated = matched + wrong + not_found
    attempted = matched + wrong
    strict = sum(bool(item["matched_strict"]) for item in items)
    relative_errors = [
        abs(item["rel_error"])
        for item in items
        if item["rel_error"] is not None and item["outcome"] in {"matched", "wrong"}
    ]
    return {
        "n": len(items),
        "eval_n": evaluated,
        "matched": matched,
        "wrong": wrong,
        "not_found": not_found,
        "no_response": counts["no_response"],
        "skipped": counts["skipped"],
        "accuracy": matched / evaluated if evaluated else None,
        "accuracy_strict": strict / evaluated if evaluated else None,
        "attempt_rate": attempted / evaluated if evaluated else None,
        "precision_when_found": matched / attempted if attempted else None,
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
        "n",
        "matched",
        "wrong",
        "not_found",
        "accuracy",
        "accuracy_strict",
        "precision_when_found",
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
    wrong_buckets: Counter[str],
    output_dir: Path,
    parquet_path: Path,
    tolerance: float,
    strict_tolerance: float,
    aggregations: dict[str, list[dict]],
) -> str:
    lines = [
        "# Needle KPI benchmark",
        "",
        f"- Responses: `{output_dir / 'responses.jsonl'}`",
        f"- Ground truth: `{parquet_path}`",
        f"- Match tolerance: ±{tolerance:.2%} (strict: ±{strict_tolerance:.3%})",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {_format_metric(value)} |" for key, value in overall.items())
    lines.extend(["", "## Wrong-answer diagnostics", ""])
    if wrong_buckets:
        lines.extend(["| bucket | count |", "| --- | --- |"])
        lines.extend(f"| {key} | {value} |" for key, value in wrong_buckets.most_common())
    else:
        lines.append("_(no wrong answers)_")
    for title, key in (
        ("Per KPI", "kpi"),
        ("Per fiscal year", "year"),
        ("Per ground-truth source", "source"),
        ("Per unit class", "unit_class"),
    ):
        lines.extend(["", f"## {title}", "", _metrics_table(aggregations[key], key)])
    return "\n".join(lines) + "\n"


def score_needle(
    *,
    output_dir: Path,
    parquet_path: Path,
    tolerance: float = 0.01,
    strict_tolerance: float = 0.0005,
    zero_eps: float = 0.5,
) -> dict:
    """Score one Needle run using projected ground-truth columns from Parquet."""
    if min(tolerance, strict_tolerance, zero_eps) < 0:
        raise ValueError("score tolerances must be non-negative")
    responses = _load_responses(output_dir / "responses.jsonl")
    ground_truth = _load_ground_truth(parquet_path)

    rows: list[dict] = []
    for query_id, record in responses.items():
        truth = ground_truth.get(query_id)
        outcome, relative_error, ratio = _classify(
            record,
            truth,
            tolerance=tolerance,
            zero_eps=zero_eps,
        )
        prediction = record.get("value")
        matched_strict = bool(
            outcome in {"matched", "wrong"}
            and truth is not None
            and prediction is not None
            and _within(float(prediction), truth.value, strict_tolerance, zero_eps)
        )
        wrong_bucket = (
            _wrong_bucket(
                record,
                truth,
                ground_truth,
                ratio,
                tolerance=tolerance,
                zero_eps=zero_eps,
            )
            if outcome == "wrong" and truth is not None
            else ""
        )
        kpi = record.get("kpi") or (truth.kpi if truth else None)
        rows.append(
            {
                "query_id": query_id,
                "ticker": record.get("ticker") or (truth.ticker if truth else None),
                "kpi": kpi,
                "year": record.get("year") or (truth.year if truth else None),
                "unit_class": _unit_class(str(kpi)) if kpi else None,
                "outcome": outcome,
                "wrong_bucket": wrong_bucket,
                "matched_strict": matched_strict,
                "gt_value": truth.value if truth else None,
                "pred_value": prediction,
                "value_verbatim": record.get("value_verbatim"),
                "unit_scale": record.get("unit_scale"),
                "page": record.get("page"),
                "rel_error": relative_error,
                "ratio": ratio,
                "source": truth.source if truth else None,
                "industry": truth.industry if truth else None,
                "exchange": truth.exchange if truth else None,
                "company_name": truth.company_name if truth else None,
                "model": record.get("model"),
                "report_name": record.get("report_name"),
                "latency_s": record.get("latency_s"),
                "run_status": record.get("status"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_fields = [
        "query_id",
        "ticker",
        "kpi",
        "year",
        "unit_class",
        "outcome",
        "wrong_bucket",
        "matched_strict",
        "gt_value",
        "pred_value",
        "value_verbatim",
        "unit_scale",
        "page",
        "rel_error",
        "ratio",
        "source",
        "industry",
        "exchange",
        "company_name",
        "model",
        "report_name",
        "latency_s",
        "run_status",
    ]
    _write_csv(output_dir / "scored.csv", rows, scored_fields)

    metric_fields = [
        "n",
        "eval_n",
        "matched",
        "wrong",
        "not_found",
        "no_response",
        "skipped",
        "accuracy",
        "accuracy_strict",
        "attempt_rate",
        "precision_when_found",
        "median_abs_rel_error",
    ]
    aggregations: dict[str, list[dict]] = {}
    for key, filename in (
        ("kpi", "per_kpi.csv"),
        ("year", "per_year.csv"),
        ("source", "per_source.csv"),
        ("unit_class", "per_unit_class.csv"),
    ):
        aggregated = sorted(_aggregate(rows, key), key=lambda row: row[key])
        aggregations[key] = aggregated
        _write_csv(output_dir / filename, aggregated, [key, *metric_fields])

    overall = _metrics(rows)
    wrong_buckets = Counter(row["wrong_bucket"] for row in rows if row["wrong_bucket"])
    (output_dir / "summary.md").write_text(
        _render_summary(
            overall=overall,
            wrong_buckets=wrong_buckets,
            output_dir=output_dir,
            parquet_path=parquet_path,
            tolerance=tolerance,
            strict_tolerance=strict_tolerance,
            aggregations=aggregations,
        ),
        encoding="utf-8",
    )
    return overall | {"output_dir": str(output_dir)}
