"""Run state-backed LEDGER Multi-KPI extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.agents.common import SETTINGS
from finground.agents.multi_kpi import (
    MULTI_KPI_APP_NAME,
    MULTI_KPI_CONTEXT_WINDOW_TOKENS,
    MULTI_KPI_FINAL_WARNING_CALL,
    MULTI_KPI_LLM_CALL_LIMIT,
    MULTI_KPI_MAX_OUTPUT_TOKENS,
    MULTI_KPI_PROGRESS_REMINDER_CALL,
    MULTI_KPI_PROMPT_VERSION,
    MULTI_KPI_SEARCH_LIMIT,
    MULTI_KPI_SUBMISSION_DEADLINE,
    create_multi_kpi_app,
)
from finground.benchmark.adk_trajectory import AdkTrajectoryPlugin
from finground.benchmark.concurrency import map_concurrently
from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin
from finground.benchmark.llm_metrics import LlmCallCounterPlugin, MultiKpiRunMetricsPlugin
from finground.benchmark.parquet import iter_multi_reports
from finground.documents import Report
from finground.kpis import KPI_KEYS
from finground.tools import (
    MULTI_KPI_AUDIT_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    REPORT_STATE_KEY,
    build_report_state,
)


@dataclass(slots=True)
class _SelectionStats:
    report_ids: list[str] = field(default_factory=list)
    resumed: int = 0


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def _existing_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return (
            record.get("status") == "ok"
            and record.get("model") == SETTINGS.model
            and record.get("prompt_version") == MULTI_KPI_PROMPT_VERSION
        )
    except (OSError, json.JSONDecodeError):
        return False


def _load_report_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _build_partial_extraction(audit: dict) -> dict:
    return {
        "ticker": audit.get("ticker"),
        "reporting_currency": audit.get("reporting_currency"),
        "units_note": audit.get("units_note"),
        "kpis": [
            {
                "kpi": item.get("kpi"),
                "fiscal_year": item.get("fiscal_year"),
                "value": item.get("value"),
            }
            for item in audit.get("kpis", [])
            if item.get("status") in {"found", "explicit_zero"}
        ],
    }


async def _run_report(
    report: Report,
    llm_counter: LlmCallCounterPlugin,
    execution_guard: MultiKpiExecutionGuardPlugin,
    run_metrics: MultiKpiRunMetricsPlugin,
    trajectory: AdkTrajectoryPlugin | None,
) -> tuple[dict, dict]:
    app_name = MULTI_KPI_APP_NAME
    user_id = "benchmark"
    digest = hashlib.sha256(report.report_id.encode()).hexdigest()[:16]
    session_id = f"multi-{digest}"
    state = {REPORT_STATE_KEY: build_report_state(report)}
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=state,
    )
    plugins = [llm_counter, execution_guard, run_metrics]
    if trajectory is not None:
        plugins.append(trajectory)
    runner = Runner(
        app=create_multi_kpi_app(plugins=plugins),
        session_service=session_service,
    )
    prompt = (
        f"Extract all supported LEDGER KPIs from report {report.report_id}. "
        "Inspect the report through the state-backed tools and submit the final result with "
        "submit_multi_kpi_extraction."
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    limit_error: LlmCallsLimitExceededError | None = None
    try:
        async for _event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
            run_config=RunConfig(max_llm_calls=MULTI_KPI_SUBMISSION_DEADLINE),
        ):
            pass
    except LlmCallsLimitExceededError as error:
        limit_error = error

    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    extraction = session.state.get(MULTI_KPI_RESULT_STATE_KEY) if session is not None else None
    audit = (
        session.state.get(MULTI_KPI_AUDIT_STATE_KEY)
        or session.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY)
        if session is not None
        else None
    )
    if not isinstance(extraction, dict) and isinstance(audit, dict):
        extraction = _build_partial_extraction(audit)
    if not isinstance(extraction, dict):
        if limit_error is not None:
            raise limit_error
        raise RuntimeError("agent stopped without a successful multi-KPI submission")
    if not isinstance(audit, dict):
        raise RuntimeError("agent stopped without a validated multi-KPI evidence record")
    return extraction, audit


def _select_reports(
    reports: Iterable[Report],
    *,
    wanted: set[str] | None,
    limit_reports: int | None,
    resume: bool,
    raw_dir: Path,
    stats: _SelectionStats,
) -> Iterator[Report]:
    if limit_reports is not None and limit_reports < 0:
        raise ValueError("limit_reports must be non-negative")
    selected = 0
    for report in reports:
        if wanted is not None and report.report_id not in wanted:
            continue
        if limit_reports is not None and selected >= limit_reports:
            return
        selected += 1
        stats.report_ids.append(report.report_id)
        if resume and _existing_ok(raw_dir / f"{report.report_id}.json"):
            stats.resumed += 1
            continue
        yield report


async def _process_report(
    report: Report,
    *,
    trajectory_dir: Path | None = None,
) -> tuple[Report, dict]:
    started = time.monotonic()
    llm_counter = LlmCallCounterPlugin(
        max_calls=MULTI_KPI_LLM_CALL_LIMIT,
        force_tool_at_call=MULTI_KPI_SUBMISSION_DEADLINE,
        forced_tool_name="submit_multi_kpi_extraction",
    )
    execution_guard = MultiKpiExecutionGuardPlugin(
        max_calls=MULTI_KPI_LLM_CALL_LIMIT,
        max_searches=MULTI_KPI_SEARCH_LIMIT,
    )
    run_metrics = MultiKpiRunMetricsPlugin()
    trajectory = (
        AdkTrajectoryPlugin(trajectory_dir / f"{report.report_id}.jsonl")
        if trajectory_dir is not None
        else None
    )
    try:
        extraction, audit = await _run_report(
            report,
            llm_counter,
            execution_guard,
            run_metrics,
            trajectory,
        )
        covered = {
            item.get("kpi")
            for item in audit.get("kpis", [])
            if item.get("fiscal_year") == report.year
        }
        pending_kpis = [kpi for kpi in KPI_KEYS if kpi not in covered]
        complete = not pending_kpis
        record = {
            "ticker": report.ticker,
            "year": report.year,
            "exchange": report.exchange,
            "report_name": report.report_id,
            "model": SETTINGS.model,
            "prompt_version": MULTI_KPI_PROMPT_VERSION,
            "status": "ok" if complete else "incomplete",
            "extraction": extraction,
            "audit": audit,
            "error": (
                None if complete else f"incomplete KPI coverage: {len(pending_kpis)} pending"
            ),
            "coverage_count": len(KPI_KEYS) - len(pending_kpis),
            "pending_kpis": pending_kpis,
            "llm_calls": llm_counter.count,
            "prevented_early_stops": execution_guard.prevented_early_stops,
            "execution_metrics": run_metrics.snapshot(),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except Exception as error:  # noqa: BLE001 - retain one record per report
        record = {
            "ticker": report.ticker,
            "year": report.year,
            "exchange": report.exchange,
            "report_name": report.report_id,
            "model": SETTINGS.model,
            "prompt_version": MULTI_KPI_PROMPT_VERSION,
            "status": "failed",
            "extraction": None,
            "audit": None,
            "error": f"{type(error).__name__}: {error}",
            "coverage_count": 0,
            "pending_kpis": list(KPI_KEYS),
            "llm_calls": llm_counter.count,
            "prevented_early_stops": execution_guard.prevented_early_stops,
            "execution_metrics": run_metrics.snapshot(),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    if trajectory is not None:
        trajectory.finish(
            outcome=str(record["status"]),
            summary={
                "status": record["status"],
                "coverage_count": record["coverage_count"],
                "pending_kpis": record["pending_kpis"],
                "llm_calls": record["llm_calls"],
                "prevented_early_stops": record["prevented_early_stops"],
                "elapsed_s": record["elapsed_s"],
                "error": record["error"],
            },
        )
        record["trajectory"] = trajectory.snapshot()
    else:
        record["trajectory"] = None
    return report, record


async def run_multi_kpi(
    *,
    parquet_path: Path,
    output_dir: Path,
    limit_reports: int | None,
    reports_file: Path | None,
    resume: bool,
    concurrency: int,
) -> dict:
    """Run bounded concurrent state-backed ADK extraction per report."""
    reports = iter_multi_reports(parquet_path)
    wanted = _load_report_filter(reports_file)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"

    started = time.monotonic()
    processed = 0
    ok = 0
    incomplete = 0
    failed = 0
    total_llm_calls = 0
    total_prevented_early_stops = 0
    total_validation_errors = 0
    total_retryable_error_calls = 0
    total_partial_success_calls = 0
    total_repeated_validation_error_calls = 0
    aggregate_tool_calls: Counter[str] = Counter()
    aggregate_tool_statuses: Counter[str] = Counter()
    max_prompt_tokens = 0
    trajectories_complete = 0
    trajectory_write_errors = 0
    selection_stats = _SelectionStats()
    selected_reports = _select_reports(
        reports,
        wanted=wanted,
        limit_reports=limit_reports,
        resume=resume,
        raw_dir=raw_dir,
        stats=selection_stats,
    )
    async for report, record in map_concurrently(
        selected_reports,
        partial(_process_report, trajectory_dir=trajectory_dir),
        limit=concurrency,
    ):
        processed += 1
        total_llm_calls += int(record["llm_calls"])
        total_prevented_early_stops += int(record["prevented_early_stops"])
        execution_metrics = record["execution_metrics"]
        total_validation_errors += int(execution_metrics["validation_error_count"])
        total_retryable_error_calls += int(execution_metrics["retryable_error_calls"])
        total_partial_success_calls += int(execution_metrics["partial_success_calls"])
        total_repeated_validation_error_calls += int(
            execution_metrics["repeated_validation_error_calls"]
        )
        aggregate_tool_calls.update(execution_metrics["tool_calls"])
        aggregate_tool_statuses.update(execution_metrics["tool_statuses"])
        max_prompt_tokens = max(
            max_prompt_tokens,
            int(execution_metrics["model_tokens"]["prompt_max"]),
        )
        trajectory = record["trajectory"]
        trajectories_complete += int(bool(trajectory and trajectory["complete"]))
        trajectory_write_errors += int(bool(trajectory and trajectory["write_error"]))
        _write_record(raw_dir / f"{report.report_id}.json", record)
        if record["status"] == "ok":
            ok += 1
        elif record["status"] == "incomplete":
            incomplete += 1
        else:
            failed += 1

    metadata = {
        "model": SETTINGS.model,
        "prompt_version": MULTI_KPI_PROMPT_VERSION,
        "input_format": "parquet",
        "reports_file": str(reports_file) if reports_file is not None else None,
        "reports_selected": len(selection_stats.report_ids),
        "reports_processed": processed,
        "reports_resumed": selection_stats.resumed,
        "report_ids": selection_stats.report_ids,
        "ok": ok,
        "incomplete": incomplete,
        "failed": failed,
        "total_llm_calls": total_llm_calls,
        "total_prevented_early_stops": total_prevented_early_stops,
        "execution_metrics": {
            "tool_calls": dict(aggregate_tool_calls),
            "tool_statuses": dict(aggregate_tool_statuses),
            "validation_error_count": total_validation_errors,
            "retryable_error_calls": total_retryable_error_calls,
            "partial_success_calls": total_partial_success_calls,
            "repeated_validation_error_calls": total_repeated_validation_error_calls,
            "max_prompt_tokens": max_prompt_tokens,
        },
        "llm_call_limit": MULTI_KPI_LLM_CALL_LIMIT,
        "search_call_limit": MULTI_KPI_SEARCH_LIMIT,
        "adk_run_call_limit": MULTI_KPI_SUBMISSION_DEADLINE,
        "max_output_tokens": MULTI_KPI_MAX_OUTPUT_TOKENS,
        "submission_deadline": MULTI_KPI_SUBMISSION_DEADLINE,
        "budget_reminder_calls": [
            MULTI_KPI_PROGRESS_REMINDER_CALL,
            MULTI_KPI_FINAL_WARNING_CALL,
        ],
        "concurrency": concurrency,
        "context_management": {
            "adk_context_filter": "recorded_multi_kpi",
            "checkpoint_state": "multi_kpi_work_record",
            "older_tool_payloads": "single_active_retrieval_batch",
            "model_context_window_tokens": MULTI_KPI_CONTEXT_WINDOW_TOKENS,
            "llm_event_summarization": False,
        },
        "trajectory_logging": {
            "enabled": True,
            "format": "adk-lifecycle-events-jsonl",
            "directory": str(trajectory_dir),
            "content": "full_model_requests_responses_tool_calls_results_and_events",
            "partial_suffix": ".partial",
            "trajectories_complete": trajectories_complete,
            "write_errors": trajectory_write_errors,
        },
        "elapsed_s": round(time.monotonic() - started, 3),
        "ground_truth_used_for_prediction": False,
    }
    _write_record(output_dir / "run_meta.json", metadata)
    return metadata


def run_multi_kpi_sync(
    *,
    parquet_path: Path,
    output_dir: Path,
    limit_reports: int | None,
    reports_file: Path | None,
    resume: bool,
    concurrency: int,
) -> dict:
    """Run the bounded Multi-KPI benchmark from synchronous callers."""
    return asyncio.run(
        run_multi_kpi(
            parquet_path=parquet_path,
            output_dir=output_dir,
            limit_reports=limit_reports,
            reports_file=reports_file,
            resume=resume,
            concurrency=concurrency,
        )
    )
