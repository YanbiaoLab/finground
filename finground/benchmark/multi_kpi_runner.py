"""Run state-backed LEDGER Multi-KPI extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.agent import (
    MULTI_KPI_APP_NAME,
    MULTI_KPI_COMPACTION_EVENT_RETENTION,
    MULTI_KPI_COMPACTION_TOKEN_THRESHOLD,
    MULTI_KPI_FINAL_WARNING_CALL,
    MULTI_KPI_LLM_CALL_LIMIT,
    MULTI_KPI_PROGRESS_REMINDER_CALL,
    MULTI_KPI_PROMPT_VERSION,
    MULTI_KPI_SUBMISSION_DEADLINE,
    SETTINGS,
    create_multi_kpi_app,
)
from finground.benchmark.concurrency import map_concurrently
from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin
from finground.benchmark.llm_metrics import LlmCallCounterPlugin
from finground.benchmark.parquet import iter_multi_reports
from finground.documents import Report
from finground.tools import (
    MULTI_KPI_AUDIT_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
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
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
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


async def _run_report(
    report: Report,
    llm_counter: LlmCallCounterPlugin,
    execution_guard: MultiKpiExecutionGuardPlugin,
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
    runner = Runner(
        app=create_multi_kpi_app(plugins=[llm_counter, execution_guard]),
        session_service=session_service,
    )
    prompt = (
        f"Extract all supported LEDGER KPIs from report {report.report_id}. "
        "Inspect the report through the state-backed tools and submit the final result with "
        "submit_multi_kpi_extraction."
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    async for _event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=MULTI_KPI_LLM_CALL_LIMIT),
    ):
        pass

    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    extraction = session.state.get(MULTI_KPI_RESULT_STATE_KEY) if session is not None else None
    audit = session.state.get(MULTI_KPI_AUDIT_STATE_KEY) if session is not None else None
    if not isinstance(extraction, dict):
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


async def _process_report(report: Report) -> tuple[Report, dict]:
    started = time.monotonic()
    llm_counter = LlmCallCounterPlugin(
        max_calls=MULTI_KPI_LLM_CALL_LIMIT,
        force_tool_at_call=MULTI_KPI_SUBMISSION_DEADLINE,
        forced_tool_name="submit_multi_kpi_extraction",
    )
    execution_guard = MultiKpiExecutionGuardPlugin(max_calls=MULTI_KPI_LLM_CALL_LIMIT)
    try:
        extraction, audit = await _run_report(report, llm_counter, execution_guard)
        record = {
            "ticker": report.ticker,
            "year": report.year,
            "exchange": report.exchange,
            "report_name": report.report_id,
            "model": SETTINGS.model,
            "prompt_version": MULTI_KPI_PROMPT_VERSION,
            "status": "ok",
            "extraction": extraction,
            "audit": audit,
            "error": None,
            "llm_calls": llm_counter.count,
            "prevented_early_stops": execution_guard.prevented_early_stops,
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
            "llm_calls": llm_counter.count,
            "prevented_early_stops": execution_guard.prevented_early_stops,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
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

    started = time.monotonic()
    processed = 0
    ok = 0
    failed = 0
    total_llm_calls = 0
    total_prevented_early_stops = 0
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
        _process_report,
        limit=concurrency,
    ):
        processed += 1
        total_llm_calls += int(record["llm_calls"])
        total_prevented_early_stops += int(record["prevented_early_stops"])
        _write_record(raw_dir / f"{report.report_id}.json", record)
        if record["status"] == "ok":
            ok += 1
        else:
            failed += 1

    metadata = {
        "model": SETTINGS.model,
        "prompt_version": MULTI_KPI_PROMPT_VERSION,
        "input_format": "parquet",
        "reports_selected": len(selection_stats.report_ids),
        "reports_processed": processed,
        "reports_resumed": selection_stats.resumed,
        "report_ids": selection_stats.report_ids,
        "ok": ok,
        "failed": failed,
        "total_llm_calls": total_llm_calls,
        "total_prevented_early_stops": total_prevented_early_stops,
        "llm_call_limit": MULTI_KPI_LLM_CALL_LIMIT,
        "submission_deadline": MULTI_KPI_SUBMISSION_DEADLINE,
        "budget_reminder_calls": [
            MULTI_KPI_PROGRESS_REMINDER_CALL,
            MULTI_KPI_FINAL_WARNING_CALL,
        ],
        "concurrency": concurrency,
        "context_management": {
            "adk_context_filter": "recorded_multi_kpi",
            "compaction_token_threshold": MULTI_KPI_COMPACTION_TOKEN_THRESHOLD,
            "compaction_event_retention": MULTI_KPI_COMPACTION_EVENT_RETENTION,
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
