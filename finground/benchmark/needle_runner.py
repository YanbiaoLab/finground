"""Run LEDGER single-KPI needle extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from itertools import islice
from pathlib import Path

from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.agent import SETTINGS, create_needle_agent
from finground.benchmark.concurrency import map_concurrently
from finground.benchmark.llm_metrics import LlmCallCounterPlugin
from finground.benchmark.parquet import NeedleCase, iter_needle_cases
from finground.kpis import (
    KPI_DESCRIPTIONS,
    PER_SHARE_KPIS,
    SHARE_COUNT_KPIS,
    parse_query_id,
)
from finground.models import NeedleAnswer
from finground.tools import (
    NEEDLE_KPI_STATE_KEY,
    NEEDLE_RESULT_STATE_KEY,
    REPORT_STATE_KEY,
    build_report_state,
)

NEEDLE_LLM_CALL_LIMIT = 12


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _query_prompt(case: NeedleCase) -> str:
    ticker, kpi, year = parse_query_id(case.query_id)
    if kpi in PER_SHARE_KPIS:
        unit_hint = "per-share; never apply a statement-level scale"
    elif kpi in SHARE_COUNT_KPIS:
        unit_hint = "individual shares; apply the printed thousands/millions scale"
    else:
        unit_hint = "single units of reporting currency; apply the printed scale"
    return (
        f"Extract the requested KPI from report {case.report_id}.\n"
        f"Ticker: {ticker}\nFiscal year: {year}\nKPI: {kpi}\n"
        f"Canonical definition: {KPI_DESCRIPTIONS[kpi]}\n"
        f"Unit class: {unit_hint}\nQuestion: {case.query_text}\n"
        "Inspect the report through the state-backed tools and submit the final result with "
        "submit_needle_extraction."
    )


async def _run_query(case: NeedleCase, llm_counter: LlmCallCounterPlugin) -> NeedleAnswer:
    if case.report is None:
        raise ValueError("report unavailable")
    _ticker, kpi, _year = parse_query_id(case.query_id)
    session_service = InMemorySessionService()
    app_name = "finground_needle"
    user_id = "benchmark"
    digest = hashlib.sha256(case.query_id.encode()).hexdigest()[:16]
    session_id = f"needle-{digest}"
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={
            REPORT_STATE_KEY: build_report_state(case.report),
            NEEDLE_KPI_STATE_KEY: kpi,
        },
    )
    runner = Runner(
        app=App(
            name=app_name,
            root_agent=create_needle_agent(),
            plugins=[llm_counter],
        ),
        session_service=session_service,
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=_query_prompt(case))])
    async for _event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=NEEDLE_LLM_CALL_LIMIT),
    ):
        pass
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    result = session.state.get(NEEDLE_RESULT_STATE_KEY) if session is not None else None
    if not isinstance(result, dict):
        raise RuntimeError("agent stopped without a successful needle submission")
    return NeedleAnswer.model_validate(result)


async def _process_case(case: NeedleCase) -> dict:
    ticker, kpi, year = parse_query_id(case.query_id)
    started = time.monotonic()
    record = {
        "query_id": case.query_id,
        "ticker": ticker,
        "kpi": kpi,
        "year": year,
        "query_text": case.query_text,
        "report_name": case.report_id,
        "model": SETTINGS.model,
    }
    llm_counter = LlmCallCounterPlugin(max_calls=NEEDLE_LLM_CALL_LIMIT)
    if case.report is None:
        return record | {
            "status": "error",
            "error": "report unavailable",
            "llm_calls": 0,
            "latency_s": round(time.monotonic() - started, 3),
        }
    try:
        answer = await _run_query(case, llm_counter)
        return (
            record
            | answer.model_dump()
            | {
                "status": "ok",
                "llm_calls": llm_counter.count,
                "latency_s": round(time.monotonic() - started, 3),
            }
        )
    except Exception as error:  # noqa: BLE001 - retain one record per query
        return record | {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "llm_calls": llm_counter.count,
            "latency_s": round(time.monotonic() - started, 3),
        }


async def run_needle(
    *,
    parquet_path: Path,
    output_dir: Path,
    limit_queries: int | None,
    concurrency: int,
) -> dict:
    """Run bounded concurrent ADK extraction without reading KPI ground truth."""
    cases = iter_needle_cases(parquet_path)
    if limit_queries is not None:
        if limit_queries < 0:
            raise ValueError("limit_queries must be non-negative")
        cases = islice(cases, limit_queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    written = 0
    failures = 0
    total_llm_calls = 0
    started_run = time.monotonic()
    with responses_path.open("w", encoding="utf-8") as output:
        async for record in map_concurrently(
            cases,
            _process_case,
            limit=concurrency,
        ):
            output.write(json.dumps(record, default=str) + "\n")
            output.flush()
            written += 1
            total_llm_calls += int(record["llm_calls"])
            if record["status"] != "ok":
                failures += 1
    metadata = {
        "model": SETTINGS.model,
        "input_format": "parquet",
        "queries_written": written,
        "failures": failures,
        "total_llm_calls": total_llm_calls,
        "concurrency": concurrency,
        "elapsed_s": round(time.monotonic() - started_run, 3),
        "ground_truth_used_for_prediction": False,
    }
    _write_json(output_dir / "run_meta.json", metadata)
    return metadata


def run_needle_sync(
    *,
    parquet_path: Path,
    output_dir: Path,
    limit_queries: int | None,
    concurrency: int,
) -> dict:
    """Run the bounded Needle benchmark from synchronous callers."""
    return asyncio.run(
        run_needle(
            parquet_path=parquet_path,
            output_dir=output_dir,
            limit_queries=limit_queries,
            concurrency=concurrency,
        )
    )
