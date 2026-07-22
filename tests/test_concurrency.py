import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import finground.benchmark.multi_kpi_runner as multi_runner
import finground.benchmark.needle_runner as needle_runner
from finground.benchmark.concurrency import map_concurrently
from finground.documents import Report
from finground.models import NeedleAnswer

MMD_TEXT = """# Annual Report
<--- Page Split --->
# Consolidated Statements of Income
| Year | 2023 |
| Revenue | 123 |
"""


def test_bounded_map_never_exceeds_concurrency_limit() -> None:
    active = 0
    peak = 0

    async def worker(value: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    async def run() -> list[int]:
        return [result async for result in map_concurrently(range(8), worker, limit=3)]

    assert sorted(asyncio.run(run())) == list(range(8))
    assert peak == 3


def test_bounded_map_rejects_non_positive_limit() -> None:
    async def worker(value: int) -> int:
        return value

    async def run() -> None:
        async for _result in map_concurrently([1], worker, limit=0):
            pass

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(run())


def test_needle_runner_applies_concurrency_limit(tmp_path: Path, monkeypatch) -> None:
    parquet_path = tmp_path / "needle.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": [f"ACME_revenue_{year}" for year in range(2020, 2024)],
                "query_text": ["What was revenue?"] * 4,
                "ticker": ["ACME"] * 4,
                "exchange": ["NYSE"] * 4,
                "year": list(range(2020, 2024)),
                "mmd_text": [MMD_TEXT] * 4,
            }
        ),
        parquet_path,
    )
    active = 0
    peak = 0

    async def fake_run_query(_case, llm_counter) -> NeedleAnswer:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        llm_counter.count += 2
        return NeedleAnswer(found=False)

    monkeypatch.setattr(needle_runner, "_run_query", fake_run_query)
    output_dir = tmp_path / "needle-output"

    metadata = needle_runner.run_needle_sync(
        parquet_path=parquet_path,
        output_dir=output_dir,
        limit_queries=None,
        concurrency=2,
    )

    records = [
        json.loads(line) for line in (output_dir / "responses.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert peak == 2
    assert metadata["concurrency"] == 2
    assert metadata["total_llm_calls"] == 8
    assert {record["llm_calls"] for record in records} == {2}


def test_multi_runner_applies_concurrency_limit(tmp_path: Path, monkeypatch) -> None:
    parquet_path = tmp_path / "multi.parquet"
    pq.write_table(
        pa.table(
            {
                "ticker": [f"ACME{index}" for index in range(4)],
                "exchange": ["NYSE"] * 4,
                "year": [2023] * 4,
                "mmd_text": [MMD_TEXT] * 4,
            }
        ),
        parquet_path,
    )
    active = 0
    peak = 0

    async def fake_run_report(report, llm_counter, _budget_reminder) -> tuple[dict, dict]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        llm_counter.count += 3
        extraction = {
            "ticker": report.ticker,
            "reporting_currency": None,
            "units_note": None,
            "kpis": [],
        }
        return extraction, {**extraction, "notes": []}

    monkeypatch.setattr(multi_runner, "_run_report", fake_run_report)
    output_dir = tmp_path / "multi-output"

    metadata = multi_runner.run_multi_kpi_sync(
        parquet_path=parquet_path,
        output_dir=output_dir,
        limit_reports=None,
        reports_file=None,
        resume=False,
        concurrency=2,
    )

    assert len(list((output_dir / "raw").glob("*.json"))) == 4
    first_record = json.loads(next((output_dir / "raw").glob("*.json")).read_text())
    assert first_record["audit"]["notes"] == []
    assert peak == 2
    assert metadata["concurrency"] == 2
    assert metadata["total_llm_calls"] == 12
    assert metadata["llm_call_limit"] == 30
    assert metadata["prompt_version"] == "evidence-v1"
    assert metadata["submission_deadline"] == 25
    assert metadata["total_prevented_early_stops"] == 0
    assert metadata["budget_reminder_calls"] == [18, 24]
    assert len(metadata["report_ids"]) == 4
    assert metadata["context_management"] == {
        "adk_context_filter": "recorded_multi_kpi",
        "compaction_token_threshold": 18_000,
        "compaction_event_retention": 6,
    }

    resumed = multi_runner.run_multi_kpi_sync(
        parquet_path=parquet_path,
        output_dir=output_dir,
        limit_reports=2,
        reports_file=None,
        resume=True,
        concurrency=2,
    )
    assert resumed["reports_selected"] == 2
    assert resumed["reports_processed"] == 0
    assert resumed["reports_resumed"] == 2
    assert len(resumed["report_ids"]) == 2
    assert resumed["total_llm_calls"] == 0


def test_multi_runner_retains_llm_calls_when_report_fails(monkeypatch) -> None:
    async def fake_run_report(_report, llm_counter, _budget_reminder) -> dict:
        llm_counter.count += 4
        raise RuntimeError("model failed")

    monkeypatch.setattr(multi_runner, "_run_report", fake_run_report)
    report = Report("NYSE_ACME_2023", "NYSE", "ACME", 2023, MMD_TEXT)

    _report, record = asyncio.run(multi_runner._process_report(report))

    assert record["status"] == "failed"
    assert record["llm_calls"] == 4
