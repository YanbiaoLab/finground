import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from finground.benchmark.multi_kpi_runner import run_multi_kpi_sync
from finground.benchmark.needle_runner import run_needle_sync
from finground.benchmark.parquet import iter_multi_reports, iter_needle_cases
from finground.tools import build_report_state

MMD_TEXT = """# Annual Report
<--- Page Split --->
# Consolidated Statements of Income
| Year | 2023 |
| Revenue | 123 |
"""


def _write_parquet(path: Path, values: dict[str, list[object]]) -> None:
    pq.write_table(pa.table(values), path)


def test_multi_parquet_reader_projects_out_kpi_answers(tmp_path: Path) -> None:
    parquet_path = tmp_path / "multi.parquet"
    _write_parquet(
        parquet_path,
        {
            "ticker": ["ACME"],
            "exchange": ["NYSE"],
            "year": [2023],
            "mmd_text": [MMD_TEXT],
            "revenue": [123_000_000.0],
        },
    )

    report = next(iter_multi_reports(parquet_path))

    assert report.report_id == "NYSE_ACME_2023"
    assert report.mmd_text == MMD_TEXT
    assert not hasattr(report, "revenue")
    state = build_report_state(report)
    assert state["pages"][1]["text"].startswith("# Consolidated Statements")


def test_needle_parquet_reader_does_not_require_answer_or_qrel_columns(tmp_path: Path) -> None:
    parquet_path = tmp_path / "needle.parquet"
    _write_parquet(
        parquet_path,
        {
            "query_id": ["ACME_revenue_2023"],
            "query_text": ["What was ACME revenue in 2023?"],
            "ticker": ["ACME"],
            "exchange": ["NYSE"],
            "year": [2023],
            "mmd_text": [MMD_TEXT],
        },
    )

    case = next(iter_needle_cases(parquet_path))

    assert case.query_id == "ACME_revenue_2023"
    assert case.report is not None
    assert case.report.report_id == "NYSE_ACME_2023"


def test_parquet_only_runners_accept_files_without_model_calls(tmp_path: Path) -> None:
    multi_path = tmp_path / "multi.parquet"
    needle_path = tmp_path / "needle.parquet"
    _write_parquet(
        multi_path,
        {
            "ticker": ["ACME"],
            "exchange": ["NYSE"],
            "year": [2023],
            "mmd_text": [MMD_TEXT],
        },
    )
    _write_parquet(
        needle_path,
        {
            "query_id": ["ACME_revenue_2023"],
            "query_text": ["What was ACME revenue in 2023?"],
            "ticker": ["ACME"],
            "exchange": ["NYSE"],
            "year": [2023],
            "mmd_text": [MMD_TEXT],
        },
    )

    multi_output = tmp_path / "multi-output"
    needle_output = tmp_path / "needle-output"
    multi_result = run_multi_kpi_sync(
        parquet_path=multi_path,
        output_dir=multi_output,
        limit_reports=0,
        reports_file=None,
        resume=False,
        concurrency=2,
    )
    needle_result = run_needle_sync(
        parquet_path=needle_path,
        output_dir=needle_output,
        limit_queries=0,
        concurrency=2,
    )

    assert multi_result["input_format"] == "parquet"
    assert multi_result["reports_processed"] == 0
    assert needle_result["input_format"] == "parquet"
    assert needle_result["queries_written"] == 0
    assert json.loads((multi_output / "run_meta.json").read_text())["input_format"] == "parquet"
    assert (needle_output / "responses.jsonl").read_text() == ""
