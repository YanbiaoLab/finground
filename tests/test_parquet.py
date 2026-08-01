import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from finground.benchmark.kpi_samples import select_grounded_report_ids
from finground.benchmark.multi_kpi_runner import run_multi_kpi_sync
from finground.benchmark.parquet import iter_multi_reports
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


def test_parquet_runner_accepts_files_without_model_calls(tmp_path: Path) -> None:
    multi_path = tmp_path / "multi.parquet"
    _write_parquet(
        multi_path,
        {
            "ticker": ["ACME"],
            "exchange": ["NYSE"],
            "year": [2023],
            "mmd_text": [MMD_TEXT],
        },
    )

    multi_output = tmp_path / "multi-output"
    multi_result = run_multi_kpi_sync(
        parquet_path=multi_path,
        output_dir=multi_output,
        limit_reports=0,
        reports_file=None,
        resume=False,
        concurrency=2,
    )

    assert multi_result["input_format"] == "parquet"
    assert multi_result["reports_processed"] == 0
    assert json.loads((multi_output / "run_meta.json").read_text())["input_format"] == "parquet"


def test_grounded_kpi_selector_requires_visible_label_and_value(tmp_path: Path) -> None:
    path = tmp_path / "multi.parquet"
    _write_parquet(
        path,
        {
            "ticker": ["MATCH", "NO_VALUE", "NO_LABEL"],
            "exchange": ["NYSE", "NYSE", "NYSE"],
            "year": [2023, 2023, 2023],
            "mmd_text": [
                "Revenue | 1,234 | 2023",
                "Revenue | 999 | 2023",
                "Metric | 1,234 | 2023",
            ],
            "revenue": [1_234_000.0, 1_234_000.0, 1_234_000.0],
        },
    )

    assert select_grounded_report_ids(path, kpi="revenue", limit=2) == ["NYSE_MATCH_2023"]
