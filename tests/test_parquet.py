import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from finground.benchmark.kpi_samples import (
    select_grounded_report_ids,
    write_grounded_multi_parquet,
)
from finground.benchmark.multi_kpi_runner import run_multi_kpi_sync
from finground.benchmark.parquet import iter_multi_reports
from finground.kpis import KPI_ALIASES, KPI_KEYS
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


def test_grounded_kpi_selector_requires_label_and_value_in_same_evidence_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multi.parquet"
    _write_parquet(
        path,
        {
            "ticker": ["FALSE_MATCH", "TRUE_MATCH", "PARTIAL_NUMBER"],
            "exchange": ["NYSE", "NYSE", "NYSE"],
            "year": [2023, 2023, 2023],
            "mmd_text": [
                "| Short-term borrowings | 10 |\n| Derivative liabilities | 5.644 |",
                "<table><tr><td>Short-term borrowings</td><td>5.644</td></tr></table>",
                "| Short-term borrowings | 151 |",
            ],
            "short_term_borrowings": [5_644_000.0, 5_644_000.0, 51_000_000.0],
        },
    )

    assert select_grounded_report_ids(path, kpi="short_term_borrowings", limit=3) == [
        "NYSE_TRUE_MATCH_2023"
    ]


def test_grounded_multi_selector_writes_only_visible_long_answers(tmp_path: Path) -> None:
    source = tmp_path / "wide.parquet"
    output = tmp_path / "grounded-long.parquet"
    values = {kpi: float(1_000_000 + index * 1_000) for index, kpi in enumerate(KPI_KEYS)}
    evidence = "\n".join(
        f"| {KPI_ALIASES[kpi][0]} | {value / 1_000:,.0f} |" for kpi, value in values.items()
    )
    columns: dict[str, list[object]] = {
        "ticker": ["ACME"],
        "exchange": ["NYSE"],
        "company_name": ["ACME Corp"],
        "industry": ["Fixture"],
        "year": [2023],
        "mmd_text": [evidence],
    }
    columns.update({kpi: [value] for kpi, value in values.items()})
    _write_parquet(source, columns)

    result = write_grounded_multi_parquet(
        source,
        min_per_kpi=1,
        max_reports=1,
        max_per_ticker=1,
        output_file=output,
    )

    rows = pq.read_table(output).to_pylist()
    assert result["selected_report_count"] == 1
    assert result["grounded_answer_count"] == len(KPI_KEYS)
    assert set(result["per_kpi_counts"].values()) == {1}
    assert {row["kpi"] for row in rows} == set(KPI_KEYS)
    assert {row["ticker"] for row in rows} == {"ACME"}
    assert len({row["mmd_text"] for row in rows}) == 1
