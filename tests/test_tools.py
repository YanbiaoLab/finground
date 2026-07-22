from pathlib import Path
from types import SimpleNamespace

import pytest

from finground.documents import Report
from finground.kpis import KPI_KEYS
from finground.tools import (
    MULTI_KPI_AUDIT_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    NEEDLE_KPI_STATE_KEY,
    NEEDLE_RESULT_STATE_KEY,
    build_report_state,
    get_report_info,
    query_multi_kpi_progress,
    read_report_pages,
    record_multi_kpi_progress,
    search_report,
    submit_multi_kpi_extraction,
    submit_needle_extraction,
)

REPORT = Path(__file__).parent / "fixtures" / "ledger" / "report.mmd"


def _context() -> SimpleNamespace:
    report = Report("NYSE_ACME_2023", "NYSE", "ACME", 2023, REPORT.read_text())
    state = {
        "report": build_report_state(report),
        NEEDLE_KPI_STATE_KEY: "revenue",
    }
    return SimpleNamespace(
        state=state,
        actions=SimpleNamespace(skip_summarization=None),
    )


def _evidence(
    *,
    kpi: str = "revenue",
    fiscal_year: int = 2023,
    value_verbatim: str = "1,234",
    line_label: str = "Revenue",
    status: str = "found",
    unit_scale: str = "millions",
    unit_text: str | None = "(in millions, except per-share amounts)",
) -> dict:
    return {
        "kpi": kpi,
        "fiscal_year": fiscal_year,
        "status": status,
        "value_verbatim": value_verbatim,
        "unit_scale": unit_scale,
        "unit_text": unit_text,
        "unit_page": 3 if unit_text is not None else None,
        "page": 3,
        "statement": "Consolidated Statements of Operations",
        "line_label": line_label,
        "year_label": str(fiscal_year),
        "scope": "consolidated total company",
    }


def test_report_state_contains_document_pages() -> None:
    state = _context().state["report"]

    assert state["report_id"] == "NYSE_ACME_2023"
    assert len(state["pages"]) == 3
    assert "| Revenue | 1,234 | 1,100 |" in state["pages"][2]["text"]


def test_get_report_info_returns_page_count_and_bounded_outline() -> None:
    result = get_report_info(_context())

    assert result["status"] == "success"
    assert result["page_count"] == 3
    assert result["page_range"] == {"first": 1, "last": 3}
    assert result["outline"][-1] == {
        "page": 3,
        "heading": "Consolidated Statements of Operations",
    }


def test_search_report_combines_ranked_and_exact_phrase_search() -> None:
    result = search_report(
        "",
        ["Consolidated Statements of Operations"],
        None,
        5,
        _context(),
    )

    assert result["status"] == "success"
    assert result["results"][0]["page"] == 3
    assert result["results"][0]["matched_phrases"] == ["Consolidated Statements of Operations"]
    assert "Consolidated Statements of Operations" in result["results"][0]["snippet"]


def test_search_report_reads_only_state_backed_report() -> None:
    result = search_report("What was revenue?", ["revenue"], 2023, 5, _context())

    assert result["status"] == "success"
    assert result["report_id"] == "NYSE_ACME_2023"
    assert result["results"][0]["page"] == 3
    assert result["results"][0]["matched_phrases"] == ["revenue"]
    assert "(in millions, except per-share amounts)" in result["results"][0]["snippet"]
    assert "score" not in result["results"][0]


def test_search_report_normalizes_string_numbers_from_model_tool_calls() -> None:
    result = search_report("What was revenue?", ["revenue"], "2023", "5", _context())

    assert result["status"] == "success"
    assert result["results"][0]["page"] == 3


def test_search_report_returns_retryable_feedback_for_invalid_numeric_arguments() -> None:
    result = search_report("revenue", ["revenue"], "not-a-year", "5", _context())

    assert result == {
        "status": "error",
        "retryable": True,
        "error": "report tool arguments are invalid",
        "validation_errors": [{"field": "year", "message": "value must be an integer"}],
    }


def test_read_report_pages_deduplicates_and_limits_requested_pages() -> None:
    result = read_report_pages([3, 3, 2, 999], [], _context())

    assert result["status"] == "success"
    assert [page["page"] for page in result["pages"]] == [3, 2]
    assert result["missing_pages"] == [999]
    assert "| Revenue | 1,234 | 1,100 |" in result["pages"][0]["text"]


def test_read_report_pages_normalizes_string_page_numbers_from_model_tool_calls() -> None:
    result = read_report_pages(["3"], ["revenue"], _context())

    assert result["status"] == "success"
    assert result["pages"][0]["page"] == 3


def test_read_report_pages_can_return_focused_windows() -> None:
    result = read_report_pages([3], ["revenue"], _context())

    assert result["status"] == "success"
    assert result["pages"][0]["matched_phrases"] == ["revenue"]
    assert "(in millions, except per-share amounts)" in result["pages"][0]["text"]
    assert "| Revenue | 1,234 | 1,100 |" in result["pages"][0]["text"]


def test_submit_multi_kpi_returns_retryable_validation_errors() -> None:
    context = _context()
    invalid_page = _evidence()
    invalid_page["page"] = 999
    result = submit_multi_kpi_extraction(
        "WRONG",
        "usd",
        None,
        [invalid_page, _evidence()],
        context,
    )

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert {error["field"] for error in result["validation_errors"]} == {
        "ticker",
        "reporting_currency",
        "kpis.0.page",
        "kpis.1",
    }
    assert MULTI_KPI_RESULT_STATE_KEY not in context.state


def test_submit_multi_kpi_stores_ledger_result_in_state() -> None:
    context = _context()
    result = submit_multi_kpi_extraction(
        "ACME",
        "USD",
        "Values reported in millions.",
        [_evidence()],
        context,
    )

    assert result == {
        "status": "success",
        "result": {
            "ticker": "ACME",
            "reporting_currency": "USD",
            "units_note": "Values reported in millions.",
            "kpis": [{"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0}],
        },
    }
    assert context.state[MULTI_KPI_RESULT_STATE_KEY] == result["result"]
    assert context.state[MULTI_KPI_AUDIT_STATE_KEY]["kpis"][0]["value_verbatim"] == "1,234"
    assert context.actions.skip_summarization is True


def test_submit_multi_kpi_rejects_empty_rows_without_recorded_kpis() -> None:
    context = _context()

    result = submit_multi_kpi_extraction("ACME", None, None, [], context)

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert result["validation_errors"][0]["field"] == "kpis"
    assert "absent/ambiguous coverage for every KPI" in result["validation_errors"][0][
        "message"
    ]
    assert MULTI_KPI_RESULT_STATE_KEY not in context.state
    assert context.actions.skip_summarization is None


def test_record_multi_kpi_progress_merges_kpis_and_deduplicates_notes() -> None:
    context = _context()

    first = record_multi_kpi_progress(
        "USD",
        "Values reported in millions.",
        [_evidence()],
        [
            {
                "category": "evidence",
                "text": "Primary income statement contains revenue.",
                "pages": [3, 3],
            }
        ],
        context,
    )
    second = record_multi_kpi_progress(
        None,
        None,
        [
            _evidence(
                kpi="operating_income",
                value_verbatim="210",
                line_label="Operating income",
            ),
            _evidence(),
        ],
        [
            {
                "category": "evidence",
                "text": "Primary income statement contains revenue.",
                "pages": [3],
            },
            {
                "category": "scope",
                "text": "Use the consolidated statement.",
                "pages": [3],
            },
        ],
        context,
    )
    current = query_multi_kpi_progress("all", context)

    assert first == {
        "status": "success",
        "kpi_count": 1,
        "coverage_count": 1,
        "status_counts": {"found": 1},
        "note_count": 1,
        "added_kpi_count": 1,
        "updated_kpi_count": 0,
        "added_note_count": 1,
        "normalization_corrections": [],
    }
    assert second == {
        "status": "success",
        "kpi_count": 2,
        "coverage_count": 2,
        "status_counts": {"found": 2},
        "note_count": 2,
        "added_kpi_count": 1,
        "updated_kpi_count": 1,
        "added_note_count": 1,
        "normalization_corrections": [],
    }
    assert current["status"] == "success"
    assert current["view"] == "all"
    assert current["kpi_count"] == 2
    assert current["note_count"] == 2
    assert current["record"]["reporting_currency"] == "USD"
    assert current["record"]["units_note"] == "Values reported in millions."
    assert [item["kpi"] for item in current["record"]["kpis"]] == [
        "operating_income",
        "revenue",
    ]
    assert [item["value"] for item in current["record"]["kpis"]] == [
        210_000_000.0,
        1_234_000_000.0,
    ]
    assert all(item["status"] == "found" for item in current["record"]["kpis"])
    assert current["record"]["notes"] == [
        {
            "category": "evidence",
            "text": "Primary income statement contains revenue.",
            "pages": [3],
        },
        {
            "category": "scope",
            "text": "Use the consolidated statement.",
            "pages": [3],
        },
    ]
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY] == current["record"]


def test_query_multi_kpi_progress_supports_bounded_views() -> None:
    context = _context()
    record_multi_kpi_progress(
        "USD",
        None,
        [_evidence()],
        [{"category": "todo", "text": "Check the cash-flow statement.", "pages": []}],
        context,
    )

    kpis = query_multi_kpi_progress("kpis", context)
    notes = query_multi_kpi_progress("notes", context)

    assert "notes" not in kpis["record"]
    assert notes["record"] == {
        "ticker": "ACME",
        "notes": [{"category": "todo", "text": "Check the cash-flow statement.", "pages": []}],
    }


def test_record_multi_kpi_progress_rejects_invalid_note_without_mutating_state() -> None:
    context = _context()

    result = record_multi_kpi_progress(
        None,
        None,
        [],
        [{"category": "evidence", "text": "Invalid page.", "pages": [999]}],
        context,
    )

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert result["validation_errors"] == [
        {
            "field": "notes.0.pages",
            "message": "pages do not exist in the report: [999]",
        }
    ]
    assert MULTI_KPI_WORK_RECORD_STATE_KEY not in context.state


def test_submit_multi_kpi_combines_recorded_and_final_rows() -> None:
    context = _context()
    record_multi_kpi_progress(
        "USD",
        "Values reported in millions.",
        [_evidence()],
        [{"category": "decision", "text": "Use consolidated revenue.", "pages": [3]}],
        context,
    )

    result = submit_multi_kpi_extraction(
        "ACME",
        None,
        None,
        [
            _evidence(
                kpi="operating_income",
                value_verbatim="210",
                line_label="Operating income",
            )
        ],
        context,
    )

    assert result["status"] == "success"
    assert result["result"]["reporting_currency"] == "USD"
    assert result["result"]["kpis"] == [
        {"kpi": "operating_income", "fiscal_year": 2023, "value": 210_000_000.0},
        {"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0},
    ]


def test_submit_multi_kpi_accepts_work_record_without_final_rows() -> None:
    context = _context()
    record_multi_kpi_progress(
        "USD",
        None,
        [_evidence()],
        [{"category": "unit", "text": "Statement is in millions.", "pages": [3]}],
        context,
    )

    result = submit_multi_kpi_extraction("ACME", None, None, [], context)

    assert result["status"] == "success"
    assert result["result"] == {
        "ticker": "ACME",
        "reporting_currency": "USD",
        "units_note": None,
        "kpis": [{"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0}],
    }
    assert "notes" not in result["result"]


def test_record_multi_kpi_progress_derives_scale_from_exact_unit_text() -> None:
    context = _context()
    evidence = _evidence(unit_scale="units")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    assert result["normalization_corrections"] == [
        {
            "kpi": "revenue",
            "fiscal_year": 2023,
            "field": "unit_scale",
            "submitted": "units",
            "used": "millions",
            "source": "unit_text",
        }
    ]
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["value"] == 1_234_000_000.0
    assert recorded["unit_scale"] == "millions"
    assert recorded["normalization"]["multiplier"] == 1_000_000.0


def test_record_multi_kpi_progress_rejects_model_calculated_values() -> None:
    context = _context()
    evidence = _evidence()
    evidence["value"] = 1_234_000_000

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {"field": "kpis.0.value", "message": "Extra inputs are not permitted"}
    ]
    assert MULTI_KPI_WORK_RECORD_STATE_KEY not in context.state


def test_record_multi_kpi_progress_converts_eps_cents_to_currency_per_share() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] += (
        "\nEPS in cents\n| Basic earnings per share | (0.19) | (0.15) |"
    )
    evidence = _evidence(
        kpi="eps_basic",
        value_verbatim="(0.19)",
        line_label="Basic earnings per share",
        unit_scale="per_share",
        unit_text="EPS in cents",
    )

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["unit_scale"] == "currency_subunits_per_share"
    assert recorded["value"] == pytest.approx(-0.0019)


def test_record_multi_kpi_progress_validates_one_line_html_table_rows() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] = """\
<h2>Consolidated Statements of Operations</h2><p>(in millions)</p>
<table><tr><th>Line item</th><th>2023</th></tr><tr><td>Net<br>revenue</td>
<td>6,858</td></tr><tr><td>Operating income</td><td>700</td></tr></table>
"""
    evidence = _evidence(
        value_verbatim="6,858",
        line_label="Net revenue",
        unit_text="(in millions)",
    )

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == (
        6_858_000_000.0
    )


def test_record_multi_kpi_progress_distinguishes_explicit_zero_from_missing() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] += "\n| Long-term debt | - | 20 |"
    zero = _evidence(
        kpi="long_term_debt_noncurrent",
        value_verbatim="-",
        line_label="Long-term debt",
        status="explicit_zero",
    )

    recorded = record_multi_kpi_progress(None, None, [zero], [], context)
    result = submit_multi_kpi_extraction("ACME", None, None, [], context)

    assert recorded["status_counts"] == {"explicit_zero": 1}
    assert result["status"] == "success"
    assert result["result"]["kpis"] == [
        {"kpi": "long_term_debt_noncurrent", "fiscal_year": 2023, "value": 0.0}
    ]


def test_submit_multi_kpi_accepts_fully_covered_all_missing_report() -> None:
    context = _context()
    coverage = [
        {"kpi": kpi, "fiscal_year": 2023, "status": "absent"} for kpi in KPI_KEYS
    ]

    result = submit_multi_kpi_extraction("ACME", None, None, coverage, context)

    assert result == {
        "status": "success",
        "result": {
            "ticker": "ACME",
            "reporting_currency": None,
            "units_note": None,
            "kpis": [],
        },
    }


def test_submit_needle_returns_feedback_without_mutating_state() -> None:
    context = _context()

    result = submit_needle_extraction(
        True,
        1_234,
        "1,234",
        "millions",
        3,
        context,
    )

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert result["validation_errors"][0]["field"] == "value"
    assert NEEDLE_RESULT_STATE_KEY not in context.state


def test_submit_needle_rejects_a_page_outside_the_state_report() -> None:
    context = _context()

    result = submit_needle_extraction(
        True,
        1_234_000_000,
        "1,234",
        "millions",
        999,
        context,
    )

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "page",
            "message": "page must exist in the report stored in session state",
        }
    ]
    assert NEEDLE_RESULT_STATE_KEY not in context.state


def test_submit_needle_stores_ledger_result_in_state() -> None:
    context = _context()

    result = submit_needle_extraction(
        True,
        1_234_000_000,
        "1,234",
        "millions",
        3,
        context,
    )

    assert result == {
        "status": "success",
        "result": {
            "found": True,
            "value": 1_234_000_000.0,
            "value_verbatim": "1,234",
            "unit_scale": "millions",
            "page": 3,
        },
    }
    assert context.state[NEEDLE_RESULT_STATE_KEY] == result["result"]
