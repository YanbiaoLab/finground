from pathlib import Path
from types import SimpleNamespace

import pytest

from finground.documents import Report
from finground.kpis import KPI_KEYS
from finground.sec_facts import (
    SEC_FACTS_ENABLED_STATE_KEY,
    SEC_FACTS_STATE_KEY,
)
from finground.tools import (
    MULTI_KPI_ALLOW_PARTIAL_STATE_KEY,
    MULTI_KPI_AUDIT_STATE_KEY,
    MULTI_KPI_PREPARED_STATE_KEY,
    MULTI_KPI_REQUESTED_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    build_report_state,
    finalize_multi_kpi_report,
    find_kpi_source_candidates,
    get_report_info,
    inspect_primary_statements,
    prepare_multi_kpi_report,
    query_multi_kpi_progress,
    read_report_pages,
    record_multi_kpi_progress,
    search_kpi_report,
    search_report,
    submit_multi_kpi_extraction,
)
from finground.tools.submission import _line_contains_number, _semantic_row_error

REPORT = Path(__file__).parent / "fixtures" / "ledger" / "report.mmd"


def _context() -> SimpleNamespace:
    report = Report("NYSE_ACME_2023", "NYSE", "ACME", 2023, REPORT.read_text())
    state = {"report": build_report_state(report)}
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


def _absent_coverage(*excluded: str) -> list[dict]:
    return [
        {"kpi": kpi, "fiscal_year": 2023, "status": "absent"}
        for kpi in KPI_KEYS
        if kpi not in excluded
    ]


@pytest.mark.parametrize(
    "label",
    ("Total Income (Loss)", "Total revenues and other income", "Net portfolio income"),
)
def test_revenue_semantics_accepts_printed_mortgage_reit_aggregates(label: str) -> None:
    assert _semantic_row_error("revenue", label, "found") is None


@pytest.mark.parametrize("label", ("Interest income", "Net interest income", "Net income"))
def test_revenue_semantics_rejects_reit_components_and_bottom_line(label: str) -> None:
    assert _semantic_row_error("revenue", label, "found") is not None


def test_number_evidence_accepts_markdown_heading_and_adjacent_sentence() -> None:
    assert _line_contains_number(
        "## Debt maturities\n\nOur long-term debt matures as follows: $440 million in 2018.",
        "Debt maturities",
        "440",
    )


def test_capex_semantics_accepts_property_additions_on_cash_flow_statement() -> None:
    assert (
        _semantic_row_error(
            "capex",
            "Additions to oil and gas properties",
            "found",
            "Consolidated Statements of Cash Flows",
        )
        is None
    )


def test_capex_semantics_rejects_property_additions_without_cash_flow_context() -> None:
    assert (
        _semantic_row_error("capex", "Additions to oil and gas properties", "found")
        is not None
    )


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
    assert result["statement_pages"] == {
        "income_statement": [3],
        "balance_sheet": [],
        "cash_flow_statement": [],
    }


def test_get_report_info_classifies_untitled_primary_statement_tables() -> None:
    report = Report(
        "ASX_ACME_2023",
        "ASX",
        "ACME",
        2023,
        """\
<table><tr><td></td><td>2023</td></tr><tr><td>Revenue</td><td>10</td></tr>
<tr><td>Basic earnings per share</td><td>1</td></tr>
<tr><td>Diluted earnings per share</td><td>1</td></tr></table>
<--- Page Split --->
<table><tr><td>Current assets</td><td>10</td></tr><tr><td>Total assets</td><td>20</td></tr>
<tr><td>Current liabilities</td><td>5</td></tr>
<tr><td>Total liabilities</td><td>8</td></tr></table>
<--- Page Split --->
<table><tr><td>Cash flows from operating activities</td><td>10</td></tr>
<tr><td>Cash flows from investing activities</td><td>(4)</td></tr></table>
""",
    )
    context = SimpleNamespace(state={"report": build_report_state(report)})

    result = get_report_info(context)

    assert result["statement_pages"] == {
        "income_statement": [1],
        "balance_sheet": [2],
        "cash_flow_statement": [3],
    }


def test_get_report_info_classifies_consolidated_cash_flow_title() -> None:
    report = Report(
        "NASDAQ_ACME_2023",
        "NASDAQ",
        "ACME",
        2023,
        """\
## ACME, INC.
CONSOLIDATED STATEMENTS OF CASH FLOWS
(In thousands)
<table><tr><td></td><td>2023</td></tr>
<tr><td>Operating activities</td><td>10</td></tr>
<tr><td>Investing activities</td><td>(4)</td></tr>
<tr><td>Financing activities</td><td>2</td></tr></table>
""",
    )
    context = SimpleNamespace(state={"report": build_report_state(report)})

    result = get_report_info(context)

    assert result["statement_pages"]["cash_flow_statement"] == [1]


def test_get_report_info_classifies_consolidated_financial_position_title() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
## CONSOLIDATED STATEMENTS OF FINANCIAL POSITION
(In thousands)
<table><tr><td></td><td>2023</td></tr>
<tr><td>Current assets</td><td>10</td></tr>
<tr><td>Total assets</td><td>20</td></tr>
<tr><td>Current liabilities</td><td>5</td></tr>
<tr><td>Total liabilities</td><td>8</td></tr></table>
""",
    )
    context = SimpleNamespace(state={"report": build_report_state(report)})

    result = get_report_info(context)

    assert result["statement_pages"]["balance_sheet"] == [1]


def test_get_report_info_includes_adjacent_titled_continuation_page() -> None:
    report = Report(
        "NASDAQ_ACME_2023",
        "NASDAQ",
        "ACME",
        2023,
        """\
## Consolidated Statements of Cash Flows
<table><tr><td></td><td>2023</td></tr>
<tr><td>Operating activities</td><td>10</td></tr></table>
<--- Page Split --->
## Consolidated Statements of Cash Flows (continued)
<table><tr><td></td><td>2023</td></tr>
<tr><td>Investing activities</td><td>(4)</td></tr>
<tr><td>Financing activities</td><td>2</td></tr></table>
""",
    )
    context = SimpleNamespace(state={"report": build_report_state(report)})

    result = get_report_info(context)

    assert result["statement_pages"]["cash_flow_statement"] == [1, 2]


def test_inspect_primary_statements_returns_classified_pages_and_source_cells() -> None:
    result = inspect_primary_statements(_context())

    assert result["status"] == "success"
    assert result["statement_pages"]["income_statement"] == [3]
    assert [page["page"] for page in result["pages"]] == [3]
    assert result["pages"][0]["statement_group"] == "income_statement"
    assert result["source_cell_count"] > 0
    assert any(cell["row_label"] == "Revenue" for cell in result["pages"][0]["source_cells"])


def test_prepare_report_keeps_full_pages_out_of_common_agent_result() -> None:
    context = _context()

    result = prepare_multi_kpi_report(context)

    assert result["status"] == "success"
    assert result["report_id"] == "NYSE_ACME_2023"
    assert result["fiscal_year"] == 2023
    assert result["source_cell_count"] > 0
    assert "pages" not in result
    assert context.state[MULTI_KPI_PREPARED_STATE_KEY] == result
    assert prepare_multi_kpi_report(context)["reused"] is True


def test_prepare_report_inherits_all_monetary_values_document_unit() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
All monetary values, other than per share amounts, are stated in millions of U.S. dollars unless otherwise specified.
<--- Page Split --->
## Consolidated balance sheets
<table><tr><td></td><td>2023</td></tr>
<tr><td>Current portion of long-term debt</td><td>162</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )

    prepare_multi_kpi_report(context)
    result = find_kpi_source_candidates("long_term_debt_current", context)

    candidate = next(
        item for item in result["candidates"] if "current portion" in item["row_label"].lower()
    )
    assert candidate["unit_text"].startswith("All monetary values")
    assert candidate["unit_page"] == 1
    assert candidate["unit_scope"] == "document"


def test_prepare_structured_facts_finalize_without_specialist_rows(
    monkeypatch,
) -> None:
    context = _context()
    context.state[SEC_FACTS_ENABLED_STATE_KEY] = True
    context.state[MULTI_KPI_REQUESTED_STATE_KEY] = ["revenue"]
    monkeypatch.setattr(
        "finground.tools.report.resolve_sec_kpis",
        lambda *_args, **_kwargs: {
            "status": "success",
            "source": "https://data.sec.gov/example",
            "values": {
                "revenue": {
                    "value": 1_234_000_000.0,
                    "concept": "Revenues",
                },
            },
        },
    )

    prepared = prepare_multi_kpi_report(context)
    progress = query_multi_kpi_progress("kpis", context)
    finalized = finalize_multi_kpi_report(context)

    assert prepared["structured_kpis"] == ["revenue"]
    assert progress["pending_kpis"] == []
    assert finalized["status"] == "success"
    assert finalized["result"]["kpis"] == [
        {"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0},
    ]
    assert (
        context.state[MULTI_KPI_AUDIT_STATE_KEY]["structured_facts"]["revenue"]["source"]
        == "https://data.sec.gov/example"
    )
    assert context.state[MULTI_KPI_AUDIT_STATE_KEY]["structured_sources"] == {
        "sec_company_facts": {
            "status": "success",
            "source": "https://data.sec.gov/example",
            "cik": None,
        },
    }


def test_kpi_candidate_lookup_returns_only_compact_ranked_source_cells() -> None:
    context = _context()

    result = find_kpi_source_candidates("revenue", context)

    assert result["status"] == "success"
    assert result["kpi"] == "revenue"
    assert result["fiscal_year"] == 2023
    assert result["candidates"][0]["row_label"] == "Revenue"
    assert result["candidates"][0]["source_id"] == "p3:t0:r1:c1"
    assert "text" not in result["candidates"][0]


def test_apostrophe_thousands_in_table_header_is_traceable_unit_evidence() -> None:
    report = Report(
        "LSE_ACME_2023",
        "LSE",
        "ACME",
        2023,
        """\
## Consolidated income statement
<table><tr><td></td><td>2023 (&#x27;000</td><td>2022 (&#x27;000</td></tr>
<tr><td>Revenue</td><td>301,389</td><td>108,449</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )

    result = read_report_pages([1], [], context)
    revenue = next(
        cell for cell in result["pages"][0]["source_cells"] if cell["row_label"] == "Revenue"
    )

    assert revenue["unit_text"] == "'000"


def test_primary_statement_without_header_uses_first_value_as_report_year() -> None:
    report = Report(
        "NASDAQ_ACME_2023",
        "NASDAQ",
        "ACME",
        2023,
        """\
## Consolidated Statements of Operations
For the years ended December 31, 2023 and 2022
(in thousands)
<table><tr><td>Revenue</td><td>$</td><td>-</td><td>$</td><td>-</td></tr>
<tr><td>General and administrative</td><td>6,980</td><td>1,748</td></tr>
<tr><td>Operating income</td><td>(100)</td><td>50</td></tr>
<tr><td>Net income</td><td>(120)</td><td>40</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )

    result = read_report_pages([1], [], context)
    cells = result["pages"][0]["source_cells"]
    revenue = next(cell for cell in cells if cell["row_label"] == "Revenue")
    sga = next(cell for cell in cells if cell["row_label"] == "General and administrative")

    assert revenue["status"] == "explicit_zero"
    assert revenue["value_verbatim"] == "-"
    assert revenue["year_inferred"] is True
    assert sga["value_verbatim"] == "6,980"
    assert sga["year_label"] == "2023 (first value column)"
    recorded = record_multi_kpi_progress(
        "USD",
        None,
        [
            {
                "kpi": "sga_expense",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": sga["source_id"],
            }
        ],
        [],
        context,
    )
    assert recorded["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == 6_980_000


def test_record_multi_kpi_progress_rejects_source_id_for_different_number() -> None:
    context = _context()
    page = read_report_pages([3], ["Revenue"], context)
    revenue = next(
        cell for cell in page["pages"][0]["source_cells"] if cell["row_label"] == "Revenue"
    )

    result = record_multi_kpi_progress(
        "USD",
        None,
        [
            {
                "kpi": "revenue",
                "fiscal_year": 2023,
                "status": "found",
                "value_verbatim": "999",
                "source_id": revenue["source_id"],
            }
        ],
        [],
        context,
    )

    assert result["status"] == "partial_success"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.source_id",
            "message": (
                "source_id points to a different printed number; omit source_id for prose "
                "evidence or use the matching source cell"
            ),
        }
    ]


def test_statement_classifier_includes_adjacent_continuation_page() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
## Consolidated Balance Sheets
<table><tr><td></td><td>2023</td><td>2022</td></tr>
<tr><td>Current assets</td><td>10</td><td>9</td></tr>
<tr><td>Total assets</td><td>20</td><td>18</td></tr>
<tr><td>Current liabilities</td><td>5</td><td>4</td></tr></table>
<--- Page Split --->
<table><tr><td>Total liabilities</td><td>8</td><td>7</td></tr>
<tr><td>Total stockholders' equity</td><td>12</td><td>11</td></tr></table>
""",
    )
    context = SimpleNamespace(state={"report": build_report_state(report)})

    result = get_report_info(context)

    assert result["statement_pages"]["balance_sheet"] == [1, 2]


def test_period_end_share_count_is_extracted_from_verbose_balance_sheet_label() -> None:
    report = Report(
        "NASDAQ_ACME_2023",
        "NASDAQ",
        "ACME",
        2023,
        """\
## Consolidated Balance Sheets
<table><tr><td></td><td>December 31, 2023</td><td>December 31, 2022</td></tr>
<tr><td>Current assets</td><td>10</td><td>9</td></tr>
<tr><td>Total assets</td><td>20</td><td>18</td></tr>
<tr><td>Total liabilities</td><td>8</td><td>7</td></tr>
<tr><td>Common stock, $0.01 par value; 200,000,000 shares authorized;
55,831,549 and 49,153,463 shares issued and outstanding, respectively</td>
<td>558</td><td>492</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )

    candidates = find_kpi_source_candidates("shares_outstanding", context)["candidates"]
    embedded = next(
        candidate for candidate in candidates if candidate["source_id"].startswith("p1:shares:")
    )
    result = record_multi_kpi_progress(
        "USD",
        None,
        [
            {
                "kpi": "shares_outstanding",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": embedded["source_id"],
            }
        ],
        [],
        context,
    )

    assert embedded["value_verbatim"] == "55,831,549"
    assert result["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == 55_831_549


def test_finalization_applies_ledger_aligned_gross_profit_identity() -> None:
    context = _context()
    context.state[MULTI_KPI_REQUESTED_STATE_KEY] = ["gross_profit"]
    context.state[MULTI_KPI_WORK_RECORD_STATE_KEY] = {
        "ticker": "ACME",
        "reporting_currency": "USD",
        "units_note": "millions",
        "kpis": [
            {
                "kpi": "revenue",
                "fiscal_year": 2023,
                "status": "found",
                "value_verbatim": "1,234",
                "unit_scale": "millions",
                "page": 3,
                "line_label": "Revenue",
                "year_label": "2023",
                "value": 1_234_000_000,
            },
            {
                "kpi": "cost_of_revenue",
                "fiscal_year": 2023,
                "status": "found",
                "value_verbatim": "700",
                "unit_scale": "millions",
                "page": 3,
                "line_label": "Cost of revenue",
                "year_label": "2023",
                "value": 700_000_000,
            },
            {
                "kpi": "gross_profit",
                "fiscal_year": 2023,
                "status": "absent",
            },
        ],
        "notes": [],
    }

    result = finalize_multi_kpi_report(context)

    assert result["status"] == "success"
    assert context.state[MULTI_KPI_RESULT_STATE_KEY]["kpis"] == [
        {
            "kpi": "gross_profit",
            "fiscal_year": 2023,
            "value": 534_000_000.0,
        }
    ]
    assert context.state[MULTI_KPI_AUDIT_STATE_KEY]["derivations"] == [
        "gross_profit = revenue - cost_of_revenue"
    ]


def test_unlabeled_structural_total_is_source_backed_evidence() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
## Consolidated Statements of Cash Flows
(thousands of dollars)
<table><tr><td>Year ended</td><td>2023</td><td>2022</td></tr>
<tr><td>Operating activities</td><td></td><td></td></tr>
<tr><td>Adjustments for:</td><td></td><td></td></tr>
<tr><td>Depreciation and amortization</td><td>100</td><td>90</td></tr>
<tr><td>Change in working capital</td><td>20</td><td>10</td></tr>
<tr><td></td><td>325,208</td><td>247,365</td></tr>
<tr><td>Financing activities</td><td></td><td></td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )

    read_result = read_report_pages([1], [], context)
    structural_cell = next(
        cell
        for cell in read_result["pages"][0]["source_cells"]
        if cell["source_id"] == "p1:t0:r5:c1"
    )

    assert structural_cell["printed_row_label"] is None
    assert structural_cell["row_role"] == "unlabeled_numeric"
    assert structural_cell["section_label"] == "Operating activities"
    assert structural_cell["next_label"] == "Financing activities"

    record_result = record_multi_kpi_progress(
        "USD",
        "thousands of dollars",
        [
            {
                "kpi": "operating_cash_flow",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": structural_cell["source_id"],
            }
        ],
        [],
        context,
    )

    assert record_result["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == 325_208_000


def test_source_backed_explicit_zero_does_not_require_unit_text() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
## Consolidated Balance Sheets
<table><tr><td>Year ended</td><td>2023</td><td>2022</td></tr>
<tr><td>Inventory</td><td>—</td><td>10</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    read_result = read_report_pages([1], [], context)
    source_cell = next(
        cell for cell in read_result["pages"][0]["source_cells"] if cell["row_label"] == "Inventory"
    )

    result = record_multi_kpi_progress(
        "USD",
        None,
        [
            {
                "kpi": "inventory",
                "fiscal_year": 2023,
                "status": "explicit_zero",
                "source_id": source_cell["source_id"],
            }
        ],
        [],
        context,
    )

    assert result["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == 0.0


def test_source_backed_monetary_value_defaults_to_units_without_scale_header() -> None:
    report = Report(
        "NASDAQ_ACME_2023",
        "NASDAQ",
        "ACME",
        2023,
        """\
## Consolidated Balance Sheets
<table><tr><td>Year ended</td><td>2023</td><td>2022</td></tr>
<tr><td>Line of credit</td><td>$ 16,914,594</td><td>$ 6,482,848</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    read_result = read_report_pages([1], [], context)
    source_cell = next(
        cell
        for cell in read_result["pages"][0]["source_cells"]
        if cell["row_label"] == "Line of credit"
    )

    result = record_multi_kpi_progress(
        "USD",
        "No scale stated; values are actual dollars",
        [
            {
                "kpi": "short_term_borrowings",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": source_cell["source_id"],
            }
        ],
        [],
        context,
    )

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["unit_scale"] == "units"
    assert recorded["value"] == 16_914_594


def test_source_backed_value_inherits_scale_from_corroborating_source_cell() -> None:
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        """\
## Consolidated Balance Sheets
<table><tr><td>Year ended</td><td>2023</td></tr>
<tr><td>Borrowings due within one year</td><td>243</td></tr></table>
<--- Page Split --->
## Debt note
(Dollars in millions)
<table><tr><td>Year ended</td><td>2023</td></tr>
<tr><td>Short-term borrowings</td><td>243</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    read_result = read_report_pages([1, 2], [], context)
    source_cell = next(
        cell
        for cell in read_result["pages"][0]["source_cells"]
        if cell["row_label"] == "Borrowings due within one year"
    )

    result = record_multi_kpi_progress(
        "USD",
        "Dollars in millions",
        [
            {
                "kpi": "short_term_borrowings",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": source_cell["source_id"],
            }
        ],
        [],
        context,
    )

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["unit_text"] == "(Dollars in millions)"
    assert recorded["unit_page"] == 2
    assert recorded["value"] == 243_000_000


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


def test_search_kpi_report_prefers_the_kpis_primary_statement() -> None:
    result = search_kpi_report(
        "revenue",
        "financial result",
        ["revenue"],
        2023,
        3,
        _context(),
    )

    assert result["status"] == "success"
    assert result["preferred_statement_group"] == "income_statement"
    assert result["preferred_pages"] == [3]
    assert result["results"][0]["page"] == 3


def test_document_level_unit_is_inherited_with_traceable_source_page() -> None:
    report = Report(
        "NYSE_UNIT_2023",
        "NYSE",
        "UNIT",
        2023,
        (
            "All currency figures expressed herein are expressed in thousands, "
            "except share or per share amounts.\n"
            "<--- Page Split --->\n"
            "# Note — Accounts payable\n"
            "<table><tr><td></td><td>2023</td></tr>"
            "<tr><td>Accounts payable</td><td>1,534</td></tr></table>"
        ),
    )
    context = SimpleNamespace(
        state={
            "report": build_report_state(report),
            SEC_FACTS_ENABLED_STATE_KEY: False,
        },
        actions=SimpleNamespace(skip_summarization=None),
    )

    result = find_kpi_source_candidates("accounts_payable", context)

    assert result["status"] == "success"
    candidate = result["candidates"][0]
    assert candidate["unit_scope"] == "document"
    assert candidate["unit_page"] == 1
    assert "in thousands" in candidate["unit_text"]

    recorded = record_multi_kpi_progress(
        None,
        None,
        [
            {
                "kpi": "accounts_payable",
                "fiscal_year": 2023,
                "status": "found",
                "source_id": candidate["source_id"],
            }
        ],
        [],
        context,
    )

    assert recorded["status"] == "success"
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == 1_534_000


def test_search_report_reads_only_state_backed_report() -> None:
    result = search_report("What was revenue?", ["revenue"], 2023, 5, _context())

    assert result["status"] == "success"
    assert result["report_id"] == "NYSE_ACME_2023"
    assert result["results"][0]["page"] == 3
    assert result["results"][0]["matched_phrases"] == ["revenue"]
    assert "(in millions, except per-share amounts)" in result["results"][0]["snippet"]
    assert "score" not in result["results"][0]


def test_search_report_snippet_centers_match_in_one_line_html() -> None:
    context = _context()
    filler = "".join(f"<tr><td>Filler {index}</td><td>{index}</td></tr>" for index in range(80))
    context.state["report"]["pages"][2]["text"] = (
        "<table>" + filler + "<tr><td>Net revenue</td><td>6,858</td></tr>" + "</table>"
    )

    result = search_report("revenue", ["Net revenue"], 2023, 1, context)

    assert "Net revenue" in result["results"][0]["snippet"]
    assert "6,858" in result["results"][0]["snippet"]


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


def test_read_report_pages_focuses_one_line_html_around_matching_row() -> None:
    context = _context()
    filler = "".join(f"<tr><td>Filler {index}</td><td>{index}</td></tr>" for index in range(400))
    context.state["report"]["pages"][2]["text"] = (
        "<p>(in millions)</p><table>"
        + filler
        + "<tr><td>Net revenue</td><td>6,858</td></tr>"
        + "</table>"
    )

    result = read_report_pages([3], ["Net revenue"], context)

    assert "Net revenue" in result["pages"][0]["text"]
    assert "6,858" in result["pages"][0]["text"]


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
        [_evidence(), *_absent_coverage("revenue")],
        context,
    )

    assert result == {
        "status": "success",
        "completion_status": "complete",
        "coverage_count": len(KPI_KEYS),
        "pending_kpis": [],
        "result": {
            "ticker": "ACME",
            "reporting_currency": "USD",
            "units_note": "Values reported in millions.",
            "kpis": [{"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0}],
        },
    }
    assert context.state[MULTI_KPI_RESULT_STATE_KEY] == result["result"]
    revenue_audit = next(
        item
        for item in context.state[MULTI_KPI_AUDIT_STATE_KEY]["kpis"]
        if item["kpi"] == "revenue"
    )
    assert revenue_audit["value_verbatim"] == "1,234"
    assert context.actions.skip_summarization is True


def test_submit_multi_kpi_rejects_empty_rows_without_recorded_kpis() -> None:
    context = _context()

    result = submit_multi_kpi_extraction("ACME", None, None, [], context)

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert result["validation_errors"][0]["field"] == "kpis"
    assert "complete coverage for every KPI" in result["validation_errors"][0]["message"]
    assert result["pending_kpis"] == list(KPI_KEYS)
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
        "pending_count": len(KPI_KEYS) - 1,
        "pending_kpis": [kpi for kpi in KPI_KEYS if kpi != "revenue"],
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
        "pending_count": len(KPI_KEYS) - 2,
        "pending_kpis": [kpi for kpi in KPI_KEYS if kpi not in {"operating_income", "revenue"}],
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
    assert current["coverage_count"] == 2
    assert current["pending_count"] == len(KPI_KEYS) - 2
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
    assert kpis["pending_kpis"] == [kpi for kpi in KPI_KEYS if kpi != "revenue"]
    assert notes["record"] == {
        "ticker": "ACME",
        "notes": [{"category": "todo", "text": "Check the cash-flow statement.", "pages": []}],
    }


def test_structured_facts_count_as_covered_without_report_agent_rows() -> None:
    context = _context()
    context.state[SEC_FACTS_STATE_KEY] = {
        "status": "success",
        "source": "https://data.sec.gov/example",
        "values": {
            "revenue": {
                "value": 1_234_000_000.0,
                "concept": "Revenues",
            }
        },
    }

    result = query_multi_kpi_progress("kpis", context)

    assert result["coverage_count"] == 1
    assert result["pending_kpis"] == [kpi for kpi in KPI_KEYS if kpi != "revenue"]


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


def test_record_multi_kpi_progress_saves_kpis_when_a_note_is_invalid() -> None:
    context = _context()

    result = record_multi_kpi_progress(
        None,
        None,
        [_evidence()],
        [{"category": "evidence", "text": "Invalid page.", "pages": [999]}],
        context,
    )

    assert result["status"] == "partial_success"
    assert result["added_kpi_count"] == 1
    assert result["added_note_count"] == 0
    assert result["validation_errors"] == [
        {
            "field": "notes.0.pages",
            "message": "pages do not exist in the report: [999]",
        }
    ]
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["kpi"] == "revenue"


def test_record_multi_kpi_progress_saves_kpis_when_currency_is_invalid() -> None:
    context = _context()

    result = record_multi_kpi_progress("usd", None, [_evidence()], [], context)

    assert result["status"] == "partial_success"
    assert result["added_kpi_count"] == 1
    assert result["validation_errors"] == [
        {
            "field": "reporting_currency",
            "message": "reporting_currency must be a three-letter uppercase ISO code or null",
        }
    ]
    record = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]
    assert record["reporting_currency"] is None
    assert record["kpis"][0]["kpi"] == "revenue"


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
            ),
            *_absent_coverage("operating_income", "revenue"),
        ],
        context,
    )

    assert result["status"] == "success"
    assert result["completion_status"] == "complete"
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
        [_evidence(), *_absent_coverage("revenue")],
        [{"category": "unit", "text": "Statement is in millions.", "pages": [3]}],
        context,
    )

    result = submit_multi_kpi_extraction("ACME", None, None, [], context)

    assert result["status"] == "success"
    assert result["completion_status"] == "complete"
    assert result["result"] == {
        "ticker": "ACME",
        "reporting_currency": "USD",
        "units_note": None,
        "kpis": [{"kpi": "revenue", "fiscal_year": 2023, "value": 1_234_000_000.0}],
    }
    assert "notes" not in result["result"]


def test_submit_multi_kpi_rejects_partial_nonempty_result_before_deadline() -> None:
    context = _context()

    result = submit_multi_kpi_extraction("ACME", "USD", None, [_evidence()], context)

    assert result["status"] == "error"
    assert result["retryable"] is True
    assert result["coverage_count"] == 1
    assert result["pending_kpis"] == [kpi for kpi in KPI_KEYS if kpi != "revenue"]
    assert MULTI_KPI_RESULT_STATE_KEY not in context.state


def test_submit_multi_kpi_marks_forced_partial_result_incomplete() -> None:
    context = _context()
    context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] = True

    result = submit_multi_kpi_extraction("ACME", "USD", None, [_evidence()], context)

    assert result["status"] == "success"
    assert result["completion_status"] == "incomplete"
    assert result["coverage_count"] == 1
    assert result["pending_kpis"] == [kpi for kpi in KPI_KEYS if kpi != "revenue"]
    assert context.state[MULTI_KPI_RESULT_STATE_KEY] == result["result"]


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


def test_record_multi_kpi_progress_rejects_multiplier_without_visible_unit_text() -> None:
    context = _context()
    evidence = _evidence(unit_text=None, unit_scale="millions")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.unit_text",
            "message": "scaled values require exact visible unit_text and unit_page",
        }
    ]


def test_record_rejects_generic_current_noncurrent_liabilities_as_debt() -> None:
    context = _context()
    evidence = _evidence(
        kpi="long_term_debt_current",
        value_verbatim="6.108",
        line_label="Current portion of non-current liabilities",
    )

    result = record_multi_kpi_progress("USD", "millions", [evidence], [], context)

    assert result["status"] == "partial_success"
    assert result["validation_errors"][0] == {
        "field": "kpis.0.line_label",
        "message": (
            "semantic mismatch: long_term_debt_current requires a debt, borrowing, "
            "note, or lease row; a generic current portion of all non-current "
            "liabilities is too broad"
        ),
    }


def test_record_multi_kpi_progress_rejects_units_when_page_header_is_scaled() -> None:
    context = _context()
    evidence = _evidence(unit_text=None, unit_scale="units")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.unit_text",
            "message": (
                "cited page header indicates millions; copy its exact unit_text and unit_page "
                "instead of using units"
            ),
        }
    ]


def test_record_multi_kpi_progress_accepts_units_without_a_multiplier_header() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] = context.state["report"]["pages"][2][
        "text"
    ].replace("(in millions, except per-share amounts)", "(in dollars)")
    evidence = _evidence(unit_text=None, unit_scale="units")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["unit_scale"] == "units"
    assert recorded["value"] == 1_234.0


def test_record_multi_kpi_progress_does_not_save_missing_unit_evidence() -> None:
    context = _context()
    evidence = _evidence(unit_text="(in billions)", unit_scale="billions")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.unit_text",
            "message": "unit_text was not found on unit_page",
        }
    ]
    assert MULTI_KPI_WORK_RECORD_STATE_KEY not in context.state


def test_record_multi_kpi_progress_does_not_require_descriptive_audit_fields() -> None:
    context = _context()
    evidence = _evidence()
    evidence["statement"] = None
    evidence["scope"] = None

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["statement"] is None
    assert recorded["scope"] is None


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


def test_record_multi_kpi_progress_saves_valid_rows_when_one_schema_row_is_invalid() -> None:
    context = _context()
    invalid = _evidence(kpi="operating_income", line_label="Operating income")
    invalid["value"] = 210_000_000

    result = record_multi_kpi_progress(
        None,
        None,
        [_evidence(), invalid],
        [],
        context,
    )

    assert result["status"] == "partial_success"
    assert result["added_kpi_count"] == 1
    assert result["validation_errors"] == [
        {"field": "kpis.1.value", "message": "Extra inputs are not permitted"}
    ]
    assert [item["kpi"] for item in context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"]] == [
        "revenue"
    ]


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
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["value"] == (6_858_000_000.0)


def test_record_multi_kpi_progress_rejects_value_from_wrong_year_column() -> None:
    context = _context()
    evidence = _evidence(value_verbatim="1,100")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.value_verbatim",
            "message": "number was not found in the cited fiscal-year cell",
        }
    ]
    assert MULTI_KPI_WORK_RECORD_STATE_KEY not in context.state


def test_record_multi_kpi_progress_keeps_valid_rows_when_one_row_is_invalid() -> None:
    context = _context()
    invalid = _evidence(
        kpi="operating_income",
        value_verbatim="999",
        line_label="Operating income",
    )

    result = record_multi_kpi_progress(
        "USD",
        "millions",
        [_evidence(), invalid],
        [],
        context,
    )

    assert result["status"] == "partial_success"
    assert result["retryable"] is True
    assert result["added_kpi_count"] == 1
    assert result["validation_errors"] == [
        {
            "field": "kpis.1.value_verbatim",
            "message": "number was not found in the cited fiscal-year cell",
        }
    ]
    assert [item["kpi"] for item in context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"]] == [
        "revenue"
    ]
    assert result["accepted_kpis"] == ["revenue"]
    assert result["repair_queue"] == [
        {
            "index": 1,
            "kpi": "operating_income",
            "validation_errors": [
                {
                    "field": "kpis.1.value_verbatim",
                    "message": "number was not found in the cited fiscal-year cell",
                }
            ],
        }
    ]


def test_record_multi_kpi_progress_rejects_non_report_fiscal_year() -> None:
    context = _context()
    evidence = _evidence(fiscal_year=2022, value_verbatim="1,100")

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.fiscal_year",
            "message": "fiscal_year must match report fiscal year 2023",
        }
    ]
    assert MULTI_KPI_WORK_RECORD_STATE_KEY not in context.state


def test_record_multi_kpi_progress_normalizes_interest_expense_as_positive_cost() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] += "\n| Interest expense | (113) | (101) |"
    evidence = _evidence(
        kpi="interest_expense",
        value_verbatim="(113)",
        line_label="Interest expense",
    )

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["value"] == 113_000_000.0
    assert recorded["normalization"]["sign_rule"] == "positive_magnitude"


@pytest.mark.parametrize("kpi,line_label", [("capex", "Purchase of equipment"), ("dividends_paid", "Dividends paid")])
def test_record_multi_kpi_progress_preserves_lse_outflow_sign(
    kpi: str,
    line_label: str,
) -> None:
    report = Report(
        "LSE_ACME.L_2023",
        "LSE",
        "ACME.L",
        2023,
        f"""\
## Consolidated cash flow statement
£'000
<table><tr><td></td><td>2023</td></tr>
<tr><td>{line_label}</td><td>(113)</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    evidence = {
        "kpi": kpi,
        "fiscal_year": 2023,
        "status": "found",
        "value_verbatim": "(113)",
        "unit_scale": "thousands",
        "unit_text": "£'000",
        "unit_page": 1,
        "page": 1,
        "line_label": line_label,
        "year_label": "2023",
    }

    result = record_multi_kpi_progress("GBP", None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["value"] == -113_000.0
    assert recorded["normalization"]["sign_rule"] == "as_reported"


def test_record_multi_kpi_progress_sums_verified_capex_component_sources() -> None:
    report = Report(
        "LSE_ACME.L_2023",
        "LSE",
        "ACME.L",
        2023,
        """\
## Consolidated cash flow statement
£'000
<table><tr><td></td><td>2023</td></tr>
<tr><td>Purchase of property, plant and equipment</td><td>(2,098)</td></tr>
<tr><td>Capitalised development costs and purchased software</td><td>(1,711)</td></tr>
<tr><td>Acquisition of businesses</td><td>(5,114)</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    page = read_report_pages(
        [1],
        ["Purchase of property", "Capitalised development"],
        context,
    )
    cells = {
        cell["row_label"]: cell["source_id"]
        for cell in page["pages"][0]["source_cells"]
    }
    evidence = {
        "kpi": "capex",
        "fiscal_year": 2023,
        "status": "found",
        "statement": "Consolidated cash flow statement",
        "source_ids": [
            cells["Purchase of property, plant and equipment"],
            cells["Capitalised development costs and purchased software"],
        ],
    }

    result = record_multi_kpi_progress("GBP", None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["value"] == -3_809_000.0
    assert recorded["source_ids"] == evidence["source_ids"]
    assert "Acquisition of businesses" not in recorded["line_label"]


def test_record_multi_kpi_progress_accepts_number_in_prose_on_page_with_tables() -> None:
    report = Report(
        "LSE_ACME.L_2023",
        "LSE",
        "ACME.L",
        2023,
        """\
## Share capital
For the year ended 31 August 2023
The allotted, called up and fully paid share capital is made up of 22,626,466 ordinary shares.
<table><tr><td>Share capital</td><td>226</td></tr></table>
""",
    )
    context = SimpleNamespace(
        state={"report": build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=None),
    )
    evidence = {
        "kpi": "shares_outstanding",
        "fiscal_year": 2023,
        "status": "found",
        "value_verbatim": "22,626,466",
        "unit_scale": "units",
        "page": 1,
        "line_label": (
            "The allotted, called up and fully paid share capital is made up of "
            "22,626,466 ordinary shares."
        ),
        "year_label": "31 August 2023",
    }

    result = record_multi_kpi_progress("GBP", None, [evidence], [], context)

    assert result["status"] == "success"
    recorded = context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]
    assert recorded["value"] == 22_626_466.0


def test_record_multi_kpi_progress_requires_printed_number_sign() -> None:
    context = _context()
    context.state["report"]["pages"][2]["text"] += "\n| Interest expense | (113) | (101) |"
    evidence = _evidence(
        kpi="interest_expense",
        value_verbatim="113",
        line_label="Interest expense",
    )

    result = record_multi_kpi_progress(None, None, [evidence], [], context)

    assert result["status"] == "error"
    assert result["validation_errors"] == [
        {
            "field": "kpis.0.value_verbatim",
            "message": "number was not found in the cited fiscal-year cell",
        }
    ]


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
    result = submit_multi_kpi_extraction(
        "ACME",
        None,
        None,
        _absent_coverage("long_term_debt_noncurrent"),
        context,
    )

    assert recorded["status_counts"] == {"explicit_zero": 1}
    assert result["status"] == "success"
    assert result["result"]["kpis"] == [
        {"kpi": "long_term_debt_noncurrent", "fiscal_year": 2023, "value": 0.0}
    ]


def test_submit_multi_kpi_accepts_fully_covered_all_missing_report() -> None:
    context = _context()
    coverage = [{"kpi": kpi, "fiscal_year": 2023, "status": "absent"} for kpi in KPI_KEYS]

    result = submit_multi_kpi_extraction("ACME", None, None, coverage, context)

    assert result == {
        "status": "success",
        "completion_status": "complete",
        "coverage_count": len(KPI_KEYS),
        "pending_kpis": [],
        "result": {
            "ticker": "ACME",
            "reporting_currency": None,
            "units_note": None,
            "kpis": [],
        },
    }


def test_finalize_multi_kpi_report_submits_from_recorded_state() -> None:
    context = _context()
    coverage = [{"kpi": kpi, "fiscal_year": 2023, "status": "absent"} for kpi in KPI_KEYS]
    recorded = record_multi_kpi_progress(None, None, coverage, [], context)

    result = finalize_multi_kpi_report(context)

    assert recorded["coverage_count"] == len(KPI_KEYS)
    assert result["status"] == "success"
    assert result["completion_status"] == "complete"
    assert context.state[MULTI_KPI_RESULT_STATE_KEY] == result["result"]
