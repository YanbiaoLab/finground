"""State-backed report inspection tools for FinGround agents."""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from google.adk.tools import ToolContext

from finground.documents import Page, Report, load_report_pages
from finground.kpis import KPI_ALIASES, KPI_KEYS
from finground.retrieval import expand_search_phrases, rank_pages
from finground.sec_facts import (
    SEC_FACTS_ENABLED_STATE_KEY,
    SEC_FACTS_STATE_KEY,
    resolve_sec_kpis,
)
from finground.table_evidence import extract_source_cells

MAX_SEARCH_RESULTS = 8
MAX_READ_PAGES = 3
MAX_OUTLINE_ITEMS = 80
MAX_SEARCH_SNIPPET_CHARS = 1_200
MAX_PAGE_TEXT_CHARS = 24_000
MAX_FOCUSED_TEXT_CHARS = 12_000
FOCUS_CONTEXT_LINES = 4
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
REPORT_STATE_KEY = "report"
MULTI_KPI_SOURCE_CELLS_STATE_KEY = "multi_kpi_source_cells"
MULTI_KPI_PREPARED_STATE_KEY = "multi_kpi_prepared_report"
MULTI_KPI_DOCUMENT_UNIT_STATE_KEY = "multi_kpi_document_unit"
STATEMENT_HEADING_TERMS = {
    "income_statement": (
        "consolidated statement of operations",
        "consolidated statements of operations",
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated income statement",
        "consolidated statements of earnings",
        "consolidated statement of profit or loss",
        "consolidated statement of comprehensive income",
        "statement of profit or loss",
        "statement of comprehensive income",
    ),
    "balance_sheet": (
        "consolidated balance sheet",
        "consolidated balance sheets",
        "consolidated statement of financial position",
        "consolidated statements of financial position",
        "statement of financial position",
        "statements of financial position",
    ),
    "cash_flow_statement": (
        "consolidated statement of cash flows",
        "consolidated statements of cash flows",
        "consolidated cash flow statement",
        "consolidated cash flow statements",
        "condensed consolidated statement of cash flows",
        "condensed consolidated statements of cash flows",
        "statement of consolidated cash flows",
        "statements of consolidated cash flows",
        "statement of cash flows",
        "statements of cash flows",
        "cash flow statement",
        "cash flow statements",
    ),
}
STATEMENT_STRUCTURE_TERMS = {
    "income_statement": (
        "revenue",
        "cost of",
        "operating income",
        "income from operations",
        "profit before",
        "income before",
        "net income",
        "profit for the year",
        "basic",
        "diluted",
    ),
    "balance_sheet": (
        "current assets",
        "total assets",
        "current liabilities",
        "total liabilities",
        "total equity",
        "cash and cash equivalents",
    ),
    "cash_flow_statement": (
        "cash flows from operating activities",
        "cash flows from investing activities",
        "cash flows from financing activities",
        "net cash provided by operating activities",
        "net cash used in operating activities",
        "net cash from operating activities",
    ),
}
STATEMENT_STRUCTURE_THRESHOLDS = {
    "income_statement": 3,
    "balance_sheet": 4,
    "cash_flow_statement": 2,
}
KPI_STATEMENT_GROUP = {
    **dict.fromkeys(
        (
            "revenue",
            "cost_of_revenue",
            "gross_profit",
            "rd_expense",
            "sga_expense",
            "operating_income",
            "interest_expense",
            "income_tax_expense",
            "net_income",
            "eps_basic",
            "eps_diluted",
        ),
        "income_statement",
    ),
    **dict.fromkeys(
        (
            "total_assets",
            "total_liabilities",
            "stockholders_equity",
            "stockholders_equity_incl_nci",
            "cash_and_equivalents",
            "long_term_debt_total",
            "long_term_debt_noncurrent",
            "long_term_debt_current",
            "short_term_borrowings",
            "inventory",
            "accounts_receivable",
            "accounts_payable",
            "shares_outstanding",
        ),
        "balance_sheet",
    ),
    **dict.fromkeys(
        (
            "cash_incl_restricted",
            "operating_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "capex",
            "depreciation_amortization",
            "dividends_paid",
        ),
        "cash_flow_statement",
    ),
}

if set(KPI_STATEMENT_GROUP) != set(KPI_KEYS):
    raise RuntimeError("every canonical KPI must have a primary statement group")

DOCUMENT_UNIT_RE = re.compile(
    r"""(?ix)
    (?:
        \b(?:all|unless\s+otherwise\s+(?:noted|stated),?\s+all)
        [^.\n<>]{0,160}
        \b(?:currency\s+figures|amounts|dollar\s+amounts|financial\s+information)\b
        [^.\n<>]{0,160}
        \b(?:expressed|presented|reported)\s+in\s+
        (?:thousands|millions|billions)\b
        [^.\n<>]{0,100}
      |
        \ball\s+monetary\s+values\b
        [^.\n<>]{0,160}
        \b(?:are\s+)?(?:stated|expressed|presented|reported)\s+in\s+
        (?:thousands|millions|billions)\b
        [^.\n<>]{0,100}
    )
    """
)


def _argument_error(field: str, message: str) -> dict:
    return {
        "status": "error",
        "retryable": True,
        "error": "report tool arguments are invalid",
        "validation_errors": [{"field": field, "message": message}],
    }


def _normalize_integer(
    value: object, field: str, *, allow_none: bool = False
) -> tuple[int | None, dict | None]:
    if value is None:
        if allow_none:
            return None, None
        return None, _argument_error(field, "value must be an integer")
    if isinstance(value, bool):
        return None, _argument_error(field, "value must be an integer")
    if isinstance(value, int):
        return value, None
    if isinstance(value, float) and value.is_integer():
        return int(value), None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value), None
    return None, _argument_error(field, "value must be an integer")


def _validate_phrases(value: object, field: str) -> tuple[list[str] | None, dict | None]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None, _argument_error(field, "value must be a list of strings")
    return value, None


def build_report_state(report: Report) -> dict[str, Any]:
    """Create the JSON-serializable session state used by all report tools."""
    pages = load_report_pages(report)
    return {
        "report_id": report.report_id,
        "exchange": report.exchange,
        "ticker": report.ticker,
        "year": report.year,
        "pages": [
            {
                "raw_index": page.raw_index,
                "display_number": page.display_number,
                "text": page.text,
            }
            for page in pages
        ],
    }


def report_from_state(
    tool_context: ToolContext,
) -> tuple[dict[str, Any] | None, list[Page] | None, dict | None]:
    """Load and validate the current report from ADK session state."""
    report = tool_context.state.get(REPORT_STATE_KEY)
    if not isinstance(report, dict):
        return (
            None,
            None,
            {
                "status": "error",
                "error": "report is missing from session state",
            },
        )
    raw_pages = report.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return (
            None,
            None,
            {
                "status": "error",
                "error": "report.pages is missing or empty in session state",
            },
        )
    try:
        pages = [
            Page(raw_index=int(page["raw_index"]), text=str(page["text"])) for page in raw_pages
        ]
    except (KeyError, TypeError, ValueError) as error:
        return (
            None,
            None,
            {
                "status": "error",
                "error": f"report.pages is invalid: {error}",
            },
        )
    return report, pages, None


def _normalized_text(text: str) -> str:
    visible_text = re.sub(r"<[^>]+>", " ", unescape(text))
    return " ".join(visible_text.casefold().split())


def _document_unit_evidence(pages: list[Page]) -> dict[str, Any] | None:
    for page in pages:
        visible = unescape(re.sub(r"<[^>]+>", " ", page.text))
        match = DOCUMENT_UNIT_RE.search(visible)
        if match is not None:
            return {
                "unit_text": " ".join(match.group().split()),
                "unit_page": page.display_number,
            }
    return None


def _inherit_document_unit(
    cells: list[dict[str, Any]],
    document_unit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if document_unit is None:
        return cells
    return [
        (
            {
                **cell,
                "unit_text": document_unit["unit_text"],
                "unit_page": document_unit["unit_page"],
                "unit_scope": "document",
            }
            if cell.get("unit_text") is None
            else {
                **cell,
                "unit_page": cell.get("unit_page", cell.get("page")),
                "unit_scope": "local",
            }
        )
        for cell in cells
    ]


def _has_statement_title(page_text: str, terms: tuple[str, ...]) -> bool:
    table_start = page_text.casefold().find("<table")
    title_end = table_start if 0 <= table_start <= 2_000 else 1_200
    for line in page_text[:title_end].splitlines():
        normalized_line = _normalized_text(line).lstrip("# ")
        if any(
            normalized_line == term
            or (normalized_line.startswith(term) and len(normalized_line) <= len(term) + 80)
            for term in terms
        ):
            return True
    return False


def _classified_statement_pages(pages: list[Page]) -> dict[str, list[int]]:
    candidates: dict[str, list[tuple[int, int, bool]]] = {
        statement: [] for statement in STATEMENT_HEADING_TERMS
    }
    for page in pages:
        headings = HEADING_RE.findall(page.text)
        first_heading = headings[0].casefold() if headings else ""
        is_secondary_section = first_heading.startswith(
            ("note ", "notes ", "independent auditor", "directors")
        )
        page_text = _normalized_text(page.text)
        has_table = "<table" in page.text.casefold() or "|" in page.text
        for statement, title_terms in STATEMENT_HEADING_TERMS.items():
            title_match = (
                not is_secondary_section
                and has_table
                and _has_statement_title(page.text, title_terms)
            )
            structure_score = sum(
                term in page_text for term in STATEMENT_STRUCTURE_TERMS[statement]
            )
            if title_match or (
                has_table and structure_score >= STATEMENT_STRUCTURE_THRESHOLDS[statement]
            ):
                candidates[statement].append(
                    (
                        (20 if title_match else 0) + structure_score,
                        page.display_number,
                        title_match,
                    )
                )
    classified: dict[str, list[int]] = {}
    page_by_number = {page.display_number: page for page in pages}
    for statement, items in candidates.items():
        ranked = sorted(items, key=lambda item: (-item[0], item[1]))
        if not ranked:
            classified[statement] = []
            continue
        primary_page = ranked[0][1]
        selected = {primary_page}
        selected.update(
            page
            for _score, page, title_match in ranked[1:]
            if title_match and abs(page - primary_page) == 1
        )
        for adjacent_number in (primary_page - 1, primary_page + 1):
            adjacent = page_by_number.get(adjacent_number)
            if adjacent is None:
                continue
            headings = HEADING_RE.findall(adjacent.text)
            first_heading = headings[0].casefold() if headings else ""
            if first_heading.startswith(("note ", "notes ", "independent auditor", "directors")):
                continue
            normalized = _normalized_text(adjacent.text)
            has_table = "<table" in adjacent.text.casefold() or "|" in adjacent.text
            continuation_score = sum(
                term in normalized for term in STATEMENT_STRUCTURE_TERMS[statement]
            )
            if has_table and continuation_score >= 1:
                selected.add(adjacent_number)
        classified[statement] = sorted(selected)[:MAX_READ_PAGES]
    return classified


def get_report_info(tool_context: ToolContext) -> dict:
    """Return report metadata, a heading outline, and classified primary-statement pages."""
    report, pages, error = report_from_state(tool_context)
    if error is not None:
        return error
    assert report is not None and pages is not None
    outline = []
    for page in pages:
        headings = HEADING_RE.findall(page.text)
        if headings and len(outline) < MAX_OUTLINE_ITEMS:
            outline.append({"page": page.display_number, "heading": headings[0][:160]})
    return {
        "status": "success",
        "report_id": report.get("report_id"),
        "exchange": report.get("exchange"),
        "ticker": report.get("ticker"),
        "fiscal_year": report.get("year"),
        "page_count": len(pages),
        "page_range": {
            "first": pages[0].display_number,
            "last": pages[-1].display_number,
        },
        "outline": outline,
        "statement_pages": _classified_statement_pages(pages),
    }


def inspect_primary_statements(tool_context: ToolContext) -> dict:
    """Return report metadata plus all classified primary-statement pages.

    Returns:
        Report metadata, primary-statement page text, structural source cells,
        and the statement group assigned to each page.
    """
    info = get_report_info(tool_context)
    if info.get("status") != "success":
        return info

    statement_pages = info["statement_pages"]
    inspected_pages: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    source_cell_count = 0
    for statement_group, page_numbers in statement_pages.items():
        result = read_report_pages(page_numbers, [], tool_context)
        if result.get("status") != "success":
            return result
        for page in result["pages"]:
            page_number = page["page"]
            if page_number in seen_pages:
                continue
            seen_pages.add(page_number)
            inspected_pages.append({**page, "statement_group": statement_group})
            source_cell_count += len(page.get("source_cells", []))

    return {
        **info,
        "pages": inspected_pages,
        "source_cell_count": source_cell_count,
        "missing_statement_groups": [
            statement_group
            for statement_group, page_numbers in statement_pages.items()
            if not page_numbers
        ],
    }


def prepare_multi_kpi_report(tool_context: ToolContext) -> dict:
    """Index primary statements once and return compact report-wide metadata.

    Returns:
        Compact report metadata, classified statement page numbers, and source
        cell counts. Full page payloads remain isolated from the coordinator.
    """
    prepared = tool_context.state.get(MULTI_KPI_PREPARED_STATE_KEY)
    source_cells = tool_context.state.get(MULTI_KPI_SOURCE_CELLS_STATE_KEY)
    if isinstance(prepared, dict) and isinstance(source_cells, dict):
        return {**prepared, "reused": True}

    result = inspect_primary_statements(tool_context)
    if result.get("status") != "success":
        return result
    report_state = tool_context.state.get(REPORT_STATE_KEY, {})
    report_pages = report_state.get("pages", []) if isinstance(report_state, dict) else []
    existing_cells = tool_context.state.get(MULTI_KPI_SOURCE_CELLS_STATE_KEY, {})
    all_source_cells = dict(existing_cells) if isinstance(existing_cells, dict) else {}
    state_pages = [
        Page(raw_index=int(page["raw_index"]), text=str(page["text"])) for page in report_pages
    ]
    document_unit = _document_unit_evidence(state_pages)
    tool_context.state[MULTI_KPI_DOCUMENT_UNIT_STATE_KEY] = document_unit
    for page in report_pages:
        page_number = int(page.get("display_number", 0))
        for cell in _inherit_document_unit(
            extract_source_cells(
                str(page.get("text", "")),
                page_number=page_number,
                fiscal_year=int(result.get("fiscal_year", 0)),
                allow_implicit_year=False,
            ),
            document_unit,
        ):
            all_source_cells[cell["source_id"]] = cell
    tool_context.state[MULTI_KPI_SOURCE_CELLS_STATE_KEY] = all_source_cells
    report_text = "\n".join(str(page.get("text", "")) for page in report_pages)[:40_000]
    sec_facts = (
        resolve_sec_kpis(
            str(result.get("ticker", "")),
            int(result.get("fiscal_year", 0)),
            report_text,
        )
        if tool_context.state.get(SEC_FACTS_ENABLED_STATE_KEY)
        else {"status": "disabled", "values": {}}
    )
    # Route short-term borrowings through its evidence specialist so the printed
    # bank-line/commercial-paper scope is checked independently of long-term-debt
    # current maturities.
    if isinstance(sec_facts.get("values"), dict):
        sec_facts["values"].pop("short_term_borrowings", None)
    tool_context.state[SEC_FACTS_STATE_KEY] = sec_facts
    compact = {
        "status": "success",
        "report_id": result.get("report_id"),
        "exchange": result.get("exchange"),
        "ticker": result.get("ticker"),
        "fiscal_year": result.get("fiscal_year"),
        "page_count": result.get("page_count"),
        "statement_pages": result.get("statement_pages", {}),
        "missing_statement_groups": result.get("missing_statement_groups", []),
        "source_cell_count": len(all_source_cells),
        "structured_fact_count": len(sec_facts.get("values", {})),
        "structured_kpis": [kpi for kpi in KPI_KEYS if kpi in sec_facts.get("values", {})],
        "pending_kpis": [kpi for kpi in KPI_KEYS if kpi not in sec_facts.get("values", {})],
        "structured_fact_status": sec_facts.get("status"),
        "reused": False,
    }
    tool_context.state[MULTI_KPI_PREPARED_STATE_KEY] = compact
    return compact


def _candidate_source_text(cell: dict[str, Any]) -> tuple[str, str]:
    row_label = str(cell.get("printed_row_label") or cell.get("row_label") or "")
    context = " ".join(
        str(cell.get(key) or "")
        for key in ("row_label", "section_label", "previous_label", "next_label")
    )

    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

    return normalize(row_label), normalize(context)


def _candidate_score(kpi: str, cell: dict[str, Any], aliases: tuple[str, ...]) -> int:
    row_label, context = _candidate_source_text(cell)
    context_tokens = set(context.split())
    prior_context = " ".join(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                str(cell.get(key) or "") for key in ("row_label", "section_label", "previous_label")
            ).casefold(),
        )
    )
    if kpi in {"eps_basic", "eps_diluted", "shares_outstanding"} and (
        "weighted average" in prior_context
    ):
        return 0
    if kpi == "depreciation_amortization" and "accumulated depreciation" in row_label:
        return 0
    if kpi == "revenue" and any(
        term in row_label for term in ("proceeds", "unearned revenue", "deferred revenue")
    ):
        return 0
    if kpi == "total_liabilities" and (
        "current liabilities" in row_label or ("liabilities" in row_label and "equity" in row_label)
    ):
        return 0
    best = 0
    for alias in aliases:
        normalized_alias = " ".join(re.findall(r"[a-z0-9]+", alias.casefold()))
        if not normalized_alias:
            continue
        meaningful_tokens = {token for token in normalized_alias.split() if len(token) >= 2}
        if row_label == normalized_alias:
            best = max(best, 100)
        elif normalized_alias in row_label:
            best = max(best, 80)
        elif normalized_alias in context:
            best = max(best, 55)
        elif len(meaningful_tokens) >= 2 and meaningful_tokens <= context_tokens:
            best = max(best, 30)
    return best


def find_kpi_source_candidates(kpi: str, tool_context: ToolContext) -> dict:
    """Return compact pre-indexed source cells matching one canonical KPI.

    Args:
        kpi: Canonical KPI key whose candidate rows should be retrieved.

    Returns:
        Report metadata, ranked source cells, and exact fallback search labels.
    """
    if kpi not in KPI_KEYS:
        return {
            "status": "error",
            "retryable": False,
            "error": f"unknown canonical KPI: {kpi}",
        }
    prepared = prepare_multi_kpi_report(tool_context)
    if prepared.get("status") != "success":
        return prepared
    raw_source_cells = tool_context.state.get(MULTI_KPI_SOURCE_CELLS_STATE_KEY, {})
    source_cells = raw_source_cells if isinstance(raw_source_cells, dict) else {}
    aliases = KPI_ALIASES[kpi]
    statement_group = KPI_STATEMENT_GROUP[kpi]
    preferred_pages = set(prepared.get("statement_pages", {}).get(statement_group, []))
    ranked = [
        (score, int(cell.get("page", 0)) in preferred_pages, cell)
        for cell in source_cells.values()
        if isinstance(cell, dict) and (score := _candidate_score(kpi, cell, aliases)) > 0
    ]
    ranked.sort(
        key=lambda item: (
            -item[0],
            -int(item[1]),
            int(item[2].get("page", 0)),
            str(item[2].get("source_id", "")),
        )
    )
    candidates = [
        {
            "source_id": cell.get("source_id"),
            "page": cell.get("page"),
            "row_label": cell.get("row_label"),
            "printed_row_label": cell.get("printed_row_label"),
            "row_role": cell.get("row_role"),
            "section_label": cell.get("section_label"),
            "previous_label": cell.get("previous_label"),
            "next_label": cell.get("next_label"),
            "year_label": cell.get("year_label"),
            "value_verbatim": cell.get("value_verbatim"),
            "status": cell.get("status"),
            "unit_text": cell.get("unit_text"),
            "unit_page": cell.get("unit_page"),
            "unit_scope": cell.get("unit_scope"),
            "preferred_statement_page": preferred,
            "match_score": score,
        }
        for score, preferred, cell in ranked[:16]
    ]
    return {
        "status": "success",
        "report_id": prepared.get("report_id"),
        "ticker": prepared.get("ticker"),
        "fiscal_year": prepared.get("fiscal_year"),
        "kpi": kpi,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "fallback_search_phrases": list(aliases),
        "primary_statement_pages": prepared.get("statement_pages", {}),
        "preferred_statement_group": statement_group,
    }


def _matched_phrases(text: str, phrases: list[str]) -> list[str]:
    lowered = text.casefold()
    return [phrase for phrase in phrases if phrase.casefold() in lowered]


def _logical_source_lines(text: str) -> list[str]:
    """Split compact HTML tables into row-sized lines without altering source tags."""
    structured = re.sub(r"</tr>\s*", "</tr>\n", text, flags=re.IGNORECASE)
    structured = re.sub(r"</table>\s*", "</table>\n", structured, flags=re.IGNORECASE)
    return structured.splitlines()


def _truncate_around_match(text: str, terms: list[str], limit: int) -> str:
    if len(text) <= limit:
        return text
    lowered = text.casefold()
    positions = [
        position
        for term in terms
        if term
        for position in [lowered.find(term.casefold())]
        if position >= 0
    ]
    if not positions:
        return text[:limit]
    match_position = min(positions)
    start = max(0, match_position - limit // 3)
    return text[start : start + limit]


def _search_snippet(text: str, query: str, phrases: list[str]) -> str:
    lines = _logical_source_lines(text)
    if not lines:
        return ""
    lowered_phrases = [phrase.casefold() for phrase in phrases]
    query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))

    def line_score(line: str) -> tuple[int, int]:
        lowered = line.casefold()
        phrase_score = sum(1 for phrase in lowered_phrases if phrase in lowered)
        token_score = sum(1 for token in query_tokens if token in lowered)
        return phrase_score, token_score

    best_index = max(range(len(lines)), key=lambda index: line_score(lines[index]))
    start = max(0, best_index - FOCUS_CONTEXT_LINES)
    end = min(len(lines), best_index + FOCUS_CONTEXT_LINES + 1)
    header = lines[: min(FOCUS_CONTEXT_LINES, start)]
    window = lines[start:end]
    snippet = "\n...\n".join(chunk for chunk in ("\n".join(header), "\n".join(window)) if chunk)
    return _truncate_around_match(
        snippet.strip(),
        [*phrases, query],
        MAX_SEARCH_SNIPPET_CHARS,
    )


def search_report(
    query: str,
    phrases: list[str],
    year: int | None,
    limit: int,
    tool_context: ToolContext,
) -> dict:
    """Locate report pages using ranked search plus exact phrase matching.

    Args:
        query: Natural-language search objective, KPI, or financial-statement name.
        phrases: Exact labels or headings to prioritize. Canonical KPI keys expand to aliases.
        year: Fiscal year to prioritize, or null when the search is year-independent.
        limit: Maximum number of candidate pages to return.

    Returns:
        Ranked page numbers with matched labels and focused source snippets.
    """
    report, pages, error = report_from_state(tool_context)
    if error is not None:
        return error
    assert report is not None and pages is not None
    if not isinstance(query, str):
        return _argument_error("query", "value must be a string")
    normalized_phrases, phrase_error = _validate_phrases(phrases, "phrases")
    if phrase_error is not None:
        return phrase_error
    normalized_year, year_error = _normalize_integer(year, "year", allow_none=True)
    if year_error is not None:
        return year_error
    normalized_limit, limit_error = _normalize_integer(limit, "limit")
    if limit_error is not None:
        return limit_error
    assert normalized_phrases is not None and normalized_limit is not None
    cleaned_query = query.strip()
    expanded_phrases = expand_search_phrases(normalized_phrases)
    if not cleaned_query and not expanded_phrases:
        return {
            "status": "error",
            "error": "query or phrases must contain at least one search value",
        }
    if normalized_year is not None and not 1900 <= normalized_year <= 2100:
        return {"status": "error", "error": "year must be between 1900 and 2100 or null"}
    bounded_limit = min(max(1, normalized_limit), MAX_SEARCH_RESULTS)
    hits = rank_pages(pages, cleaned_query, normalized_phrases, normalized_year)[:bounded_limit]
    return {
        "status": "success",
        "report_id": report.get("report_id"),
        "results": [
            {
                "page": hit.page.display_number,
                "matched_phrases": _matched_phrases(hit.page.text, expanded_phrases),
                "snippet": _search_snippet(hit.page.text, cleaned_query, expanded_phrases),
            }
            for hit in hits
        ],
    }


def search_kpi_report(
    kpi: str,
    query: str,
    phrases: list[str],
    year: int | None,
    limit: int,
    tool_context: ToolContext,
) -> dict:
    """Search with the canonical KPI's expected primary-statement pages preferred."""
    if kpi not in KPI_KEYS:
        return {
            "status": "error",
            "retryable": False,
            "error": f"unknown canonical KPI: {kpi}",
        }
    report, pages, error = report_from_state(tool_context)
    if error is not None:
        return error
    assert report is not None and pages is not None
    if not isinstance(query, str):
        return _argument_error("query", "value must be a string")
    normalized_phrases, phrase_error = _validate_phrases(phrases, "phrases")
    if phrase_error is not None:
        return phrase_error
    normalized_year, year_error = _normalize_integer(year, "year", allow_none=True)
    if year_error is not None:
        return year_error
    normalized_limit, limit_error = _normalize_integer(limit, "limit")
    if limit_error is not None:
        return limit_error
    assert normalized_phrases is not None and normalized_limit is not None
    cleaned_query = query.strip()
    expanded_phrases = expand_search_phrases(normalized_phrases)
    if not cleaned_query and not expanded_phrases:
        return {
            "status": "error",
            "error": "query or phrases must contain at least one search value",
        }
    if normalized_year is not None and not 1900 <= normalized_year <= 2100:
        return {"status": "error", "error": "year must be between 1900 and 2100 or null"}
    statement_pages = _classified_statement_pages(pages)[KPI_STATEMENT_GROUP[kpi]]
    preferred_pages = frozenset(statement_pages)
    bounded_limit = min(max(1, normalized_limit), MAX_SEARCH_RESULTS)
    hits = rank_pages(
        pages,
        cleaned_query,
        normalized_phrases,
        normalized_year,
        preferred_page_numbers=preferred_pages,
    )[:bounded_limit]
    return {
        "status": "success",
        "report_id": report.get("report_id"),
        "preferred_statement_group": KPI_STATEMENT_GROUP[kpi],
        "preferred_pages": statement_pages,
        "results": [
            {
                "page": hit.page.display_number,
                "matched_phrases": _matched_phrases(hit.page.text, expanded_phrases),
                "snippet": _search_snippet(hit.page.text, cleaned_query, expanded_phrases),
            }
            for hit in hits
        ],
    }


def _focused_page_text(text: str, phrases: list[str]) -> tuple[str, list[str], bool]:
    lines = _logical_source_lines(text)
    matching_indexes = [
        index
        for index, line in enumerate(lines)
        if any(phrase.casefold() in line.casefold() for phrase in phrases)
    ]
    if not matching_indexes:
        return "", [], False

    selected_indexes: set[int] = set()
    selected_indexes.update(range(min(FOCUS_CONTEXT_LINES, len(lines))))
    for index in matching_indexes:
        selected_indexes.update(
            range(
                max(0, index - FOCUS_CONTEXT_LINES),
                min(len(lines), index + FOCUS_CONTEXT_LINES + 1),
            )
        )
    chunks: list[str] = []
    previous_index: int | None = None
    for index in sorted(selected_indexes):
        if previous_index is not None and index > previous_index + 1:
            chunks.append("...")
        chunks.append(lines[index])
        previous_index = index
    focused = "\n".join(chunks).strip()
    truncated = len(focused) > MAX_FOCUSED_TEXT_CHARS
    return (
        _truncate_around_match(focused, phrases, MAX_FOCUSED_TEXT_CHARS),
        _matched_phrases(text, phrases),
        truncated,
    )


def read_report_pages(
    page_numbers: list[int], focus_phrases: list[str], tool_context: ToolContext
) -> dict:
    """Read up to three exact pages, optionally restricted to focused text windows.

    Args:
        page_numbers: Displayed page numbers selected from search results.
        focus_phrases: Labels to read with surrounding lines, or an empty list for full pages.

    Returns:
        Immutable page text or focused windows, plus unavailable requested page numbers.
    """
    report, pages, error = report_from_state(tool_context)
    if error is not None:
        return error
    assert report is not None and pages is not None
    if not isinstance(page_numbers, list):
        return _argument_error("page_numbers", "value must be a list of integers")
    normalized_numbers: list[int] = []
    for index, number in enumerate(page_numbers):
        normalized, number_error = _normalize_integer(number, f"page_numbers.{index}")
        if number_error is not None:
            return number_error
        assert normalized is not None
        normalized_numbers.append(normalized)
    normalized_phrases, phrase_error = _validate_phrases(focus_phrases, "focus_phrases")
    if phrase_error is not None:
        return phrase_error
    assert normalized_phrases is not None
    unique_numbers = list(dict.fromkeys(normalized_numbers))[:MAX_READ_PAGES]
    by_number = {page.display_number: page for page in pages}
    expanded_phrases = expand_search_phrases(normalized_phrases)
    found = []
    source_cells: list[dict[str, Any]] = []
    fiscal_year = int(report.get("year", 0))
    primary_page_numbers = {
        page_number
        for page_numbers in _classified_statement_pages(pages).values()
        for page_number in page_numbers
    }
    for number in unique_numbers:
        if number not in by_number:
            continue
        page_text = by_number[number].text
        if expanded_phrases:
            text, matched, truncated = _focused_page_text(page_text, expanded_phrases)
            page_result = {"page": number, "text": text, "matched_phrases": matched}
        else:
            truncated = len(page_text) > MAX_PAGE_TEXT_CHARS
            page_result = {"page": number, "text": page_text[:MAX_PAGE_TEXT_CHARS]}
        if truncated:
            page_result["truncated"] = True
        raw_document_unit = tool_context.state.get(MULTI_KPI_DOCUMENT_UNIT_STATE_KEY)
        document_unit = raw_document_unit if isinstance(raw_document_unit, dict) else None
        page_source_cells = _inherit_document_unit(
            extract_source_cells(
                page_text,
                page_number=number,
                fiscal_year=fiscal_year,
                allow_implicit_year=number in primary_page_numbers,
            ),
            document_unit,
        )
        if page_source_cells:
            page_result["source_cells"] = page_source_cells
            source_cells.extend(page_source_cells)
        found.append(page_result)
    existing_source_cells = tool_context.state.get(MULTI_KPI_SOURCE_CELLS_STATE_KEY, {})
    accumulated_source_cells = (
        dict(existing_source_cells) if isinstance(existing_source_cells, dict) else {}
    )
    accumulated_source_cells.update({cell["source_id"]: cell for cell in source_cells})
    tool_context.state[MULTI_KPI_SOURCE_CELLS_STATE_KEY] = accumulated_source_cells
    missing = [number for number in unique_numbers if number not in by_number]
    return {
        "status": "success",
        "report_id": report.get("report_id"),
        "pages": found,
        "source_cell_count": len(source_cells),
        "missing_pages": missing,
    }
