"""State-backed report inspection tools for FinGround agents."""

from __future__ import annotations

import re
from typing import Any

from google.adk.tools import ToolContext

from finground.documents import Page, Report, load_report_pages
from finground.retrieval import expand_search_phrases, rank_pages

MAX_SEARCH_RESULTS = 8
MAX_READ_PAGES = 3
MAX_OUTLINE_ITEMS = 80
MAX_SEARCH_SNIPPET_CHARS = 1_200
MAX_PAGE_TEXT_CHARS = 24_000
MAX_FOCUSED_TEXT_CHARS = 12_000
FOCUS_CONTEXT_LINES = 4
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
REPORT_STATE_KEY = "report"


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


def get_report_info(tool_context: ToolContext) -> dict:
    """Return report metadata, page range, and a bounded heading outline."""
    report, pages, error = report_from_state(tool_context)
    if error is not None:
        return error
    assert report is not None and pages is not None
    outline = []
    for page in pages:
        match = HEADING_RE.search(page.text)
        if match:
            outline.append({"page": page.display_number, "heading": match.group(1)[:160]})
            if len(outline) >= MAX_OUTLINE_ITEMS:
                break
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
    }


def _matched_phrases(text: str, phrases: list[str]) -> list[str]:
    lowered = text.casefold()
    return [phrase for phrase in phrases if phrase.casefold() in lowered]


def _search_snippet(text: str, query: str, phrases: list[str]) -> str:
    lines = text.splitlines()
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
    return "\n".join(lines[start:end]).strip()[:MAX_SEARCH_SNIPPET_CHARS]


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


def _focused_page_text(text: str, phrases: list[str]) -> tuple[str, list[str], bool]:
    lines = text.splitlines()
    matching_indexes = [
        index
        for index, line in enumerate(lines)
        if any(phrase.casefold() in line.casefold() for phrase in phrases)
    ]
    if not matching_indexes:
        return "", [], False

    selected_indexes: set[int] = set()
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
        focused[:MAX_FOCUSED_TEXT_CHARS],
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
        found.append(page_result)
    missing = [number for number in unique_numbers if number not in by_number]
    return {
        "status": "success",
        "report_id": report.get("report_id"),
        "pages": found,
        "missing_pages": missing,
    }
