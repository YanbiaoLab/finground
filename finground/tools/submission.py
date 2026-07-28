"""Validated submission tools that persist benchmark results in session state."""

from __future__ import annotations

import math
import re
from collections import Counter
from html import unescape
from typing import Any

from google.adk.tools import ToolContext
from pydantic import TypeAdapter, ValidationError

from finground.kpis import KPI_KEYS, POSITIVE_MAGNITUDE_KPIS, POSITIVE_OUTFLOW_KPIS
from finground.models import (
    MultiKpiEvidence,
    MultiKpiEvidenceCandidate,
    MultiKpiNote,
    MultiKpiRecordView,
    MultiKpiWorkRecord,
    NeedleAnswer,
    ReportExtraction,
    UnitScale,
)
from finground.normalize import (
    NUMBER_TOKEN_RE,
    SCALE_MULTIPLIERS,
    detect_scale,
    normalize_needle_answer,
    normalize_value,
    parse_financial_number,
    validate_needle_evidence,
)

from .report import report_from_state
from .structured import JsonSchemaFunctionTool

ISO_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

NEEDLE_KPI_STATE_KEY = "needle_kpi"
NEEDLE_RESULT_STATE_KEY = "needle_result"
MULTI_KPI_WORK_RECORD_STATE_KEY = "multi_kpi_work_record"
MULTI_KPI_RESULT_STATE_KEY = "multi_kpi_result"
MULTI_KPI_AUDIT_STATE_KEY = "multi_kpi_audit"
MULTI_KPI_ALLOW_PARTIAL_STATE_KEY = "temp:multi_kpi_allow_partial_submission"

_EVIDENCE_CANDIDATES = TypeAdapter(list[MultiKpiEvidenceCandidate])
_MULTI_KPI_NOTES = TypeAdapter(list[MultiKpiNote])
_EXPLICIT_ZERO_MARKERS = {"-", "−", "–", "—", "nil"}
MAX_MULTI_KPI_RECORD_ROWS = 8

_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(
    r"<t[dh]\b([^>]*)>(.*?)</t[dh]>",
    re.IGNORECASE | re.DOTALL,
)


def _multi_kpi_submission_errors(
    extraction: ReportExtraction,
    report: dict[str, Any],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    expected_ticker = str(report.get("ticker", ""))
    if extraction.ticker != expected_ticker:
        errors.append(
            {
                "field": "ticker",
                "message": f"ticker must match report ticker {expected_ticker!r}",
            }
        )
    if extraction.reporting_currency is not None and not ISO_CURRENCY_RE.fullmatch(
        extraction.reporting_currency
    ):
        errors.append(
            {
                "field": "reporting_currency",
                "message": ("reporting_currency must be a three-letter uppercase ISO code or null"),
            }
        )
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(extraction.kpis):
        key = (item.kpi, item.fiscal_year)
        if key in seen:
            errors.append(
                {
                    "field": f"kpis.{index}",
                    "message": f"duplicate KPI/year pair: {item.kpi}/{item.fiscal_year}",
                }
            )
        seen.add(key)
        if item.value is None:
            errors.append(
                {
                    "field": f"kpis.{index}.value",
                    "message": "omit unavailable KPIs instead of submitting a null value",
                }
            )
        elif not math.isfinite(item.value):
            errors.append(
                {
                    "field": f"kpis.{index}.value",
                    "message": "value must be a finite number",
                }
            )
    return errors


def _validation_error_response(
    error: ValidationError, message: str, *, prefix: str | None = None
) -> dict:
    return {
        "status": "error",
        "retryable": True,
        "error": message,
        "validation_errors": [
            {
                "field": ".".join(
                    str(part)
                    for part in ((prefix,) if prefix is not None else ()) + item["loc"]
                ),
                "message": item["msg"],
            }
            for item in error.errors()
        ],
    }


def _normalized_source_text(text: str) -> str:
    visible_text = re.sub(r"<[^>]+>", " ", unescape(text))
    return " ".join(visible_text.casefold().split())


def _source_rows(page_text: str) -> list[str]:
    html_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", page_text, flags=re.IGNORECASE | re.DOTALL)
    return html_rows or page_text.splitlines() or [page_text]


def _line_contains_number(page_text: str, line_label: str, value_verbatim: str) -> bool:
    try:
        expected = parse_financial_number(value_verbatim)
    except ValueError:
        return False
    normalized_label = _normalized_source_text(line_label)
    for line in _source_rows(page_text):
        if normalized_label not in _normalized_source_text(line):
            continue
        for token in NUMBER_TOKEN_RE.findall(line):
            try:
                observed = parse_financial_number(token)
            except ValueError:
                continue
            if math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                return True
    return False


def _line_contains_zero_marker(page_text: str, line_label: str, marker: str) -> bool:
    normalized_label = _normalized_source_text(line_label)
    normalized_marker = marker.strip().casefold()
    return any(
        normalized_label in _normalized_source_text(line)
        and normalized_marker in unescape(line).casefold()
        for line in _source_rows(page_text)
    )


def _html_span(attributes: str, name: str) -> int:
    match = re.search(
        rf"\b{name}\s*=\s*[\"']?(\d+)",
        attributes,
        flags=re.IGNORECASE,
    )
    return max(1, int(match.group(1))) if match else 1


def _expand_table_rows(
    rows: list[list[tuple[str, int, int]]],
) -> list[list[str]]:
    grid: list[list[str]] = []
    active_rowspans: dict[int, tuple[int, str]] = {}
    for raw_row in rows:
        values = {column: item[1] for column, item in active_rowspans.items()}
        next_rowspans = {
            column: (item[0] - 1, item[1])
            for column, item in active_rowspans.items()
            if item[0] > 1
        }
        column = 0
        for text, colspan, rowspan in raw_row:
            while any(column + offset in values for offset in range(colspan)):
                column += 1
            for offset in range(colspan):
                target = column + offset
                values[target] = text
                if rowspan > 1:
                    next_rowspans[target] = (rowspan - 1, text)
            column += colspan
        active_rowspans = next_rowspans
        if values:
            grid.append([values.get(index, "") for index in range(max(values) + 1)])
    return grid


def _html_table_grids(page_text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    for table_match in _HTML_TABLE_RE.finditer(page_text):
        rows: list[list[tuple[str, int, int]]] = []
        for row_match in _HTML_ROW_RE.finditer(table_match.group()):
            cells = [
                (
                    _normalized_source_text(cell_html),
                    _html_span(attributes, "colspan"),
                    _html_span(attributes, "rowspan"),
                )
                for attributes, cell_html in _HTML_CELL_RE.findall(row_match.group(1))
            ]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(_expand_table_rows(rows))
    return tables


def _is_markdown_separator_row(row: list[str]) -> bool:
    return bool(row) and all(
        not cell or re.fullmatch(r":?-{3,}:?", cell.strip()) is not None for cell in row
    )


def _markdown_table_grids(page_text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped[1:-1].split("|")]
            if not _is_markdown_separator_row(cells):
                current.append(cells)
        elif current:
            if len(current) > 1:
                tables.append(current)
            current = []
    if len(current) > 1:
        tables.append(current)
    return tables


def _cell_contains_number(cell: str, value_verbatim: str) -> bool:
    candidate_tokens = NUMBER_TOKEN_RE.findall(value_verbatim)
    if len(candidate_tokens) != 1:
        return False
    try:
        expected = parse_financial_number(candidate_tokens[0])
    except ValueError:
        return False
    for token in NUMBER_TOKEN_RE.findall(cell):
        try:
            observed = parse_financial_number(token)
        except ValueError:
            continue
        if math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            return True
    return False


def _cell_contains_zero_marker(cell: str, marker: str) -> bool:
    normalized_cell = re.sub(r"[$€£¥\s]", "", unescape(cell)).casefold()
    return normalized_cell == marker.strip().casefold()


def _value_matches_year_column(
    page_text: str,
    candidate: MultiKpiEvidenceCandidate,
) -> bool | None:
    """Return whether table structure aligns the source token to the cited year.

    ``None`` means the page did not expose a parseable row/year grid, so callers
    can retain the existing conservative row-level fallback for OCR prose.
    """
    assert candidate.line_label is not None
    assert candidate.value_verbatim is not None
    normalized_label = _normalized_source_text(candidate.line_label)
    year = str(candidate.fiscal_year)
    attempted_alignment = False
    for table in [*_html_table_grids(page_text), *_markdown_table_grids(page_text)]:
        for row_index, row in enumerate(table):
            if not any(normalized_label in _normalized_source_text(cell) for cell in row):
                continue
            year_columns = {
                column
                for header in table[: min(row_index, 4)]
                for column, cell in enumerate(header)
                if re.search(rf"(?<!\d){re.escape(year)}(?!\d)", cell)
            }
            if not year_columns:
                continue
            attempted_alignment = True
            for column in year_columns:
                if column >= len(row):
                    continue
                if candidate.status == "explicit_zero":
                    if _cell_contains_zero_marker(row[column], candidate.value_verbatim):
                        return True
                elif _cell_contains_number(row[column], candidate.value_verbatim):
                    return True
    return False if attempted_alignment else None


def _resolve_multi_kpi_scale(
    candidate: MultiKpiEvidenceCandidate,
) -> tuple[UnitScale | None, str | None]:
    if candidate.unit_text is None and candidate.unit_scale not in {None, "unknown"}:
        return candidate.unit_scale, "agent"
    detected = detect_scale(candidate.unit_text or "", candidate.kpi)
    if detected != "unknown":
        return detected, "unit_text" if candidate.unit_text else "kpi"
    if candidate.unit_scale not in {None, "unknown"}:
        return candidate.unit_scale, "agent"
    return None, None


def _normalize_multi_kpi_candidates(
    raw_candidates: list[object],
    pages: list[Any],
    report_year: int,
) -> tuple[list[MultiKpiEvidence], list[dict[str, str]], list[dict[str, Any]]]:
    try:
        candidates = _EVIDENCE_CANDIDATES.validate_python(raw_candidates)
    except ValidationError as error:
        response = _validation_error_response(
            error,
            "KPI evidence does not match the work-record schema",
            prefix="kpis",
        )
        return [], response["validation_errors"], []

    errors: list[dict[str, str]] = []
    corrections: list[dict[str, Any]] = []
    normalized: list[MultiKpiEvidence] = []
    page_by_number = {page.display_number: page.text for page in pages}
    seen: set[tuple[str, int]] = set()
    for index, candidate in enumerate(candidates):
        key = (candidate.kpi, candidate.fiscal_year)
        if key in seen:
            errors.append(
                {
                    "field": f"kpis.{index}",
                    "message": (
                        f"duplicate KPI/year coverage row: "
                        f"{candidate.kpi}/{candidate.fiscal_year}"
                    ),
                }
            )
            continue
        seen.add(key)

        if candidate.fiscal_year != report_year:
            errors.append(
                {
                    "field": f"kpis.{index}.fiscal_year",
                    "message": f"fiscal_year must match report fiscal year {report_year}",
                }
            )
            continue

        if candidate.status not in {"found", "explicit_zero"}:
            normalized.append(MultiKpiEvidence(**candidate.model_dump()))
            continue

        assert candidate.page is not None
        assert candidate.line_label is not None
        assert candidate.value_verbatim is not None
        page_text = page_by_number.get(candidate.page)
        if page_text is None:
            errors.append(
                {
                    "field": f"kpis.{index}.page",
                    "message": "page does not exist in the report",
                }
            )
            continue
        if candidate.year_label and candidate.year_label.casefold() not in page_text.casefold():
            errors.append(
                {
                    "field": f"kpis.{index}.year_label",
                    "message": "year_label was not found on the cited page",
                }
            )

        if candidate.unit_text is not None:
            unit_page_text = page_by_number.get(candidate.unit_page)
            if unit_page_text is None:
                errors.append(
                    {
                        "field": f"kpis.{index}.unit_page",
                        "message": "unit_page does not exist in the report",
                    }
                )
            elif _normalized_source_text(candidate.unit_text) not in _normalized_source_text(
                unit_page_text
            ):
                errors.append(
                    {
                        "field": f"kpis.{index}.unit_text",
                        "message": "unit_text was not found on unit_page",
                    }
                )

        scale, scale_source = _resolve_multi_kpi_scale(candidate)
        if scale is None:
            errors.append(
                {
                    "field": f"kpis.{index}.unit_scale",
                    "message": "scale could not be resolved from unit_text or unit_scale",
                }
            )
            continue
        if candidate.unit_scale not in {None, "unknown", scale}:
            corrections.append(
                {
                    "kpi": candidate.kpi,
                    "fiscal_year": candidate.fiscal_year,
                    "field": "unit_scale",
                    "submitted": candidate.unit_scale,
                    "used": scale,
                    "source": scale_source,
                }
            )

        if candidate.status == "explicit_zero":
            marker = candidate.value_verbatim.strip().casefold()
            if marker not in _EXPLICIT_ZERO_MARKERS:
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": (
                            "explicit_zero requires an exact printed dash or nil marker; "
                            "use found for a printed numeric zero"
                        ),
                    }
                )
                continue
            aligned = _value_matches_year_column(page_text, candidate)
            if aligned is False:
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "zero marker was not found in the cited fiscal-year cell",
                    }
                )
                continue
            if aligned is None and not _line_contains_zero_marker(
                page_text, candidate.line_label, marker
            ):
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "zero marker was not found on the cited labelled row",
                    }
                )
                continue
            value = 0.0
            parsed_number = 0.0
            sign_rule = "explicit_zero"
        else:
            tokens = NUMBER_TOKEN_RE.findall(candidate.value_verbatim)
            if len(tokens) != 1:
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "value_verbatim must contain exactly one printed number",
                    }
                )
                continue
            aligned = _value_matches_year_column(page_text, candidate)
            if aligned is False:
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "number was not found in the cited fiscal-year cell",
                    }
                )
                continue
            if aligned is None and not _line_contains_number(
                page_text,
                candidate.line_label,
                candidate.value_verbatim,
            ):
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "number was not found on the cited labelled row",
                    }
                )
                continue
            try:
                parsed_number = parse_financial_number(tokens[0])
                value = normalize_value(tokens[0], scale, candidate.kpi)
            except ValueError:
                errors.append(
                    {
                        "field": f"kpis.{index}.value_verbatim",
                        "message": "value_verbatim is not a valid financial number",
                    }
                )
                continue
            if candidate.kpi in POSITIVE_MAGNITUDE_KPIS:
                value = abs(value)
                sign_rule = (
                    "positive_outflow"
                    if candidate.kpi in POSITIVE_OUTFLOW_KPIS
                    else "positive_magnitude"
                )
            else:
                sign_rule = "as_reported"

        normalized.append(
            MultiKpiEvidence(
                **candidate.model_dump(exclude={"unit_scale"}),
                unit_scale=scale,
                value=value,
                normalization={
                    "parsed_number": parsed_number,
                    "multiplier": SCALE_MULTIPLIERS[scale],
                    "sign_rule": sign_rule,
                    "formula": (
                        "explicit printed zero"
                        if sign_rule == "explicit_zero"
                        else f"abs({parsed_number} * {SCALE_MULTIPLIERS[scale]})"
                        if sign_rule in {"positive_outflow", "positive_magnitude"}
                        else f"{parsed_number} * {SCALE_MULTIPLIERS[scale]}"
                    ),
                },
            )
        )
    return normalized, errors, corrections


def _work_record_counts(work_record: MultiKpiWorkRecord) -> tuple[int, Counter[str]]:
    status_counts = Counter(item.status for item in work_record.kpis)
    extracted_count = status_counts["found"] + status_counts["explicit_zero"]
    return extracted_count, status_counts


def _pending_multi_kpis(work_record: MultiKpiWorkRecord, report_year: int) -> list[str]:
    covered = {
        item.kpi for item in work_record.kpis if item.fiscal_year == report_year
    }
    return [kpi for kpi in KPI_KEYS if kpi not in covered]


def _load_multi_kpi_work_record(tool_context: ToolContext) -> MultiKpiWorkRecord | None:
    raw_record = tool_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY)
    if raw_record is None:
        return None
    try:
        return MultiKpiWorkRecord.model_validate(raw_record)
    except ValidationError as error:
        raise RuntimeError("multi-KPI work record in session state is invalid") from error


def record_multi_kpi_progress(
    reporting_currency: str | None,
    units_note: str | None,
    kpis: list[MultiKpiEvidenceCandidate],
    notes: list[MultiKpiNote],
    tool_context: ToolContext,
) -> dict:
    """Validate and record grounded Multi-KPI evidence, coverage, and notes.

    Args:
        reporting_currency: Three-letter uppercase reporting currency, or null if still unknown.
        units_note: Short note describing observed statement scale, or null.
        kpis: Evidence/coverage rows. A found row uses kpi, fiscal_year, status="found",
            exact value_verbatim, page, statement, line_label, year_label, scope, and either exact
            unit_text plus unit_page or unit_scale. Do not pass a calculated value: this tool
            parses, scales, and signs it. Use status="explicit_zero" only for a printed dash/nil
            on the matching row and year. Use status="absent" or "ambiguous" without a value for
            unresolved coverage; those rows are retained for work tracking and omitted from the
            final LEDGER output.
        notes: Durable observations with category, text, and relevant report page numbers. Use
            categories evidence, unit, scope, decision, todo, or warning.

    Returns:
        Compact record statistics, or retryable field-level validation errors.
    """
    report, pages, state_error = report_from_state(tool_context)
    if state_error is not None:
        return state_error
    assert report is not None and pages is not None
    try:
        metadata = ReportExtraction.model_validate(
            {
                "ticker": str(report.get("ticker", "")),
                "reporting_currency": reporting_currency,
                "units_note": units_note,
                "kpis": [],
            }
        )
    except ValidationError as error:
        return _validation_error_response(
            error,
            "progress does not match the Multi-KPI work-record schema",
        )
    try:
        incoming_notes = _MULTI_KPI_NOTES.validate_python(notes)
    except ValidationError as error:
        return _validation_error_response(
            error,
            "notes do not match the Multi-KPI work-record schema",
            prefix="notes",
        )
    incoming_kpis, evidence_errors, corrections = _normalize_multi_kpi_candidates(
        kpis,
        pages,
        int(report.get("year", 0)),
    )
    errors = _multi_kpi_submission_errors(metadata, report)
    valid_pages = {page.display_number for page in pages}
    for note_index, note in enumerate(incoming_notes):
        missing_pages = [page for page in note.pages if page not in valid_pages]
        if missing_pages:
            errors.append(
                {
                    "field": f"notes.{note_index}.pages",
                    "message": f"pages do not exist in the report: {missing_pages}",
                }
            )
    if errors:
        return {
            "status": "error",
            "retryable": True,
            "error": "progress record failed basic data checks",
            "validation_errors": [*errors, *evidence_errors],
        }
    if (
        not incoming_kpis
        and not incoming_notes
        and metadata.reporting_currency is None
        and metadata.units_note is None
    ):
        validation_errors = evidence_errors or [
            {
                "field": "record",
                "message": "record at least one KPI, note, currency, or units observation",
            }
        ]
        return {
            "status": "error",
            "retryable": True,
            "error": "progress record failed basic data checks",
            "validation_errors": validation_errors,
        }

    work_record = _load_multi_kpi_work_record(tool_context)
    indexed = (
        {(item.kpi, item.fiscal_year): item for item in work_record.kpis}
        if work_record is not None
        else {}
    )
    added = 0
    updated = 0
    for item in incoming_kpis:
        key = (item.kpi, item.fiscal_year)
        if key in indexed:
            updated += 1
        else:
            added += 1
        indexed[key] = item

    existing_notes = work_record.notes if work_record is not None else []
    note_keys = {(note.category, note.text, tuple(note.pages)) for note in existing_notes}
    added_notes = []
    for note in incoming_notes:
        key = (note.category, note.text, tuple(note.pages))
        if key not in note_keys:
            added_notes.append(note)
            note_keys.add(key)

    combined = MultiKpiWorkRecord(
        ticker=metadata.ticker,
        reporting_currency=(
            metadata.reporting_currency
            if metadata.reporting_currency is not None
            else work_record.reporting_currency
            if work_record is not None
            else None
        ),
        units_note=(
            metadata.units_note
            if metadata.units_note is not None
            else work_record.units_note
            if work_record is not None
            else None
        ),
        kpis=sorted(indexed.values(), key=lambda item: (item.fiscal_year, item.kpi)),
        notes=[*existing_notes, *added_notes],
    )
    result = combined.model_dump()
    tool_context.state[MULTI_KPI_WORK_RECORD_STATE_KEY] = result
    extracted_count, status_counts = _work_record_counts(combined)
    pending_kpis = _pending_multi_kpis(combined, int(report.get("year", 0)))
    response = {
        "status": "partial_success" if evidence_errors else "success",
        "kpi_count": extracted_count,
        "coverage_count": len(combined.kpis),
        "pending_count": len(pending_kpis),
        "pending_kpis": pending_kpis,
        "status_counts": dict(status_counts),
        "note_count": len(combined.notes),
        "added_kpi_count": added,
        "updated_kpi_count": updated,
        "added_note_count": len(added_notes),
        "normalization_corrections": corrections,
    }
    if evidence_errors:
        response.update(
            {
                "retryable": True,
                "error": "valid progress was saved; correct the rejected KPI rows",
                "validation_errors": evidence_errors,
            }
        )
    return response


def query_multi_kpi_progress(view: MultiKpiRecordView, tool_context: ToolContext) -> dict:
    """Query recorded Multi-KPI progress as all data, KPI rows only, or notes only.

    Args:
        view: Select all, kpis, or notes to control how much recorded context is returned.

    Returns:
        The selected validated work-record view and compact counts.
    """
    report, _pages, state_error = report_from_state(tool_context)
    if state_error is not None:
        return state_error
    assert report is not None
    work_record = _load_multi_kpi_work_record(tool_context)
    if work_record is None:
        work_record = MultiKpiWorkRecord(ticker=str(report.get("ticker", "")))
    record = work_record.model_dump()
    if view == "kpis":
        record = {
            "ticker": work_record.ticker,
            "reporting_currency": work_record.reporting_currency,
            "units_note": work_record.units_note,
            "kpis": [
                {
                    "kpi": item.kpi,
                    "fiscal_year": item.fiscal_year,
                    "status": item.status,
                    "value": item.value,
                    "unit_scale": item.unit_scale,
                    "page": item.page,
                    "line_label": item.line_label,
                }
                for item in work_record.kpis
            ],
        }
    elif view == "notes":
        record = {"ticker": work_record.ticker, "notes": record["notes"]}
    extracted_count, status_counts = _work_record_counts(work_record)
    pending_kpis = _pending_multi_kpis(work_record, int(report.get("year", 0)))
    return {
        "status": "success",
        "view": view,
        "kpi_count": extracted_count,
        "coverage_count": len(work_record.kpis),
        "pending_count": len(pending_kpis),
        "pending_kpis": pending_kpis,
        "status_counts": dict(status_counts),
        "note_count": len(work_record.notes),
        "record": record,
    }


def submit_needle_extraction(
    found: bool,
    value: float | None,
    value_verbatim: str | None,
    unit_scale: UnitScale | None,
    page: int | None,
    tool_context: ToolContext,
) -> dict:
    """Validate and store the final LEDGER-compatible needle answer.

    Args:
        found: Whether the exact KPI value was found for the requested fiscal year.
        value: Normalized value in raw single units, or null when not found.
        value_verbatim: Exact numeric token printed in the report, or null when not found.
        unit_scale: Applied scale, or null when not found.
        page: One-indexed cited report page, or null when not found.

    Returns:
        Success with the validated NeedleAnswer, or retryable field-level errors.
    """
    try:
        answer = NeedleAnswer.model_validate(
            {
                "found": found,
                "value": value,
                "value_verbatim": value_verbatim,
                "unit_scale": unit_scale,
                "page": page,
            }
        )
    except ValidationError as error:
        return _validation_error_response(
            error,
            "answer does not match the LEDGER NeedleAnswer schema",
        )

    errors: list[dict[str, str]] = []
    if answer.found:
        if answer.unit_scale is None:
            errors.append({"field": "unit_scale", "message": "found answers require a scale"})
        if answer.page is None:
            errors.append({"field": "page", "message": "found answers require a cited page"})
        if answer.value is not None and not math.isfinite(answer.value):
            errors.append({"field": "value", "message": "value must be a finite number"})
    else:
        errors.extend(
            {
                "field": field,
                "message": "not-found answers require this field to be null",
            }
            for field in ("value", "value_verbatim", "unit_scale", "page")
            if getattr(answer, field) is not None
        )

    report, pages, state_error = report_from_state(tool_context)
    if state_error is not None:
        return state_error
    assert report is not None and pages is not None
    kpi = tool_context.state.get(NEEDLE_KPI_STATE_KEY)
    if not isinstance(kpi, str) or not kpi:
        return {
            "status": "error",
            "retryable": False,
            "error": "needle_kpi is missing from state",
        }
    valid_pages = {page_item.display_number for page_item in pages}
    if answer.found and answer.page not in valid_pages:
        errors.append(
            {
                "field": "page",
                "message": "page must exist in the report stored in session state",
            }
        )
    if errors:
        return {
            "status": "error",
            "retryable": True,
            "error": "answer failed basic data checks",
            "validation_errors": errors,
        }

    normalized, normalization_trace = normalize_needle_answer(answer, kpi)
    if normalized.found and normalization_trace["status"] == "unverified":
        return {
            "status": "error",
            "retryable": True,
            "error": "answer normalization could not be verified",
            "validation_errors": [
                {
                    "field": "value",
                    "message": str(normalization_trace.get("reason", "normalization failed")),
                }
            ],
        }
    if normalization_trace["status"] == "corrected":
        return {
            "status": "error",
            "retryable": True,
            "error": "value does not match value_verbatim and unit_scale",
            "validation_errors": [
                {
                    "field": "value",
                    "message": (
                        f"expected normalized value {normalization_trace['computed_value']}"
                    ),
                }
            ],
        }

    cited_page_text = next(
        (page_item.text for page_item in pages if page_item.display_number == normalized.page),
        None,
    )
    validated, evidence_trace = validate_needle_evidence(normalized, cited_page_text)
    if normalized.found and not validated.found:
        return {
            "status": "error",
            "retryable": True,
            "error": "answer evidence validation failed",
            "validation_errors": [
                {
                    "field": "value_verbatim",
                    "message": str(evidence_trace.get("reason", "evidence validation failed")),
                }
            ],
        }

    result = validated.model_dump()
    tool_context.state[NEEDLE_RESULT_STATE_KEY] = result
    return {"status": "success", "result": result}


def submit_multi_kpi_extraction(
    ticker: str,
    reporting_currency: str | None,
    units_note: str | None,
    kpis: list[MultiKpiEvidenceCandidate],
    tool_context: ToolContext,
) -> dict:
    """Build and store the final LEDGER extraction from grounded evidence.

    Args:
        ticker: Ticker copied exactly from get_report_info.
        reporting_currency: Three-letter uppercase reporting currency, or null if unstated.
        units_note: Short note describing observed statement scale, or null.
        kpis: Final unrecorded evidence/coverage rows in the same format accepted by
            record_multi_kpi_progress. Do not pass calculated values. Pass an empty list to submit
            the recorded evidence unchanged. Found and explicit-zero evidence becomes the final
            Ledger kpis list; absent and ambiguous coverage is omitted. An empty final extraction
            is accepted only after all 31 KPI keys have explicit absent/ambiguous coverage for the
            report fiscal year.

    Returns:
        Success with the validated ReportExtraction, or retryable field-level errors.
    """
    report, pages, state_error = report_from_state(tool_context)
    if state_error is not None:
        return state_error
    assert report is not None and pages is not None
    try:
        metadata = ReportExtraction.model_validate(
            {
                "ticker": ticker,
                "reporting_currency": reporting_currency,
                "units_note": units_note,
                "kpis": [],
            }
        )
    except ValidationError as error:
        return _validation_error_response(
            error,
            "extraction does not match the LEDGER ReportExtraction schema",
        )

    incoming_kpis, evidence_errors, _corrections = _normalize_multi_kpi_candidates(
        kpis,
        pages,
        int(report.get("year", 0)),
    )
    errors = _multi_kpi_submission_errors(metadata, report)
    errors.extend(evidence_errors)
    if errors:
        return {
            "status": "error",
            "retryable": True,
            "error": "extraction evidence failed basic data checks",
            "validation_errors": errors,
        }

    work_record = _load_multi_kpi_work_record(tool_context)
    indexed = (
        {(item.kpi, item.fiscal_year): item for item in work_record.kpis}
        if work_record is not None
        else {}
    )
    indexed.update({(item.kpi, item.fiscal_year): item for item in incoming_kpis})

    combined = MultiKpiWorkRecord(
        ticker=metadata.ticker,
        reporting_currency=(
            metadata.reporting_currency
            if metadata.reporting_currency is not None
            else work_record.reporting_currency
            if work_record is not None
            else None
        ),
        units_note=(
            metadata.units_note
            if metadata.units_note is not None
            else work_record.units_note
            if work_record is not None
            else None
        ),
        kpis=sorted(indexed.values(), key=lambda item: (item.fiscal_year, item.kpi)),
        notes=work_record.notes if work_record is not None else [],
    )
    extraction = ReportExtraction(
        ticker=combined.ticker,
        reporting_currency=combined.reporting_currency,
        units_note=combined.units_note,
        kpis=[
            {
                "kpi": item.kpi,
                "fiscal_year": item.fiscal_year,
                "value": item.value,
            }
            for item in combined.kpis
            if item.status in {"found", "explicit_zero"}
        ],
    )
    errors = _multi_kpi_submission_errors(extraction, report)
    report_year = int(report.get("year", 0))
    pending_kpis = _pending_multi_kpis(combined, report_year)
    allow_partial = bool(tool_context.state.get(MULTI_KPI_ALLOW_PARTIAL_STATE_KEY))
    if pending_kpis and not allow_partial:
        errors.append(
            {
                "field": "kpis",
                "message": (
                    f"complete coverage for every KPI in fiscal year {report_year} is required; "
                    f"pending {len(pending_kpis)}: " + ", ".join(pending_kpis)
                ),
            }
        )
    if errors:
        return {
            "status": "error",
            "retryable": True,
            "error": "combined extraction failed basic data checks",
            "validation_errors": errors,
            "coverage_count": len(KPI_KEYS) - len(pending_kpis),
            "pending_kpis": pending_kpis,
        }

    result = extraction.model_dump()
    audit = combined.model_dump()
    tool_context.state[MULTI_KPI_WORK_RECORD_STATE_KEY] = audit
    tool_context.state[MULTI_KPI_AUDIT_STATE_KEY] = audit
    tool_context.state[MULTI_KPI_RESULT_STATE_KEY] = result
    actions = getattr(tool_context, "actions", None)
    if actions is not None:
        actions.skip_summarization = True
    return {
        "status": "success",
        "completion_status": "incomplete" if pending_kpis else "complete",
        "coverage_count": len(KPI_KEYS) - len(pending_kpis),
        "pending_kpis": pending_kpis,
        "result": result,
    }


_NULLABLE_STRING_SCHEMA = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
}
_EVIDENCE_SCHEMA = MultiKpiEvidenceCandidate.model_json_schema()
_NOTE_SCHEMA = MultiKpiNote.model_json_schema()

record_multi_kpi_progress_tool = JsonSchemaFunctionTool(
    record_multi_kpi_progress,
    parameters_json_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reporting_currency": _NULLABLE_STRING_SCHEMA,
            "units_note": _NULLABLE_STRING_SCHEMA,
            "kpis": {
                "type": "array",
                "items": _EVIDENCE_SCHEMA,
                "maxItems": MAX_MULTI_KPI_RECORD_ROWS,
            },
            "notes": {"type": "array", "items": _NOTE_SCHEMA},
        },
        "required": ["reporting_currency", "units_note", "kpis", "notes"],
    },
)

submit_multi_kpi_extraction_tool = JsonSchemaFunctionTool(
    submit_multi_kpi_extraction,
    parameters_json_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ticker": {"type": "string", "minLength": 1},
            "reporting_currency": _NULLABLE_STRING_SCHEMA,
            "units_note": _NULLABLE_STRING_SCHEMA,
            "kpis": {"type": "array", "items": _EVIDENCE_SCHEMA},
        },
        "required": ["ticker", "reporting_currency", "units_note", "kpis"],
    },
)
