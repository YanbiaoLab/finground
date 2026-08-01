"""Validated submission tools that persist benchmark results in session state."""

from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy
from html import unescape
from typing import Any

from google.adk.tools import ToolContext
from pydantic import ValidationError

from finground.kpis import KPI_KEYS, POSITIVE_MAGNITUDE_KPIS, POSITIVE_OUTFLOW_KPIS
from finground.models import (
    MultiKpiEvidence,
    MultiKpiEvidenceCandidate,
    MultiKpiNote,
    MultiKpiRecordView,
    MultiKpiWorkRecord,
    ReportExtraction,
    UnitScale,
)
from finground.normalize import (
    NUMBER_TOKEN_RE,
    SCALE_MULTIPLIERS,
    detect_scale,
    normalize_value,
    parse_financial_number,
)
from finground.sec_facts import SEC_FACTS_STATE_KEY

from .report import (
    MULTI_KPI_SOURCE_CELLS_STATE_KEY,
    REPORT_STATE_KEY,
    report_from_state,
)
from .structured import JsonSchemaFunctionTool

ISO_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

MULTI_KPI_WORK_RECORD_STATE_KEY = "multi_kpi_work_record"
MULTI_KPI_RESULT_STATE_KEY = "multi_kpi_result"
MULTI_KPI_AUDIT_STATE_KEY = "multi_kpi_audit"
MULTI_KPI_ALLOW_PARTIAL_STATE_KEY = "multi_kpi_allow_partial_submission"
MULTI_KPI_REQUESTED_STATE_KEY = "multi_kpi_requested"

_EXPLICIT_ZERO_MARKERS = {"-", "−", "–", "—", "nil"}
MAX_MULTI_KPI_RECORD_ROWS = 16
_INVALID_SOURCE_CANDIDATE = object()

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
                    str(part) for part in ((prefix,) if prefix is not None else ()) + item["loc"]
                ),
                "message": item["msg"],
            }
            for item in error.errors()
        ],
    }


def _normalized_source_text(text: str) -> str:
    visible_text = re.sub(r"<[^>]+>", " ", unescape(text))
    return " ".join(visible_text.casefold().split())


def _semantic_row_error(
    kpi: str,
    line_label: str,
    status: str,
    statement: str | None = None,
) -> str | None:
    """Reject only row labels that cannot represent the requested canonical KPI."""
    label = _normalized_source_text(line_label)

    required_patterns: dict[str, tuple[str, ...]] = {
        "revenue": (
            r"\brevenues?\b",
            r"\bnet sales\b",
            r"\btotal sales\b",
            r"^sales$",
            r"\b(?:net product|oil and gas) sales\b",
            r"^total income(?: \(loss\))?$",
            r"^total revenues and other income$",
            r"^net portfolio income$",
        ),
        "cost_of_revenue": (
            r"\bcost of (?:revenues?|sales|goods|services|operations)\b",
            r"\bcost of goods (?:and services )?sold\b",
            r"\bcost of products sold\b",
            r"\bcosts? of operations\b",
        ),
        "gross_profit": (
            r"\bgross profit\b",
            r"\bgross margin\b",
            r"\bproduction margin\b",
        ),
        "rd_expense": (
            r"\bresearch and development\b",
            r"\br\s*&\s*d\b",
            r"\bdesign and development\b",
            r"\bresearch (?:and|&) engineering\b",
        ),
        "sga_expense": (
            r"\bselling,?\s+general,?\s+(?:and\s+)?administrative\b",
            r"\bgeneral\s+(?:and|&)\s+administrative\b",
            r"\bmarketing and administrative\b",
        ),
        "operating_income": (
            r"\boperating (?:income|profit|loss)\b",
            r"\b(?:income|loss) from operations\b",
            r"\bearnings from operations\b",
        ),
        "interest_expense": (r"\binterest expense\b", r"\bfinance costs?\b"),
        "income_tax_expense": (
            r"\bincome tax (?:expense|benefit|provision|recovery)\b",
            r"\bprovision for income taxes\b",
            r"\bprovision .* for income taxes\b",
            r"\bbenefit for income taxes\b",
            r"\btax expense \(?benefit\)?\b",
            r"\bincome taxes\b",
        ),
        "net_income": (
            r"\bnet (?:income|loss|earnings)\b",
            r"\bprofit (?:for the year|attributable)\b",
            r"\bincome \(?loss\)? from continuing operations attributable\b",
        ),
        "total_assets": (r"\btotal assets\b",),
        "total_liabilities": (r"\btotal liabilities\b",),
        "inventory": (r"\binventor(?:y|ies)\b",),
        "accounts_receivable": (
            r"\baccounts receivable\b",
            r"\btrade receivables?\b",
            r"\breceivables?\b",
            r"\bsales receivable\b",
        ),
        "accounts_payable": (r"\baccounts payable\b", r"\btrade payables?\b"),
        "operating_cash_flow": (
            r"\bnet cash .*\boperating activities\b",
            r"\bcash .*\boperating activities\b",
            r"\bnet operating cash\b",
            r"\bunlabeled numeric total.*\boperating activities\b",
        ),
        "investing_cash_flow": (
            r"\bnet cash .*\binvesting activities\b",
            r"\bcash .*\binvesting activities\b",
            r"\bnet investing cash\b",
            r"\bunlabeled numeric total.*\binvesting activities\b",
        ),
        "financing_cash_flow": (
            r"\bnet cash .*\bfinancing activities\b",
            r"\bcash .*\bfinancing activities\b",
            r"\bnet financing cash\b",
            r"\bunlabeled numeric total.*\bfinancing activities\b",
        ),
        "capex": (
            r"\b(?:purchase|purchases|payments?|acquisition|acquisitions) .*(?:property|plant|equipment)\b",
            r"\bcapital expenditures?\b",
            r"\b(?:additions|expenditures) .*(?:property|properties|plant|equipment|fixed assets|long-lived assets)\b",
        ),
        "depreciation_amortization": (
            r"\bdepreciation and amorti[sz]ation\b",
            r"\bdepreciation & amorti[sz]ation\b",
            r"\bdepreciation,? depletion and amorti[sz]ation\b",
            r"\bdepletion,? depreciation,? and amorti[sz]ation\b",
            r"\bdepreciation\b",
        ),
        "dividends_paid": (
            r"\bdividends? paid\b",
            r"\bpayments? of dividends?\b",
            r"\bcash dividends?\b",
            r"\bdividends? to (?:shareholders|stockholders)\b",
            r"\bdividend payments?\b",
            r"\bcash distributions?\b",
            r"\bdistributions? paid\b",
        ),
    }
    patterns = required_patterns.get(kpi)
    if patterns is not None and not any(re.search(pattern, label) for pattern in patterns):
        return f"{kpi} requires a row label matching its canonical financial concept"

    if kpi == "revenue" and re.search(r"\b(?:proceeds|gain|cost|unearned|deferred)\b", label):
        return "revenue requires the operating top line, not proceeds, gains, costs, or deferred revenue"
    if kpi == "gross_profit" and re.search(r"(?:%|percent|percentage)", label):
        return "gross_profit requires a monetary amount, not a margin percentage"
    if kpi == "interest_expense" and ("interest income" in label or "interest rate" in label):
        return "interest_expense excludes interest income and interest rates"
    if kpi == "income_tax_expense" and "income taxes paid" in label:
        return "income_tax_expense is accrual tax expense/benefit, not cash taxes paid"
    if (
        kpi == "capex"
        and re.search(r"\badditions?\b", label)
        and "cash flow" not in _normalized_source_text(statement or "")
    ):
        return (
            "capex requires a cash purchase/payment row; PP&E or property additions "
            "are accrual asset-note movements"
        )
    if kpi == "dividends_paid" and "per share" in label:
        return "dividends_paid requires the cash outflow, not dividends per share"

    cash_flow_activities = {
        "operating_cash_flow": "operating",
        "investing_cash_flow": "investing",
        "financing_cash_flow": "financing",
    }
    expected_activity = cash_flow_activities.get(kpi)
    if expected_activity is not None:
        conflicting = [
            activity
            for activity in ("operating", "investing", "financing")
            if activity != expected_activity and f"{activity} activities" in label
        ]
        if conflicting:
            return (
                f"{kpi} requires the net {expected_activity}-activities row; "
                f"the cited label is for {conflicting[0]} activities"
            )

    if kpi == "total_liabilities":
        if "liabilit" not in label:
            return "total_liabilities requires an explicitly labelled liabilities total"
        if (
            "liabilities and shareholders' equity" in label
            or "liabilities and stockholders' equity" in label
            or "liabilities and equity" in label
        ):
            return "total_liabilities excludes equity; cite the standalone total liabilities row"
        if "current liabilities" in label:
            return (
                "total_liabilities requires all liabilities, not the current-liabilities subtotal"
            )

    if kpi == "operating_income" and re.search(
        r"\b(?:income|earnings|profit)\s+before\s+(?:income\s+)?tax",
        label,
    ):
        return "operating_income requires an operating/EBIT row, not a pre-tax income row"

    if (
        kpi == "income_tax_expense"
        and ("current income tax" in label or "deferred income tax" in label)
        and "total" not in label
    ):
        return "income_tax_expense requires the total tax expense/benefit, not one tax component"

    if kpi == "shares_outstanding":
        if "weighted average" in label or "weighted-average" in label:
            return (
                "shares_outstanding requires a period-end share count, not weighted-average shares"
            )
        if "authorized" in label and "outstanding" not in label:
            return "authorized shares are not shares outstanding"

    if kpi in {"accounts_receivable", "accounts_payable"}:
        is_party_component = "related parties" in label or "unrelated parties" in label
        if is_party_component and "total" not in label:
            return f"{kpi} requires the total current balance, not one counterparty component"
    if kpi == "accounts_payable" and "accounts payable" in label and "accrued" in label:
        return "accounts_payable excludes accrued-expense balances combined into the cited row"

    if kpi in {"stockholders_equity", "stockholders_equity_incl_nci"} and (
        "percent" in label or "per share" in label or "return on" in label
    ):
        return f"{kpi} requires the period-end equity balance, not a ratio or per-share row"

    if kpi == "short_term_borrowings" and "interest rate" in label:
        return "short_term_borrowings requires the period-end principal balance, not its rate"
    if (
        kpi == "short_term_borrowings"
        and status == "explicit_zero"
        and re.search(r"\b(?:current portion|current maturities)\b", label)
    ):
        return (
            "a zero combined/current-maturities row does not establish a standalone "
            "short-term-borrowings balance"
        )

    if kpi == "long_term_debt_total" and (
        "future principal payment" in label or "future principle payment" in label
    ):
        return "long_term_debt_total requires the period-end balance, not future payment maturities"
    if (
        kpi == "long_term_debt_current"
        and "current portion of non-current liabilities" in label
        and not re.search(r"\b(?:debt|borrowings?|notes?|leases?)\b", label)
    ):
        return (
            "long_term_debt_current requires a debt, borrowing, note, or lease row; "
            "a generic current portion of all non-current liabilities is too broad"
        )

    return None


def _detected_unit_scale(kpi: str, unit_text: str | None) -> UnitScale:
    if kpi == "shares_outstanding" and re.search(
        r"\bexcept\s+(?:share|shares|share\s+and\s+per\s+share)\b",
        (unit_text or "").casefold(),
    ):
        return "units"
    return detect_scale(unit_text or "", kpi)


def _source_rows(page_text: str) -> list[str]:
    html_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", page_text, flags=re.IGNORECASE | re.DOTALL)
    prose_lines = [line for line in page_text.splitlines() if line.strip()]
    return [*html_rows, *prose_lines] or [page_text]


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
    for index, line in enumerate(page_text.splitlines()):
        heading = _normalized_source_text(line).lstrip("# ").strip()
        if heading != normalized_label:
            continue
        for following in page_text.splitlines()[index + 1 : index + 5]:
            if following.lstrip().startswith("#"):
                break
            for token in NUMBER_TOKEN_RE.findall(following):
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
    detected = _detected_unit_scale(candidate.kpi, candidate.unit_text)
    if detected != "unknown":
        return detected, "unit_text" if candidate.unit_text else "kpi"
    if candidate.unit_scale not in {None, "unknown"}:
        return candidate.unit_scale, "agent"
    return None, None


def _normalize_multi_kpi_candidates(
    raw_candidates: list[object],
    pages: list[Any],
    report_year: int,
    *,
    ledger_exchange: str | None = None,
    source_backed_indexes: set[int] | None = None,
) -> tuple[list[MultiKpiEvidence], list[dict[str, str]], list[dict[str, Any]]]:
    source_backed_indexes = source_backed_indexes or set()
    errors: list[dict[str, str]] = []
    candidates: list[tuple[int, MultiKpiEvidenceCandidate]] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if raw_candidate is _INVALID_SOURCE_CANDIDATE:
            continue
        try:
            candidates.append((index, MultiKpiEvidenceCandidate.model_validate(raw_candidate)))
        except ValidationError as error:
            response = _validation_error_response(
                error,
                "KPI evidence does not match the work-record schema",
                prefix=f"kpis.{index}",
            )
            errors.extend(response["validation_errors"])

    corrections: list[dict[str, Any]] = []
    normalized: list[MultiKpiEvidence] = []
    page_by_number = {page.display_number: page.text for page in pages}
    seen: set[tuple[str, int]] = set()
    for index, candidate in candidates:
        key = (candidate.kpi, candidate.fiscal_year)
        if key in seen:
            errors.append(
                {
                    "field": f"kpis.{index}",
                    "message": (
                        f"duplicate KPI/year coverage row: {candidate.kpi}/{candidate.fiscal_year}"
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
        semantic_error = _semantic_row_error(
            candidate.kpi,
            candidate.line_label,
            candidate.status,
            candidate.statement,
        )
        if semantic_error is not None:
            errors.append(
                {
                    "field": f"kpis.{index}.line_label",
                    "message": f"semantic mismatch: {semantic_error}",
                }
            )
            continue
        page_text = page_by_number.get(candidate.page)
        if page_text is None:
            errors.append(
                {
                    "field": f"kpis.{index}.page",
                    "message": "page does not exist in the report",
                }
            )
            continue
        evidence_error_count = len(errors)
        if (
            index not in source_backed_indexes
            and candidate.year_label
            and candidate.year_label.casefold() not in page_text.casefold()
        ):
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

        if candidate.unit_text is None and candidate.unit_scale in {
            "thousands",
            "millions",
            "billions",
        }:
            errors.append(
                {
                    "field": f"kpis.{index}.unit_text",
                    "message": "scaled values require exact visible unit_text and unit_page",
                }
            )
        elif (
            index not in source_backed_indexes
            and candidate.unit_text is None
            and candidate.unit_scale == "units"
        ):
            visible_header_scale = detect_scale(page_text[:2_000], candidate.kpi)
            if visible_header_scale in {"thousands", "millions", "billions"}:
                errors.append(
                    {
                        "field": f"kpis.{index}.unit_text",
                        "message": (
                            f"cited page header indicates {visible_header_scale}; copy its exact "
                            "unit_text and unit_page instead of using units"
                        ),
                    }
                )
        if len(errors) > evidence_error_count:
            continue

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
            aligned = (
                True
                if index in source_backed_indexes
                else _value_matches_year_column(page_text, candidate)
            )
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
            aligned = (
                True
                if index in source_backed_indexes
                else _value_matches_year_column(page_text, candidate)
            )
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
            preserve_ledger_outflow_sign = (
                candidate.kpi in POSITIVE_OUTFLOW_KPIS
                and (ledger_exchange or "").upper() == "LSE"
            )
            if preserve_ledger_outflow_sign:
                value = parsed_number * SCALE_MULTIPLIERS[scale]
                sign_rule = "as_reported"
            elif candidate.kpi in POSITIVE_MAGNITUDE_KPIS:
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


def _expand_source_backed_candidates(
    raw_candidates: list[object],
    tool_context: ToolContext,
) -> tuple[list[object], list[dict[str, str]], set[int]]:
    source_cells = tool_context.state.get(MULTI_KPI_SOURCE_CELLS_STATE_KEY, {})
    if not isinstance(source_cells, dict):
        source_cells = {}
    report = tool_context.state.get(REPORT_STATE_KEY, {})
    report_pages = report.get("pages", []) if isinstance(report, dict) else []
    page_text_by_number = {
        page.get("display_number"): str(page.get("text", ""))
        for page in report_pages
        if isinstance(page, dict)
    }

    def source_statement(raw_statement: object, page_numbers: list[object]) -> str | None:
        if isinstance(raw_statement, str) and raw_statement.strip():
            return raw_statement
        if all(
            "cash flow" in _normalized_source_text(page_text_by_number.get(page, "")[:1_500])
            for page in page_numbers
        ):
            return "Audited cash flow statement"
        return None

    expanded: list[object] = []
    errors: list[dict[str, str]] = []
    source_backed_indexes: set[int] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        if isinstance(raw_candidate, dict) and raw_candidate.get("source_ids"):
            source_ids = raw_candidate.get("source_ids")
            if raw_candidate.get("kpi") != "capex" or not isinstance(source_ids, list):
                errors.append(
                    {
                        "field": f"kpis.{index}.source_ids",
                        "message": "multiple source cells are supported only for capex",
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
            sources = [source_cells.get(source_id) for source_id in source_ids]
            if len(source_ids) < 2 or any(not isinstance(source, dict) for source in sources):
                errors.append(
                    {
                        "field": f"kpis.{index}.source_ids",
                        "message": "every capex component source_id must be available",
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
            typed_sources = [source for source in sources if isinstance(source, dict)]
            labels = [str(source.get("row_label", "")) for source in typed_sources]
            normalized_labels = [_normalized_source_text(label) for label in labels]
            has_property_component = any(
                re.search(r"\b(?:property|properties|plant|equipment|fixed assets)\b", label)
                for label in normalized_labels
            )
            allowed_component = all(
                re.search(
                    r"\b(?:purchase|purchases|additions|capitalised|capitalized)\b",
                    label,
                )
                and re.search(
                    r"\b(?:property|properties|plant|equipment|software|development|intangible)\b",
                    label,
                )
                for label in normalized_labels
            )
            years = {source.get("fiscal_year") for source in typed_sources}
            units = {source.get("unit_text") for source in typed_sources}
            statuses = {source.get("status") for source in typed_sources}
            if (
                not has_property_component
                or not allowed_component
                or len(years) != 1
                or len(units) != 1
                or statuses != {"found"}
            ):
                errors.append(
                    {
                        "field": f"kpis.{index}.source_ids",
                        "message": (
                            "capex components must be same-year, same-unit printed cash-investment "
                            "rows and include a property/plant/equipment component"
                        ),
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
            try:
                component_values = [
                    parse_financial_number(str(source["value_verbatim"]))
                    for source in typed_sources
                ]
            except (KeyError, ValueError):
                errors.append(
                    {
                        "field": f"kpis.{index}.source_ids",
                        "message": "capex component cells must contain financial numbers",
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
            total = sum(component_values)
            total_token = f"({abs(total):g})" if total < 0 else f"{total:g}"
            unit_text = next(iter(units))
            unit_scale = _detected_unit_scale("capex", unit_text)
            if unit_scale == "unknown":
                unit_scale = "units"
            first = typed_sources[0]
            source_backed_indexes.add(index)
            expanded.append(
                {
                    "kpi": "capex",
                    "fiscal_year": next(iter(years)),
                    "status": "found",
                    "value_verbatim": total_token,
                    "unit_scale": unit_scale,
                    "unit_text": unit_text,
                    "unit_page": first.get("unit_page", first["page"]),
                    "page": first["page"],
                    "statement": source_statement(
                        raw_candidate.get("statement"),
                        [source.get("page") for source in typed_sources],
                    ),
                    "line_label": " + ".join(labels),
                    "year_label": first["year_label"],
                    "scope": "sum of printed capex cash-investment components",
                    "source_ids": source_ids,
                }
            )
            continue
        if not isinstance(raw_candidate, dict) or "source_id" not in raw_candidate:
            expanded.append(raw_candidate)
            continue
        source_id = raw_candidate.get("source_id")
        source = source_cells.get(source_id) if isinstance(source_id, str) else None
        if not isinstance(source, dict):
            errors.append(
                {
                    "field": f"kpis.{index}.source_id",
                    "message": "source_id is not available in the active page batch",
                }
            )
            expanded.append(_INVALID_SOURCE_CANDIDATE)
            continue
        submitted_value = raw_candidate.get("value_verbatim")
        if submitted_value is not None:
            try:
                submitted_number = parse_financial_number(str(submitted_value))
                source_number = parse_financial_number(str(source.get("value_verbatim", "")))
            except ValueError:
                submitted_number = source_number = math.nan
            if not math.isclose(
                abs(submitted_number),
                abs(source_number),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    {
                        "field": f"kpis.{index}.source_id",
                        "message": (
                            "source_id points to a different printed number; omit source_id for "
                            "prose evidence or use the matching source cell"
                        ),
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
        submitted_status = raw_candidate.get("status")
        source_status = source.get("status")
        if submitted_status != source_status:
            errors.append(
                {
                    "field": f"kpis.{index}.status",
                    "message": f"status must match source cell status {source_status!r}",
                }
            )
            expanded.append(_INVALID_SOURCE_CANDIDATE)
            continue
        kpi = raw_candidate.get("kpi")
        if not isinstance(kpi, str):
            errors.append(
                {
                    "field": f"kpis.{index}.kpi",
                    "message": "source-backed evidence requires a KPI key",
                }
            )
            expanded.append(_INVALID_SOURCE_CANDIDATE)
            continue
        unit_text = source.get("unit_text")
        unit_page = source.get("unit_page", source["page"]) if unit_text is not None else None
        if unit_text is None and source_status == "found":
            try:
                source_number = parse_financial_number(str(source.get("value_verbatim", "")))
            except ValueError:
                source_number = math.nan
            for corroborating_source in source_cells.values():
                if not isinstance(corroborating_source, dict):
                    continue
                corroborating_unit = corroborating_source.get("unit_text")
                if (
                    corroborating_unit is None
                    or corroborating_source.get("fiscal_year") != source.get("fiscal_year")
                    or corroborating_source.get("status") != source_status
                    or _semantic_row_error(
                        kpi,
                        str(corroborating_source.get("row_label", "")),
                        str(source_status),
                        None,
                    )
                    is not None
                ):
                    continue
                try:
                    corroborating_number = parse_financial_number(
                        str(corroborating_source.get("value_verbatim", ""))
                    )
                except ValueError:
                    continue
                if math.isclose(
                    abs(source_number),
                    abs(corroborating_number),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    unit_text = corroborating_unit
                    unit_page = corroborating_source.get(
                        "unit_page", corroborating_source.get("page")
                    )
                    break
        if (
            unit_text is None
            and source_status != "explicit_zero"
            and kpi not in {"eps_basic", "eps_diluted", "shares_outstanding"}
        ):
            visible_header_scale = detect_scale(
                page_text_by_number.get(source.get("page"), "")[:2_000], kpi
            )
            if visible_header_scale in {"thousands", "millions", "billions"}:
                errors.append(
                    {
                        "field": f"kpis.{index}.source_id",
                        "message": (
                            f"source page indicates {visible_header_scale} but the source cell "
                            "has no traceable unit text"
                        ),
                    }
                )
                expanded.append(_INVALID_SOURCE_CANDIDATE)
                continue
        unit_scale = _detected_unit_scale(kpi, unit_text)
        if unit_scale == "unknown":
            unit_scale = "units"
        source_backed_indexes.add(index)
        printed_row_label = source.get("printed_row_label")
        section_label = source.get("section_label")
        line_label = (
            printed_row_label
            if isinstance(printed_row_label, str)
            else f"Unlabeled numeric total in {section_label}"
            if isinstance(section_label, str)
            else source["row_label"]
        )
        expanded.append(
            {
                "kpi": kpi,
                "fiscal_year": source["fiscal_year"],
                "status": source_status,
                "value_verbatim": source["value_verbatim"],
                "unit_scale": unit_scale,
                "unit_text": unit_text,
                "unit_page": unit_page,
                "page": source["page"],
                "statement": source_statement(
                    raw_candidate.get("statement"),
                    [source.get("page")],
                ),
                "line_label": line_label,
                "year_label": source["year_label"],
                "scope": (
                    source["row_label"]
                    if printed_row_label is None and isinstance(source.get("row_label"), str)
                    else None
                ),
            }
        )
    return expanded, errors, source_backed_indexes


def _work_record_counts(work_record: MultiKpiWorkRecord) -> tuple[int, Counter[str]]:
    status_counts = Counter(item.status for item in work_record.kpis)
    extracted_count = status_counts["found"] + status_counts["explicit_zero"]
    return extracted_count, status_counts


def _requested_multi_kpis(tool_context: ToolContext) -> tuple[str, ...]:
    requested = tool_context.state.get(MULTI_KPI_REQUESTED_STATE_KEY)
    if isinstance(requested, list):
        selected = tuple(kpi for kpi in KPI_KEYS if kpi in requested)
        if selected:
            return selected
    return KPI_KEYS


def _pending_multi_kpis(
    work_record: MultiKpiWorkRecord,
    report_year: int,
    requested_kpis: tuple[str, ...] = KPI_KEYS,
    structured_kpis: set[str] | None = None,
) -> list[str]:
    covered = {item.kpi for item in work_record.kpis if item.fiscal_year == report_year}
    covered.update(structured_kpis or ())
    return [kpi for kpi in requested_kpis if kpi not in covered]


def _structured_kpis(tool_context: ToolContext) -> set[str]:
    sec_facts = tool_context.state.get(SEC_FACTS_STATE_KEY, {})
    values = sec_facts.get("values", {}) if isinstance(sec_facts, dict) else {}
    return {
        kpi
        for kpi, fact in values.items()
        if kpi in KPI_KEYS and isinstance(fact, dict) and isinstance(fact.get("value"), int | float)
    }


def _deterministic_derived_values(
    work_record: MultiKpiWorkRecord,
    report_year: int,
) -> tuple[dict[str, float], list[str]]:
    """Apply only LEDGER-aligned, auditable statement identities."""
    rows = {item.kpi: item for item in work_record.kpis if item.fiscal_year == report_year}
    values = {
        kpi: float(item.value)
        for kpi, item in rows.items()
        if item.status in {"found", "explicit_zero"} and item.value is not None
    }
    derived: dict[str, float] = {}
    formulas: list[str] = []

    def absent(kpi: str) -> bool:
        item = rows.get(kpi)
        return item is not None and item.status == "absent"

    def add(kpi: str, value: float, formula: str) -> None:
        if absent(kpi) and math.isfinite(value):
            derived[kpi] = value
            formulas.append(f"{kpi} = {formula}")

    if {"revenue", "cost_of_revenue"} <= values.keys():
        add(
            "gross_profit",
            values["revenue"] - values["cost_of_revenue"],
            "revenue - cost_of_revenue",
        )
    if {"revenue", "gross_profit"} <= values.keys():
        add(
            "cost_of_revenue",
            values["revenue"] - values["gross_profit"],
            "revenue - gross_profit",
        )
    if {"total_assets", "stockholders_equity_incl_nci"} <= values.keys():
        add(
            "total_liabilities",
            values["total_assets"] - values["stockholders_equity_incl_nci"],
            "total_assets - stockholders_equity_incl_nci",
        )
    elif {"total_assets", "stockholders_equity"} <= values.keys() and absent(
        "stockholders_equity_incl_nci"
    ):
        add(
            "total_liabilities",
            values["total_assets"] - values["stockholders_equity"],
            "total_assets - stockholders_equity (NCI-inclusive equity absent)",
        )
    if {"total_assets", "total_liabilities"} <= values.keys():
        add(
            "stockholders_equity_incl_nci",
            values["total_assets"] - values["total_liabilities"],
            "total_assets - total_liabilities",
        )
    if {"long_term_debt_noncurrent", "long_term_debt_current"} <= values.keys():
        add(
            "long_term_debt_total",
            values["long_term_debt_noncurrent"] + values["long_term_debt_current"],
            "long_term_debt_noncurrent + long_term_debt_current",
        )
    return derived, formulas


def _load_multi_kpi_work_record(tool_context: ToolContext) -> MultiKpiWorkRecord | None:
    raw_record = tool_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY)
    if raw_record is None:
        return None
    try:
        return MultiKpiWorkRecord.model_validate(raw_record)
    except ValidationError as error:
        raise RuntimeError("multi-KPI work record in session state is invalid") from error


def _evidence_repair_queue(
    candidates: list[MultiKpiEvidenceCandidate],
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    indexed_errors: dict[int, list[dict[str, str]]] = {}
    for error in errors:
        match = re.match(r"^kpis\.(\d+)\.", error.get("field", ""))
        if match is not None:
            indexed_errors.setdefault(int(match.group(1)), []).append(error)
    repair_queue = []
    for index, validation_errors in sorted(indexed_errors.items()):
        if index >= len(candidates):
            continue
        candidate = candidates[index]
        kpi = candidate.kpi if isinstance(candidate, MultiKpiEvidenceCandidate) else None
        if kpi is None and isinstance(candidate, dict):
            kpi = candidate.get("kpi")
        repair_queue.append(
            {
                "index": index,
                "kpi": kpi,
                "validation_errors": validation_errors,
            }
        )
    return repair_queue


def record_multi_kpi_progress(
    reporting_currency: str | None,
    units_note: str | None,
    kpis: list[object],
    notes: list[MultiKpiNote],
    tool_context: ToolContext,
) -> dict:
    """Validate and record grounded Multi-KPI evidence, coverage, and notes.

    Args:
        reporting_currency: Three-letter uppercase reporting currency, or null if still unknown.
        units_note: Short note describing observed statement scale, or null.
        kpis: Evidence/coverage rows. Prefer a source-backed found/explicit-zero row containing
            only kpi, status, and a source_id from the active read_report_pages result. For report
            prose without a source_id, use the full evidence fields. Use status="absent" or
            "ambiguous" without a value for unresolved coverage; those rows are retained for work
            tracking and omitted from the final LEDGER output.
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
    note_errors: list[dict[str, str]] = []
    indexed_notes: list[tuple[int, MultiKpiNote]] = []
    for note_index, raw_note in enumerate(notes):
        try:
            indexed_notes.append((note_index, MultiKpiNote.model_validate(raw_note)))
        except ValidationError as error:
            response = _validation_error_response(
                error,
                "note does not match the Multi-KPI work-record schema",
                prefix=f"notes.{note_index}",
            )
            note_errors.extend(response["validation_errors"])
    expanded_kpis, source_errors, source_backed_indexes = _expand_source_backed_candidates(
        kpis,
        tool_context,
    )
    incoming_kpis, evidence_errors, corrections = _normalize_multi_kpi_candidates(
        expanded_kpis,
        pages,
        int(report.get("year", 0)),
        ledger_exchange=str(report.get("exchange", "")),
        source_backed_indexes=source_backed_indexes,
    )
    evidence_errors = [*source_errors, *evidence_errors]
    metadata_errors = _multi_kpi_submission_errors(metadata, report)
    if metadata_errors:
        metadata = metadata.model_copy(update={"reporting_currency": None})
    valid_pages = {page.display_number for page in pages}
    valid_notes: list[MultiKpiNote] = []
    for note_index, note in indexed_notes:
        missing_pages = [page for page in note.pages if page not in valid_pages]
        if missing_pages:
            note_errors.append(
                {
                    "field": f"notes.{note_index}.pages",
                    "message": f"pages do not exist in the report: {missing_pages}",
                }
            )
        else:
            valid_notes.append(note)
    incoming_notes = valid_notes
    recoverable_errors = [*metadata_errors, *note_errors, *evidence_errors]
    if (
        not incoming_kpis
        and not incoming_notes
        and metadata.reporting_currency is None
        and metadata.units_note is None
    ):
        validation_errors = recoverable_errors or [
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
            "accepted_kpis": [],
            "repair_queue": _evidence_repair_queue(kpis, validation_errors),
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
    requested_kpis = _requested_multi_kpis(tool_context)
    pending_kpis = _pending_multi_kpis(
        combined,
        int(report.get("year", 0)),
        requested_kpis,
        _structured_kpis(tool_context),
    )
    response = {
        "status": "partial_success" if recoverable_errors else "success",
        "kpi_count": extracted_count,
        "coverage_count": len(requested_kpis) - len(pending_kpis),
        "pending_count": len(pending_kpis),
        "pending_kpis": pending_kpis,
        "status_counts": dict(status_counts),
        "note_count": len(combined.notes),
        "added_kpi_count": added,
        "updated_kpi_count": updated,
        "added_note_count": len(added_notes),
        "normalization_corrections": corrections,
    }
    if recoverable_errors:
        response.update(
            {
                "retryable": True,
                "error": "valid progress was saved; correct only the rejected fields",
                "validation_errors": recoverable_errors,
                "accepted_kpis": [item.kpi for item in incoming_kpis],
                "repair_queue": _evidence_repair_queue(kpis, evidence_errors),
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
    requested_kpis = _requested_multi_kpis(tool_context)
    pending_kpis = _pending_multi_kpis(
        work_record,
        int(report.get("year", 0)),
        requested_kpis,
        _structured_kpis(tool_context),
    )
    return {
        "status": "success",
        "view": view,
        "kpi_count": extracted_count,
        "coverage_count": len(requested_kpis) - len(pending_kpis),
        "pending_count": len(pending_kpis),
        "pending_kpis": pending_kpis,
        "status_counts": dict(status_counts),
        "note_count": len(work_record.notes),
        "record": record,
    }


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
        ledger_exchange=str(report.get("exchange", "")),
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
    requested_kpis = _requested_multi_kpis(tool_context)
    derived_values, derivation_formulas = _deterministic_derived_values(
        combined,
        int(report.get("year", 0)),
    )
    report_year = int(report.get("year", 0))
    output_values = {
        item.kpi: float(item.value)
        for item in combined.kpis
        if item.kpi in requested_kpis
        and item.status in {"found", "explicit_zero"}
        and item.value is not None
    }
    output_values.update(
        {kpi: value for kpi, value in derived_values.items() if kpi in requested_kpis}
    )
    sec_facts = tool_context.state.get(SEC_FACTS_STATE_KEY, {})
    structured_values = sec_facts.get("values", {}) if isinstance(sec_facts, dict) else {}
    structured_audit: dict[str, Any] = {}
    for kpi in requested_kpis:
        fact = structured_values.get(kpi)
        if not isinstance(fact, dict) or not isinstance(fact.get("value"), int | float):
            continue
        output_values[kpi] = float(fact["value"])
        structured_audit[kpi] = {
            "value": float(fact["value"]),
            "concept": fact.get("concept"),
            "source": fact.get("source") or sec_facts.get("source"),
        }
    extraction = ReportExtraction(
        ticker=combined.ticker,
        reporting_currency=combined.reporting_currency,
        units_note=combined.units_note,
        kpis=[
            {
                "kpi": kpi,
                "fiscal_year": report_year,
                "value": value,
            }
            for kpi, value in output_values.items()
        ],
    )
    errors = _multi_kpi_submission_errors(extraction, report)
    pending_kpis = _pending_multi_kpis(
        combined,
        report_year,
        requested_kpis,
        _structured_kpis(tool_context),
    )
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
            "coverage_count": len(requested_kpis) - len(pending_kpis),
            "pending_kpis": pending_kpis,
        }

    result = extraction.model_dump()
    work_record_result = combined.model_dump()
    audit = {
        **work_record_result,
        "derivations": derivation_formulas,
        "structured_facts": structured_audit,
        "structured_sources": {
            "sec_company_facts": {
                "status": sec_facts.get("status"),
                "source": sec_facts.get("source"),
                "cik": sec_facts.get("cik"),
            },
        },
    }
    tool_context.state[MULTI_KPI_WORK_RECORD_STATE_KEY] = work_record_result
    tool_context.state[MULTI_KPI_AUDIT_STATE_KEY] = audit
    tool_context.state[MULTI_KPI_RESULT_STATE_KEY] = result
    actions = getattr(tool_context, "actions", None)
    if actions is not None:
        actions.skip_summarization = True
    return {
        "status": "success",
        "completion_status": "incomplete" if pending_kpis else "complete",
        "coverage_count": len(requested_kpis) - len(pending_kpis),
        "pending_kpis": pending_kpis,
        "result": result,
    }


def finalize_multi_kpi_report(tool_context: ToolContext) -> dict:
    """Submit the state-backed Multi-KPI result without duplicating evidence.

    Returns:
        Complete extraction when all KPI specialists have recorded a status, or
        compact retryable feedback listing the remaining KPI specialists.
    """
    report, _pages, state_error = report_from_state(tool_context)
    if state_error is not None:
        return state_error
    assert report is not None
    work_record = _load_multi_kpi_work_record(tool_context)
    return submit_multi_kpi_extraction(
        ticker=str(report.get("ticker", "")),
        reporting_currency=(work_record.reporting_currency if work_record is not None else None),
        units_note=work_record.units_note if work_record is not None else None,
        kpis=[],
        tool_context=tool_context,
    )


_NULLABLE_STRING_SCHEMA = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
}
_EVIDENCE_SCHEMA = MultiKpiEvidenceCandidate.model_json_schema()
_RECORD_EVIDENCE_SCHEMA = deepcopy(_EVIDENCE_SCHEMA)
_RECORD_EVIDENCE_SCHEMA["properties"]["source_id"] = {
    "type": "string",
    "minLength": 1,
    "description": "Opaque source cell ID returned by the active read_report_pages result.",
}
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
                "items": _RECORD_EVIDENCE_SCHEMA,
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
