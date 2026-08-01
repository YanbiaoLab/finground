"""Compact, deterministic source cells extracted from report tables."""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from finground.normalize import NUMBER_TOKEN_RE

_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(
    r"<t[dh]\b([^>]*)>(.*?)</t[dh]>",
    re.IGNORECASE | re.DOTALL,
)
_UNIT_LINE_RE = re.compile(
    r"""(?ix)
    (?:
        \([^()<>\n]{0,120}\b(?:thousands|millions|billions)\b[^()<>\n]{0,120}\)
        |
        (?:amounts?\s+)?in\s+(?:thousands|millions|billions)\b[^<\n]{0,120}
        |
        (?:thousands|millions|billions)\s+of\s+
        (?:[a-z]+\s+){0,4}(?:dollars|euros|pounds|yen|shares)\b
        |
        (?:[$€£¥]\s*)?['’]000\b
    )
    """
)
_ZERO_MARKERS = {"-", "−", "–", "—", "nil"}
_MAJOR_SECTION_RE = re.compile(
    r"\b(?:operating|investing|financing)\s+activities\b|"
    r"\b(?:assets|liabilities|equity)\b",
    re.IGNORECASE,
)


def _visible_text(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", unescape(text)).split())


def _normalized(text: str) -> str:
    return _visible_text(text).casefold()


def _html_span(attributes: str, name: str) -> int:
    match = re.search(
        rf"\b{name}\s*=\s*[\"']?(\d+)",
        attributes,
        flags=re.IGNORECASE,
    )
    return max(1, int(match.group(1))) if match else 1


def _expand_rows(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
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
                    _visible_text(cell_html),
                    _html_span(attributes, "colspan"),
                    _html_span(attributes, "rowspan"),
                )
                for attributes, cell_html in _HTML_CELL_RE.findall(row_match.group(1))
            ]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(_expand_rows(rows))
    return tables


def _markdown_table_grids(page_text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped[1:-1].split("|")]
            is_separator = bool(cells) and all(
                not cell or re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
            )
            if not is_separator:
                current.append(cells)
        elif current:
            if len(current) > 1:
                tables.append(current)
            current = []
    if len(current) > 1:
        tables.append(current)
    return tables


def _page_unit_text(page_text: str) -> str | None:
    match = _UNIT_LINE_RE.search(page_text[:4_000])
    return _visible_text(match.group()) if match is not None else None


def _html_table_unit_texts(page_text: str) -> list[str | None]:
    """Return the nearest governing unit text for each HTML table."""
    units: list[str | None] = []
    page_unit = _page_unit_text(page_text)
    previous_table_end = 0
    for table_match in _HTML_TABLE_RE.finditer(page_text):
        prefix_start = max(previous_table_end, table_match.start() - 2_000)
        prefix = unescape(page_text[prefix_start : table_match.start()])
        table_header = unescape(table_match.group()[:1_000])
        prefix_matches = list(_UNIT_LINE_RE.finditer(prefix))
        header_match = _UNIT_LINE_RE.search(table_header)
        match = header_match or (prefix_matches[-1] if prefix_matches else None)
        units.append(_visible_text(match.group()) if match is not None else page_unit)
        previous_table_end = table_match.end()
    return units


def _share_count_source_cells(
    page_text: str,
    *,
    page_number: int,
    fiscal_year: int,
) -> list[dict[str, Any]]:
    """Extract period-end common-share counts embedded inside verbose row labels."""
    visible_cells = [
        _visible_text(cell_html) for _attributes, cell_html in _HTML_CELL_RE.findall(page_text)
    ]
    candidates = [
        *visible_cells,
        *(_visible_text(line) for line in page_text.splitlines() if "<table" not in line),
    ]
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for text in candidates:
        lowered = text.casefold()
        if (
            "outstanding" not in lowered
            or not any(term in lowered for term in ("common", "ordinary", "sub-share"))
            or "weighted average" in lowered
            or "preferred" in lowered
        ):
            continue
        relevant = text
        authorized = lowered.rfind("authorized")
        if authorized >= 0:
            relevant = text[authorized + len("authorized") :]
        numeric = []
        for token in NUMBER_TOKEN_RE.findall(relevant):
            try:
                value = abs(float(re.sub(r"[^0-9.]", "", token)))
            except ValueError:
                continue
            if value >= 1_000 and not 1990 <= value <= 2100:
                numeric.append((token.strip(), value))
        if not numeric:
            continue
        value_verbatim = numeric[0][0]
        key = (text, value_verbatim)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "source_id": f"p{page_number}:shares:{len(results)}",
                "page": page_number,
                "row_label": text,
                "printed_row_label": text,
                "row_role": "embedded_share_count",
                "section_label": "Common shares outstanding",
                "previous_label": None,
                "next_label": None,
                "year_label": f"{fiscal_year} (first period-end share count)",
                "year_inferred": True,
                "fiscal_year": fiscal_year,
                "value_verbatim": value_verbatim,
                "status": "found",
                "unit_text": None,
            }
        )
    return results


def _year_columns(table: list[list[str]], row_index: int, fiscal_year: int) -> dict[int, str]:
    year = str(fiscal_year)
    return {
        column: cell
        for header in table[: min(row_index, 4)]
        for column, cell in enumerate(header)
        if len(cell) <= 80 and re.search(rf"(?<!\d){re.escape(year)}(?!\d)", cell)
    }


def _row_label(row: list[str], year_columns: dict[int, str]) -> str | None:
    first_year_column = min(year_columns)
    for cell in row[:first_year_column]:
        if cell and re.search(r"[A-Za-z]", cell):
            return cell
    return None


def _row_has_target_value(row: list[str], year_columns: dict[int, str]) -> bool:
    for column in year_columns:
        if column >= len(row):
            continue
        cell = row[column].strip()
        if NUMBER_TOKEN_RE.findall(cell):
            return True
        if re.sub(r"[$€£¥\s]", "", cell).casefold() in _ZERO_MARKERS:
            return True
    return False


def _structural_context(
    table: list[list[str]],
    row_index: int,
    year_columns: dict[int, str],
) -> dict[str, str | None]:
    previous_label: str | None = None
    section_label: str | None = None
    nearest_heading: str | None = None
    for earlier_row in reversed(table[:row_index]):
        label = _row_label(earlier_row, year_columns)
        if label is None:
            continue
        if previous_label is None:
            previous_label = label
        if not _row_has_target_value(earlier_row, year_columns):
            if nearest_heading is None:
                nearest_heading = label
            if _MAJOR_SECTION_RE.search(label):
                section_label = label
                break
    if section_label is None:
        section_label = nearest_heading

    next_label = next(
        (
            label
            for later_row in table[row_index + 1 :]
            if (label := _row_label(later_row, year_columns)) is not None
        ),
        None,
    )
    context_parts = [
        f"section: {section_label}" if section_label is not None else None,
        f"after: {previous_label}" if previous_label is not None else None,
        f"before: {next_label}" if next_label is not None else None,
    ]
    context = "; ".join(part for part in context_parts if part is not None)
    return {
        "row_label": f"[unlabeled numeric row; {context}]",
        "printed_row_label": None,
        "row_role": "unlabeled_numeric",
        "section_label": section_label,
        "previous_label": previous_label,
        "next_label": next_label,
    }


def extract_source_cells(
    page_text: str,
    *,
    page_number: int,
    fiscal_year: int,
    allow_implicit_year: bool = False,
) -> list[dict[str, Any]]:
    """Return target-year table cells that can be cited by an opaque source ID."""
    source_cells: list[dict[str, Any]] = []
    html_tables = _html_table_grids(page_text)
    html_units = _html_table_unit_texts(page_text)
    tables = [
        *((table, html_units[index]) for index, table in enumerate(html_tables)),
        *((table, _page_unit_text(page_text)) for table in _markdown_table_grids(page_text)),
    ]
    for table_index, (table, unit_text) in enumerate(tables):
        for row_index, row in enumerate(table):
            year_columns = _year_columns(table, row_index, fiscal_year)
            inferred_year = False
            if not year_columns and allow_implicit_year:
                label_column = next(
                    (
                        column
                        for column, cell in enumerate(row)
                        if cell and re.search(r"[A-Za-z]", cell)
                    ),
                    None,
                )
                value_column = next(
                    (
                        column
                        for column, cell in enumerate(row)
                        if label_column is not None
                        and column > label_column
                        and (
                            bool(NUMBER_TOKEN_RE.findall(cell))
                            or re.sub(r"[$€£¥\s]", "", cell).casefold() in _ZERO_MARKERS
                        )
                    ),
                    None,
                )
                if value_column is not None:
                    year_columns = {value_column: f"{fiscal_year} (first value column)"}
                    inferred_year = True
            if not year_columns:
                continue
            printed_row_label = _row_label(row, year_columns)
            nearby_context = _structural_context(table, row_index, year_columns)
            structural_context = (
                {
                    **nearby_context,
                    "row_label": printed_row_label,
                    "printed_row_label": printed_row_label,
                    "row_role": "labeled",
                }
                if printed_row_label is not None
                else nearby_context
            )
            for column, year_label in year_columns.items():
                if column >= len(row):
                    continue
                cell = row[column].strip()
                tokens = NUMBER_TOKEN_RE.findall(cell)
                zero_marker = re.sub(r"[$€£¥\s]", "", cell).casefold()
                if len(tokens) == 1:
                    value_verbatim = tokens[0]
                    status = "found"
                elif zero_marker in _ZERO_MARKERS:
                    value_verbatim = cell
                    status = "explicit_zero"
                else:
                    continue
                source_cells.append(
                    {
                        "source_id": (f"p{page_number}:t{table_index}:r{row_index}:c{column}"),
                        "page": page_number,
                        **structural_context,
                        "year_label": year_label,
                        "year_inferred": inferred_year,
                        "fiscal_year": fiscal_year,
                        "value_verbatim": value_verbatim,
                        "status": status,
                        "unit_text": unit_text,
                    }
                )
    source_cells.extend(
        _share_count_source_cells(
            page_text,
            page_number=page_number,
            fiscal_year=fiscal_year,
        )
    )
    return source_cells
