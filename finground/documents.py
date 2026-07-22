"""Page-aligned report parsing and discovery for FinGround agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PAGE_SPLIT_RE = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Page:
    """One non-empty page, retaining LEDGER's raw zero-based split index."""

    raw_index: int
    text: str

    @property
    def display_number(self) -> int:
        return self.raw_index + 1


@dataclass(frozen=True, slots=True)
class Report:
    report_id: str
    exchange: str
    ticker: str
    year: int
    mmd_text: str = field(repr=False)


def split_pages(raw: str) -> list[Page]:
    """Split exactly like LEDGER retrieval: blank segments still consume an index."""
    pages: list[Page] = []
    for raw_index, segment in enumerate(PAGE_SPLIT_RE.split(raw)):
        text = segment.strip()
        if text:
            pages.append(Page(raw_index=raw_index, text=text))
    return pages


def load_report_pages(report: Report) -> list[Page]:
    """Load page-aligned OCR Markdown embedded in a Parquet report row."""
    return split_pages(report.mmd_text)
