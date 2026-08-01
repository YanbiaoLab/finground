"""Select small, report-grounded KPI evaluation slices from LEDGER Parquet."""

from __future__ import annotations

import math
import re
from html import unescape
from pathlib import Path

from finground.benchmark.parquet import REPORT_COLUMNS, iter_parquet_rows
from finground.kpis import KPI_ALIASES, KPI_KEYS


def _visible_text(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", unescape(text)).split()).casefold()


def _printed_value_tokens(value: float) -> tuple[str, ...]:
    """Return exact printable forms at common LEDGER statement scales."""
    absolute = abs(value)
    tokens: list[str] = []
    for divisor in (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0):
        scaled = absolute / divisor
        if not math.isfinite(scaled):
            continue
        if scaled.is_integer():
            tokens.extend((f"{scaled:,.0f}", f"{scaled:.0f}"))
        else:
            tokens.extend((f"{scaled:,.3f}".rstrip("0").rstrip("."), f"{scaled:g}"))
    return tuple(dict.fromkeys(token for token in tokens if len(token) >= 2))


def select_grounded_report_ids(
    parquet_path: Path,
    *,
    kpi: str,
    limit: int,
    max_per_ticker: int = 1,
) -> list[str]:
    """Select reports whose KPI value and a canonical label are visible in report text."""
    if kpi not in KPI_KEYS:
        raise ValueError(f"unknown KPI: {kpi}")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_per_ticker < 1:
        raise ValueError("max_per_ticker must be at least 1")
    columns = (*REPORT_COLUMNS, kpi)
    labels = tuple(alias.casefold() for alias in KPI_ALIASES[kpi])
    selected: list[str] = []
    ticker_counts: dict[str, int] = {}
    for row in iter_parquet_rows(parquet_path, columns, batch_size=32):
        raw_value = row.get(kpi)
        if not isinstance(raw_value, int | float) or not math.isfinite(float(raw_value)):
            continue
        visible = _visible_text(str(row["mmd_text"]))
        if not any(label in visible for label in labels):
            continue
        if not any(token.casefold() in visible for token in _printed_value_tokens(float(raw_value))):
            continue
        ticker = str(row["ticker"])
        if ticker_counts.get(ticker, 0) >= max_per_ticker:
            continue
        selected.append(f"{row['exchange']}_{row['ticker']}_{int(row['year'])}")
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        if len(selected) == limit:
            break
    return selected


def write_grounded_report_ids(
    parquet_path: Path,
    *,
    kpi: str,
    limit: int,
    max_per_ticker: int,
    output_file: Path,
) -> dict[str, object]:
    """Write a reusable reports-file accepted by ledger-kpi."""
    report_ids = select_grounded_report_ids(
        parquet_path,
        kpi=kpi,
        limit=limit,
        max_per_ticker=max_per_ticker,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(f"{report_id}\n" for report_id in report_ids), encoding="utf-8")
    return {
        "kpi": kpi,
        "requested_limit": limit,
        "max_per_ticker": max_per_ticker,
        "selected_count": len(report_ids),
        "report_ids": report_ids,
        "output_file": str(output_file),
    }
