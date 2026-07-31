"""Streaming readers for official LEDGER Parquet inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from finground.documents import Report

REPORT_COLUMNS = ("ticker", "exchange", "year", "mmd_text")


def parquet_files(path: Path) -> list[Path]:
    """Resolve one Parquet file or a directory of Parquet shards."""
    if path.is_file():
        if path.suffix.casefold() != ".parquet":
            raise ValueError(f"expected a .parquet file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"no Parquet files found at {path}")


def iter_parquet_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    batch_size: int,
) -> Iterator[dict[str, object]]:
    """Stream only the requested Parquet columns using bounded record batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for parquet_path in parquet_files(path):
        parquet = pq.ParquetFile(parquet_path)
        missing = set(columns).difference(parquet.schema_arrow.names)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{parquet_path} is missing required columns: {names}")
        for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
            yield from batch.to_pylist()


def _report(row: dict[str, object]) -> Report:
    if any(row[name] is None for name in REPORT_COLUMNS):
        raise ValueError("ticker, exchange, year, and mmd_text must not be null")
    ticker = str(row["ticker"]).strip()
    exchange = str(row["exchange"]).strip()
    year = int(row["year"])
    mmd_text = str(row["mmd_text"])
    if not ticker or not exchange or not mmd_text.strip():
        raise ValueError("ticker, exchange, and mmd_text must be non-empty")
    report_id = f"{exchange}_{ticker}_{year}"
    return Report(report_id, exchange, ticker, year, mmd_text=mmd_text)


def iter_multi_reports(path: Path) -> Iterator[Report]:
    """Yield unique Multi-KPI reports without reading any KPI answer columns."""
    seen: set[str] = set()
    for row in iter_parquet_rows(path, REPORT_COLUMNS, batch_size=16):
        report = _report(row)
        if report.report_id in seen:
            continue
        seen.add(report.report_id)
        yield report
