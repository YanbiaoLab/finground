"""Deterministic value parsing, scale detection, and evidence validation."""

from __future__ import annotations

import re

from finground.kpis import PER_SHARE_KPIS, POSITIVE_OUTFLOW_KPIS
from finground.models import UnitScale

NUMBER_CLEAN_RE = re.compile(r"[^0-9.\-()]")
NUMBER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[$€£¥]\s*)?(?:\(\s*)?[+-]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*\))?"
)
SCALE_MULTIPLIERS: dict[UnitScale, float] = {
    "units": 1.0,
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
    "per_share": 1.0,
    "currency_subunits_per_share": 0.01,
    "unknown": 1.0,
}


def parse_financial_number(value_verbatim: str) -> float:
    cleaned = value_verbatim.strip().replace("−", "-").replace("–", "-").replace(",", "")
    negative_parentheses = "(" in cleaned and ")" in cleaned
    cleaned = NUMBER_CLEAN_RE.sub("", cleaned).replace("(", "").replace(")", "")
    if cleaned in {"", "-", "."}:
        raise ValueError(f"not a financial number: {value_verbatim!r}")
    value = float(cleaned)
    return -abs(value) if negative_parentheses else value


def detect_scale(text: str, kpi: str) -> UnitScale:
    lowered = text.lower()
    if kpi in PER_SHARE_KPIS:
        if re.search(r"\b(?:cents?|pence)\b", lowered):
            return "currency_subunits_per_share"
        return "per_share"
    if re.search(r"\b(?:in\s+)?billions?\b", lowered):
        return "billions"
    if re.search(r"\b(?:in\s+)?millions?\b", lowered):
        return "millions"
    if re.search(r"\b(?:in\s+)?thousands?\b|(?:['’]\s*000s?\b)|(?:\b000s\b)", lowered):
        return "thousands"
    return "unknown"


def normalize_value(value_verbatim: str, scale: UnitScale, kpi: str) -> float:
    value = parse_financial_number(value_verbatim) * SCALE_MULTIPLIERS[scale]
    if kpi in POSITIVE_OUTFLOW_KPIS:
        return abs(value)
    return value
