"""Deterministic value parsing, scale detection, and evidence validation."""

from __future__ import annotations

import re
from math import isclose
from typing import TypedDict

from finground.kpis import PER_SHARE_KPIS, POSITIVE_OUTFLOW_KPIS
from finground.models import NeedleAnswer, UnitScale

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


class NormalizationTrace(TypedDict, total=False):
    status: str
    reason: str
    original_value: float | None
    computed_value: float
    scale: UnitScale
    scale_source: str


class EvidenceValidationTrace(TypedDict, total=False):
    status: str
    reason: str


def _abstain(answer: NeedleAnswer) -> NeedleAnswer:
    return answer.model_copy(
        update={
            "found": False,
            "value": None,
            "value_verbatim": None,
            "unit_scale": None,
            "page": None,
        }
    )


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


def _explicit_scale(text: str, kpi: str) -> UnitScale | None:
    scale = detect_scale(text, kpi)
    if kpi in PER_SHARE_KPIS and scale == "per_share":
        return None
    return None if scale == "unknown" else scale


def normalize_needle_answer(
    answer: NeedleAnswer, kpi: str
) -> tuple[NeedleAnswer, NormalizationTrace]:
    """Recompute a model answer only when its source number is unambiguous.

    The model selects evidence and its scale; this function makes the final
    arithmetic deterministic. It never consults benchmark ground truth.
    """
    trace: NormalizationTrace = {
        "status": "unverified",
        "original_value": answer.value,
    }
    if not answer.found:
        trace["reason"] = "not_found"
        return answer, trace
    if answer.value_verbatim is None or answer.value is None:
        trace["reason"] = "missing_value_fields"
        return answer, trace

    number_tokens = NUMBER_TOKEN_RE.findall(answer.value_verbatim)
    if len(number_tokens) != 1:
        trace["reason"] = "value_verbatim_not_single_number"
        return _abstain(answer), trace

    explicit_scale = _explicit_scale(answer.value_verbatim, kpi)
    if explicit_scale is not None:
        scale = explicit_scale
        scale_source = "value_verbatim"
    elif kpi in PER_SHARE_KPIS:
        if answer.unit_scale in {"per_share", "currency_subunits_per_share"}:
            scale = answer.unit_scale
            scale_source = "model"
        else:
            scale = "per_share"
            scale_source = "kpi"
    elif answer.unit_scale not in {None, "unknown"}:
        scale = answer.unit_scale
        scale_source = "model"
    else:
        trace["reason"] = "unresolved_scale"
        return answer, trace

    try:
        computed = normalize_value(number_tokens[0], scale, kpi)
    except ValueError:
        trace["reason"] = "invalid_source_number"
        return answer, trace

    corrected = not isclose(answer.value, computed, rel_tol=1e-12, abs_tol=1e-12)
    trace.update(
        {
            "status": "corrected" if corrected else "verified",
            "computed_value": computed,
            "scale": scale,
            "scale_source": scale_source,
        }
    )
    return answer.model_copy(update={"value": computed, "unit_scale": scale}), trace


def validate_needle_evidence(
    answer: NeedleAnswer, cited_page_text: str | None
) -> tuple[NeedleAnswer, EvidenceValidationTrace]:
    """Require the answer's verbatim numeric token to exist on its cited candidate page."""
    if not answer.found:
        return answer, {"status": "skip", "reason": "not_found"}
    if cited_page_text is None:
        return _abstain(answer), {"status": "fail", "reason": "cited_page_not_selected"}
    answer_tokens = NUMBER_TOKEN_RE.findall(answer.value_verbatim or "")
    if len(answer_tokens) != 1:
        return _abstain(answer), {"status": "fail", "reason": "invalid_verbatim_number"}
    try:
        answer_number = parse_financial_number(answer_tokens[0])
    except ValueError:
        return _abstain(answer), {"status": "fail", "reason": "invalid_verbatim_number"}

    for token in NUMBER_TOKEN_RE.findall(cited_page_text):
        try:
            page_number = parse_financial_number(token)
        except ValueError:
            continue
        if isclose(abs(page_number), abs(answer_number), rel_tol=1e-12, abs_tol=1e-12):
            return answer, {"status": "pass"}
    return _abstain(answer), {
        "status": "fail",
        "reason": "verbatim_number_not_on_cited_page",
    }
