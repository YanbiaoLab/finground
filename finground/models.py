"""Stable data contracts used by FinGround agents and benchmark runners."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UnitScale = Literal[
    "units",
    "thousands",
    "millions",
    "billions",
    "per_share",
    "currency_subunits_per_share",
    "unknown",
]
MultiKpiRecordView = Literal["all", "kpis", "notes"]
MultiKpiNoteCategory = Literal["evidence", "unit", "scope", "decision", "todo", "warning"]
MultiKpiEvidenceStatus = Literal["found", "explicit_zero", "absent", "ambiguous"]
MultiKpiSignRule = Literal["as_reported", "positive_outflow", "explicit_zero"]

KpiKey = Literal[
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
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "stockholders_equity_incl_nci",
    "cash_and_equivalents",
    "cash_incl_restricted",
    "long_term_debt_total",
    "long_term_debt_noncurrent",
    "long_term_debt_current",
    "short_term_borrowings",
    "inventory",
    "accounts_receivable",
    "accounts_payable",
    "shares_outstanding",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
    "depreciation_amortization",
    "dividends_paid",
]


class ExtractedKPI(BaseModel):
    """One LEDGER KPI value in raw single units for one fiscal year."""

    model_config = ConfigDict(extra="forbid")

    kpi: KpiKey
    fiscal_year: int = Field(ge=1990, le=2100)
    value: float | None = None


class ReportExtraction(BaseModel):
    """LEDGER-compatible multi-KPI extraction for one annual report."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    reporting_currency: str | None = None
    units_note: str | None = None
    kpis: list[ExtractedKPI] = Field(default_factory=list)


class MultiKpiNote(BaseModel):
    """One durable report-analysis note retained across context reduction."""

    model_config = ConfigDict(extra="forbid")

    category: MultiKpiNoteCategory
    text: str = Field(min_length=1, max_length=1_500)
    pages: list[int] = Field(default_factory=list, max_length=12)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("note text must not be blank")
        return text

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, pages: list[int]) -> list[int]:
        if any(page < 1 for page in pages):
            raise ValueError("note pages must be positive report page numbers")
        return list(dict.fromkeys(pages))


class MultiKpiEvidenceCandidate(BaseModel):
    """Agent-selected source evidence for one KPI/year coverage decision."""

    model_config = ConfigDict(extra="forbid")

    kpi: KpiKey = Field(description="Canonical Ledger KPI key.")
    fiscal_year: int = Field(
        ge=1990,
        le=2100,
        description="Fiscal-year label of the selected statement column.",
    )
    status: MultiKpiEvidenceStatus = Field(
        description=(
            "found for a printed number; explicit_zero for a dash/nil in the matching cell; "
            "absent when no matching row exists; ambiguous when competing evidence is unresolved."
        )
    )
    value_verbatim: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Exact unscaled numeric token or explicit-zero marker copied from the report.",
    )
    unit_scale: UnitScale | None = Field(
        default=None,
        description="Observed scale category; exact unit_text takes precedence when supplied.",
    )
    unit_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="Exact visible statement unit/scale text copied from the report.",
    )
    unit_page: int | None = Field(
        default=None,
        ge=1,
        description="Report page containing unit_text; required whenever unit_text is supplied.",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="Report page containing the selected labelled row and source token.",
    )
    statement: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
        description="Primary financial-statement title for the selected row.",
    )
    line_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="Exact visible row label associated with value_verbatim.",
    )
    year_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Exact visible column label used to determine fiscal_year.",
    )
    scope: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="Consolidation and parent/NCI or other KPI-specific scope decision.",
    )

    @field_validator(
        "value_verbatim",
        "unit_text",
        "statement",
        "line_label",
        "year_label",
        "scope",
    )
    @classmethod
    def strip_evidence_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("evidence text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> MultiKpiEvidenceCandidate:
        if self.status in {"found", "explicit_zero"}:
            required = {
                "value_verbatim": self.value_verbatim,
                "page": self.page,
                "statement": self.statement,
                "line_label": self.line_label,
                "year_label": self.year_label,
                "scope": self.scope,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"{self.status} evidence requires: {', '.join(missing)}"
                )
            if self.unit_scale in {None, "unknown"} and self.unit_text is None:
                raise ValueError(
                    f"{self.status} evidence requires unit_scale or exact unit_text"
                )
            if self.unit_text is not None and self.unit_page is None:
                raise ValueError("unit_text requires unit_page")
        elif self.status == "absent":
            evidence_fields = {
                "value_verbatim": self.value_verbatim,
                "unit_scale": self.unit_scale,
                "unit_text": self.unit_text,
                "unit_page": self.unit_page,
                "page": self.page,
                "statement": self.statement,
                "line_label": self.line_label,
                "year_label": self.year_label,
                "scope": self.scope,
            }
            present = [name for name, value in evidence_fields.items() if value is not None]
            if present:
                raise ValueError(
                    "absent coverage must not claim source evidence: " + ", ".join(present)
                )
        return self


class MultiKpiNormalization(BaseModel):
    """Deterministic transformation from report token to LEDGER raw units."""

    model_config = ConfigDict(extra="forbid")

    parsed_number: float
    multiplier: float
    sign_rule: MultiKpiSignRule
    formula: str = Field(min_length=1, max_length=200)


class MultiKpiEvidence(MultiKpiEvidenceCandidate):
    """Validated evidence plus a tool-computed normalized value."""

    value: float | None = None
    normalization: MultiKpiNormalization | None = None


class MultiKpiWorkRecord(BaseModel):
    """Validated evidence, coverage decisions, and notes for one report invocation."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    reporting_currency: str | None = None
    units_note: str | None = None
    kpis: list[MultiKpiEvidence] = Field(default_factory=list)

    notes: list[MultiKpiNote] = Field(default_factory=list, max_length=200)


class NeedleAnswer(BaseModel):
    """LEDGER-compatible answer for one company/year/KPI query."""

    found: bool
    value: float | None = None
    value_verbatim: str | None = None
    unit_scale: UnitScale | None = None
    page: int | None = Field(default=None, ge=1)

    @field_validator("unit_scale", mode="before")
    @classmethod
    def normalize_unit_scale(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return {
                1: "units",
                1_000: "thousands",
                1_000_000: "millions",
                1_000_000_000: "billions",
            }.get(value, "unknown")
        normalized = str(value).strip().lower().replace("-", "_")
        for token, scale in (
            ("currency_subunits_per_share", "currency_subunits_per_share"),
            ("cents per share", "currency_subunits_per_share"),
            ("pence per share", "currency_subunits_per_share"),
            ("per_share", "per_share"),
            ("per share", "per_share"),
            ("billion", "billions"),
            ("million", "millions"),
            ("thousand", "thousands"),
            ("unit", "units"),
        ):
            if token in normalized:
                return scale
        return "unknown"

    @field_validator("page", mode="before")
    @classmethod
    def normalize_page(cls, value: object) -> object:
        if value is None or isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else value

    @model_validator(mode="after")
    def validate_presence(self) -> NeedleAnswer:
        if self.found and (self.value is None or self.value_verbatim is None):
            raise ValueError("found answers require value and value_verbatim")
        if not self.found and self.value is not None:
            raise ValueError("not-found answers must have a null value")
        return self
