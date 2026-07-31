"""Depreciation-and-amortization KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="depreciation_amortization",
    source_priority="the audited cash-flow statement, then its operating reconciliation",
    accept="a combined depreciation/depletion/amortization addback; if absent, LEDGER's tag waterfall accepts the depreciation addback",
    reject="accumulated depreciation, capex, EBITDA, and note-only asset schedules",
    search_labels=(
        "depreciation and amortization",
        "depreciation and amortisation",
        "depreciation depletion and amortization",
        "depreciation",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
