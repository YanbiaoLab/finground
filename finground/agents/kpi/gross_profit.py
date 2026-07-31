"""Gross-profit KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="gross_profit",
    source_priority="the audited consolidated income statement",
    accept="a directly printed consolidated gross profit or gross margin amount",
    reject="gross-margin percentages, segment gross profit, adjusted measures, and revenue-minus-cost calculations",
    search_labels=("gross profit", "gross margin"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
