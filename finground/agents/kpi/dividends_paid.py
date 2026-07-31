"""Dividends-paid KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="dividends_paid",
    source_priority="the audited cash-flow statement, then the equity or dividend note",
    accept="cash dividends paid to common shareholders during the fiscal year",
    reject="dividends declared, dividends per share, preferred-only dividends, noncash distributions, and financing cash-flow totals",
    search_labels=("dividends paid", "payment of dividends", "cash dividends paid"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
