"""Total-assets KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="total_assets",
    source_priority="the audited consolidated balance sheet or statement of financial position",
    accept="the period-end consolidated total assets row",
    reject="current assets, noncurrent assets, segment assets, average assets, and liabilities-and-equity totals used as a proxy",
    search_labels=("total assets",),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
