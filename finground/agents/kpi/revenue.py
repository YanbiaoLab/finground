"""Revenue KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="revenue",
    source_priority="the audited consolidated income or operations statement",
    accept="the consolidated top-line total labelled revenue, revenues, net revenue, or net sales",
    reject="segment revenue, adjusted revenue, other income, gains, and subtotals below the top line",
    search_labels=("total revenue", "net revenue", "net sales", "revenues"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
