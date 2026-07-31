"""Cost-of-revenue KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="cost_of_revenue",
    source_priority="the audited consolidated income statement and its cost-of-sales note",
    accept="a printed total cost of revenue, cost of sales, or cost of goods and services sold",
    reject="SG&A, R&D, operating expenses, individual cost components, and any derived total",
    search_labels=("cost of revenue", "cost of sales", "cost of goods sold"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
