"""Net-income KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="net_income",
    source_priority="the audited consolidated income statement",
    accept="net income or profit attributable to the parent or common shareholders",
    reject="income including non-controlling interests when a parent-only row exists, comprehensive income, and segment profit",
    search_labels=("net income attributable", "profit attributable to owners", "net income"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
