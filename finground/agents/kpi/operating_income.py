"""Operating-income KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="operating_income",
    source_priority="the audited consolidated income or operations statement",
    accept="operating income, operating profit, or income from operations for the consolidated company",
    reject="net income, pretax income, EBITDA, adjusted operating income, and segment profit",
    search_labels=("operating income", "operating profit", "income from operations"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
