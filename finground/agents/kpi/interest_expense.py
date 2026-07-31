"""Interest-expense KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="interest_expense",
    source_priority="the consolidated income statement, then the debt or finance-cost note",
    accept="gross interest expense or finance cost attributable to debt",
    reject="interest income, net interest unless expense is separately identifiable, rates, capitalized interest, and debt balances",
    search_labels=("interest expense", "finance costs", "finance expense"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
