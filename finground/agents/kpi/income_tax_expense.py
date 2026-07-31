"""Income-tax-expense KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="income_tax_expense",
    source_priority="the consolidated income statement, then the income-tax note",
    accept="the total income-tax provision, expense, or benefit for continuing consolidated operations",
    reject="current/deferred components, jurisdiction components, effective tax rates, pretax income, and cash taxes paid",
    search_labels=("income tax expense", "income tax benefit", "provision for income taxes"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
