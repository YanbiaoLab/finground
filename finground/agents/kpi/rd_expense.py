"""Research-and-development expense KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="rd_expense",
    source_priority="the consolidated income statement, then the R&D or operating-expense note",
    accept="a company-wide research and development expense amount",
    reject="SG&A, engineering headcount, capitalized development, D&A, and a combined operating-expense total",
    search_labels=("research and development expense", "research and development", "R&D"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
