"""Selling, general and administrative expense KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="sga_expense",
    source_priority="the consolidated income statement, then the operating-expense note",
    accept="a printed combined selling, general and administrative expense total; if absent, LEDGER's fallback is the general and administrative expense row",
    reject="standalone selling without G&A, R&D, and total operating expenses",
    search_labels=(
        "selling general and administrative",
        "selling, general and administrative",
        "SG&A",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
