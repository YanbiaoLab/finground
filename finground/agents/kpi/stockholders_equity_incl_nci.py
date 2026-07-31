"""Stockholders-equity-including-NCI KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="stockholders_equity_incl_nci",
    source_priority="the audited balance sheet, then the statement of changes in equity",
    accept="total equity explicitly including non-controlling or minority interests",
    reject="parent-only equity, liabilities and equity combined, equity ratios, and individual equity components",
    search_labels=(
        "total equity",
        "equity including noncontrolling interest",
        "total shareholders equity and noncontrolling interest",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
