"""Cash-and-equivalents KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="cash_and_equivalents",
    source_priority="the audited balance sheet, then the cash note",
    accept="period-end unrestricted cash and cash equivalents",
    reject="restricted cash, cash plus restricted cash, marketable securities, total liquidity, and cash-flow reconciliation totals",
    search_labels=("cash and cash equivalents",),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
