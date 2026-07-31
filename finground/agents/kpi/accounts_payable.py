"""Accounts-payable KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="accounts_payable",
    source_priority="the audited balance sheet, then the payables note",
    accept="the current consolidated accounts or trade payable total",
    reject="payables plus accrued expenses, related-party components, total current liabilities, and noncurrent payables",
    search_labels=("accounts payable", "trade payables"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
