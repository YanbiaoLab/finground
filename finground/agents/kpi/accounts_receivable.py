"""Accounts-receivable KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="accounts_receivable",
    source_priority="the audited balance sheet, then the receivables note",
    accept="current accounts or trade receivable net of allowance as a consolidated total",
    reject="related-party components, other receivables, contract assets, gross-before-allowance amounts, and noncurrent receivables",
    search_labels=("accounts receivable net", "trade receivables", "receivables net"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
