"""Noncurrent-long-term-debt KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="long_term_debt_noncurrent",
    source_priority="the audited balance sheet, then the debt note",
    accept="the noncurrent portion of long-term debt at period end",
    reject="total debt including current maturities, current debt, short-term borrowings, lease liabilities, and rates",
    search_labels=("long term debt", "noncurrent debt", "long-term borrowings"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
