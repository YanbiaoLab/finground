"""Current-long-term-debt KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="long_term_debt_current",
    source_priority="the audited balance sheet, then the debt note",
    accept="the current portion or current maturities of long-term debt",
    reject="all current liabilities, total debt, noncurrent debt, short-term borrowings, and interest rates",
    search_labels=("current portion of long term debt", "current maturities of long-term debt"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
