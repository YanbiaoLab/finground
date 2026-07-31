"""Total-long-term-debt KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="long_term_debt_total",
    source_priority="the debt note, then the audited balance sheet",
    accept="a printed total long-term debt amount including the current portion; if no such total is printed, LEDGER's historical fallback accepts the long-term debt balance net of current maturities",
    reject="current maturities alone, short-term borrowings, interest rates, future maturity schedules, and calculated sums",
    search_labels=(
        "total long term debt",
        "long-term debt including current portion",
        "total debt",
    ),
    extra_instruction="""LEDGER FALLBACK:
- If no inclusive total is printed, accept a target-year row explicitly labelled long-term debt
  net of current maturities or less current portion. Do not calculate an inclusive amount.""",
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
