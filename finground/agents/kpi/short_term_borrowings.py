"""Short-term-borrowings KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="short_term_borrowings",
    source_priority="the audited balance sheet, then the short-term financing or debt note",
    accept="short-term borrowings or debt due within one year; LEDGER also accepts a printed combined current-debt row that includes current maturities when no standalone short-term-borrowing row exists",
    reject="accounts payable, total current liabilities, noncurrent debt, interest rates, unused facilities, and future maturity schedules",
    search_labels=(
        "short term borrowings",
        "short-term debt",
        "bank loans",
        "current maturities of notes payable",
        "current portion of long-term debt",
        "commercial paper",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
