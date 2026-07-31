"""Financing-cash-flow KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="financing_cash_flow",
    source_priority="the audited consolidated statement of cash flows",
    accept="the net cash provided by or used in financing activities subtotal",
    reject="individual debt or equity transactions, dividends alone, and operating or investing subtotals",
    search_labels=(
        "net cash provided by financing activities",
        "net cash used in financing activities",
        "net cash from financing activities",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
