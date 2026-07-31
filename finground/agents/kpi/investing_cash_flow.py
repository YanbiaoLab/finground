"""Investing-cash-flow KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="investing_cash_flow",
    source_priority="the audited consolidated statement of cash flows",
    accept="the net cash provided by or used in investing activities subtotal",
    reject="individual asset purchases or proceeds, capex alone, and operating or financing subtotals",
    search_labels=(
        "net cash provided by investing activities",
        "net cash used in investing activities",
        "net cash from investing activities",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
