"""Operating-cash-flow KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="operating_cash_flow",
    source_priority="the audited consolidated statement of cash flows",
    accept="the net cash provided by or used in operating activities subtotal",
    reject="individual operating adjustments, EBITDA, free cash flow, cash balance changes, and investing or financing subtotals",
    search_labels=(
        "net cash provided by operating activities",
        "net cash used in operating activities",
        "net cash from operating activities",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
