"""Cash-including-restricted-cash KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="cash_incl_restricted",
    source_priority="the cash-flow reconciliation or cash and restricted-cash note",
    accept="a printed combined total of cash, cash equivalents, and restricted cash",
    reject="unrestricted cash alone, restricted cash alone, marketable securities, and calculated sums",
    search_labels=(
        "cash cash equivalents and restricted cash",
        "cash, cash equivalents, and restricted cash",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
