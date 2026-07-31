"""Capital-expenditure KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="capex",
    source_priority="the audited consolidated statement of cash flows",
    accept="a printed cash payment or purchase row for property, plant and equipment",
    reject="PP&E-note additions, acquisitions of businesses, asset balances, depreciation, total investing cash flow, and derived sums",
    search_labels=(
        "purchases of property plant and equipment",
        "payments for property plant and equipment",
        "capital expenditures",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
