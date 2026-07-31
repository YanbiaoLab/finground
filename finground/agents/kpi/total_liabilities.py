"""Total-liabilities KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="total_liabilities",
    source_priority="the audited consolidated balance sheet or statement of financial position",
    accept="the period-end consolidated total liabilities row by itself",
    reject="current liabilities, liabilities and equity combined, liabilities plus redeemable interests, and segment liabilities",
    search_labels=("total liabilities",),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
