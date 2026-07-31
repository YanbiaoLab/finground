"""Parent-only stockholders-equity KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="stockholders_equity",
    source_priority="the audited balance sheet, then the statement of changes in equity",
    accept="total equity attributable to parent shareholders or owners, excluding NCI",
    reject="total equity including NCI, liabilities and equity combined, equity ratios, and per-share book value",
    search_labels=(
        "total stockholders equity",
        "shareholders equity",
        "equity attributable to owners",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
