"""Shares-outstanding KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="shares_outstanding",
    source_priority="the balance sheet, report cover, then the equity or share-capital note",
    accept="common or ordinary shares actually outstanding at the fiscal period end",
    reject="authorized or issued-only shares, weighted-average EPS shares, treasury shares, currency amounts, and post-year-end cover dates",
    search_labels=(
        "common shares outstanding",
        "ordinary shares outstanding",
        "shares issued and outstanding",
    ),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
