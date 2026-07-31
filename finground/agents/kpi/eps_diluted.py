"""Diluted-EPS KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="eps_diluted",
    source_priority="the audited consolidated income statement and EPS note",
    accept="diluted earnings per common or ordinary share for the target fiscal year",
    reject="basic EPS, adjusted EPS, weighted-average shares, dividends per share, and cents not converted as subunits",
    search_labels=("diluted earnings per share", "diluted EPS"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
