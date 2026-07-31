"""Inventory KPI agent."""

from finground.agents.kpi.base import KpiAgentSpec, build_kpi_agent

SPEC = KpiAgentSpec(
    kpi="inventory",
    source_priority="the audited balance sheet, then the inventory note",
    accept="the period-end consolidated inventory net total",
    reject="raw-material or finished-goods components, inventory provisions, cost of sales, and noncurrent assets",
    search_labels=("inventory net", "inventories", "total inventory"),
)


def create_agent(*, max_output_tokens: int):
    return build_kpi_agent(SPEC, max_output_tokens=max_output_tokens)
