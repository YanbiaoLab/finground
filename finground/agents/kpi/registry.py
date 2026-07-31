"""Canonical registry for the 31 independently editable KPI agents."""

from __future__ import annotations

from collections.abc import Callable

from google.adk.agents import Agent

from finground.agents.kpi import (
    accounts_payable,
    accounts_receivable,
    capex,
    cash_and_equivalents,
    cash_incl_restricted,
    cost_of_revenue,
    depreciation_amortization,
    dividends_paid,
    eps_basic,
    eps_diluted,
    financing_cash_flow,
    gross_profit,
    income_tax_expense,
    interest_expense,
    inventory,
    investing_cash_flow,
    long_term_debt_current,
    long_term_debt_noncurrent,
    long_term_debt_total,
    net_income,
    operating_cash_flow,
    operating_income,
    rd_expense,
    revenue,
    sga_expense,
    shares_outstanding,
    short_term_borrowings,
    stockholders_equity,
    stockholders_equity_incl_nci,
    total_assets,
    total_liabilities,
)
from finground.agents.kpi.base import KpiAgentSpec
from finground.kpis import KPI_KEYS

AgentFactory = Callable[..., Agent]

_MODULES = (
    revenue,
    cost_of_revenue,
    gross_profit,
    rd_expense,
    sga_expense,
    operating_income,
    interest_expense,
    income_tax_expense,
    net_income,
    eps_basic,
    eps_diluted,
    total_assets,
    total_liabilities,
    stockholders_equity,
    stockholders_equity_incl_nci,
    cash_and_equivalents,
    cash_incl_restricted,
    long_term_debt_total,
    long_term_debt_noncurrent,
    long_term_debt_current,
    short_term_borrowings,
    inventory,
    accounts_receivable,
    accounts_payable,
    shares_outstanding,
    operating_cash_flow,
    investing_cash_flow,
    financing_cash_flow,
    capex,
    depreciation_amortization,
    dividends_paid,
)

KPI_AGENT_SPECS: dict[str, KpiAgentSpec] = {module.SPEC.kpi: module.SPEC for module in _MODULES}
KPI_AGENT_FACTORIES: dict[str, AgentFactory] = {
    module.SPEC.kpi: module.create_agent for module in _MODULES
}

if tuple(KPI_AGENT_SPECS) != KPI_KEYS or tuple(KPI_AGENT_FACTORIES) != KPI_KEYS:
    raise RuntimeError("KPI agent registry must contain exactly one module per canonical KPI")


def create_kpi_specialist_agent(kpi: str, *, max_output_tokens: int) -> Agent:
    """Create one specialist through its independently owned module."""
    try:
        factory = KPI_AGENT_FACTORIES[kpi]
    except KeyError as error:
        raise ValueError(f"unknown KPI specialist: {kpi}") from error
    return factory(max_output_tokens=max_output_tokens)
