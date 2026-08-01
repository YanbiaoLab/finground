"""Structural contract for independently optimized KPI agents."""

from pathlib import Path

import pytest

from finground.agents.kpi.registry import KPI_AGENT_FACTORIES
from finground.kpis import KPI_KEYS


def test_every_kpi_has_one_independent_agent_file() -> None:
    agent_dir = Path(__file__).resolve().parents[1] / "finground" / "agents" / "kpi"
    agent_files = {
        path.stem
        for path in agent_dir.glob("*.py")
        if path.stem not in {"__init__", "base", "registry"}
    }

    assert agent_files == set(KPI_KEYS)
    assert tuple(KPI_AGENT_FACTORIES) == KPI_KEYS


@pytest.mark.parametrize("kpi", KPI_KEYS)
def test_each_kpi_module_builds_a_scoped_agent(kpi: str) -> None:
    agent = KPI_AGENT_FACTORIES[kpi](max_output_tokens=4_096)
    tool_names = [tool.name if hasattr(tool, "name") else tool.__name__ for tool in agent.tools]

    assert agent.name == f"extract_{kpi}"
    assert agent.include_contents == "none"
    assert tool_names == [
        f"find_{kpi}_candidates",
        "search_report",
        "read_report_pages",
        "record_multi_kpi_progress",
    ]
    assert f"exactly one canonical KPI: {kpi}" in agent.instruction
    assert "Do not find, judge, or record any other KPI." in agent.instruction


@pytest.mark.parametrize("kpi", KPI_KEYS)
def test_each_kpi_owns_a_complete_instruction_in_its_module(kpi: str) -> None:
    module = __import__(f"finground.agents.kpi.{kpi}", fromlist=["INSTRUCTION"])

    assert not hasattr(module, "SPEC")
    assert KPI_AGENT_FACTORIES[kpi](max_output_tokens=4_096).instruction == module.INSTRUCTION
    assert "EVIDENCE DECISION" in module.INSTRUCTION
    assert "NORMALIZATION" in module.INSTRUCTION


def test_revenue_agent_owns_mortgage_reit_ledger_fallback() -> None:
    from finground.agents.kpi import revenue

    assert "Mortgage REITs and investment companies" in revenue.INSTRUCTION
    assert "Net portfolio income" in revenue.INSTRUCTION
    assert "Reject Net interest income by itself" in revenue.INSTRUCTION
