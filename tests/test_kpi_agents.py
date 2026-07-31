"""Structural contract for independently optimized KPI agents."""

from pathlib import Path

import pytest

from finground.agents.kpi.registry import KPI_AGENT_FACTORIES, KPI_AGENT_SPECS
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
    assert tuple(KPI_AGENT_SPECS) == KPI_KEYS


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
