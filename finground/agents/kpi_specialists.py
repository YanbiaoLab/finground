"""Compatibility facade and compact dispatcher for independent KPI agents."""

from __future__ import annotations

from typing import Any

from google.adk.tools import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from finground.agents.kpi.base import (
    KPI_AGENT_NAME_PREFIX,
    KPI_SPECIALIST_MODEL_TURN_LIMIT,
    KPI_SPECIALIST_SEARCH_LIMIT,
    kpi_agent_name,
)
from finground.agents.kpi.registry import (
    KPI_AGENT_FACTORIES,
    create_kpi_specialist_agent,
)
from finground.kpis import KPI_KEYS
from finground.sec_facts import SEC_FACTS_STATE_KEY
from finground.tools import MULTI_KPI_WORK_RECORD_STATE_KEY

MULTI_KPI_COORDINATOR_NAME = "finground_extraction_coordinator"
COMMON_TASK_AGENT_NAME = "manage_report_workflow"
KPI_DISPATCH_TOOL_NAME = "delegate_kpis"


class KpiSpecialistTool(BaseTool):
    """Dispatch canonical KPI keys to isolated module-owned agents."""

    def __init__(self, *, max_output_tokens: int) -> None:
        super().__init__(
            name=KPI_DISPATCH_TOOL_NAME,
            description=(
                "Delegate pending canonical KPIs to isolated specialists. "
                "Each specialist finds, validates, and checkpoints only its KPI."
            ),
        )
        self._specialists = {
            kpi: AgentTool(
                create_kpi_specialist_agent(kpi, max_output_tokens=max_output_tokens),
                include_plugins=True,
            )
            for kpi in KPI_KEYS
        }

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "kpis": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(KPI_KEYS)},
                        "minItems": 1,
                        "description": "Canonical KPI keys to process in isolated child sessions.",
                    },
                    "request": {
                        "type": "string",
                        "description": "Compact task or retry feedback without report text.",
                    },
                },
                "required": ["kpis", "request"],
                "additionalProperties": False,
            },
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any:
        requested = args.get("kpis", [])
        structured_values = tool_context.state.get(SEC_FACTS_STATE_KEY, {}).get("values", {})
        structured_kpis = set(structured_values) if isinstance(structured_values, dict) else set()
        completed = [kpi for kpi in requested if kpi in structured_kpis]
        failed: list[dict[str, Any]] = []
        for kpi in requested:
            if kpi in structured_kpis:
                continue
            specialist = self._specialists.get(kpi)
            if specialist is None:
                failed.append({"kpi": kpi, "error": f"unknown KPI specialist: {kpi}"})
                continue
            result = await specialist.run_async(
                args={"request": f"{args['request']} Work only on {kpi}."},
                tool_context=tool_context,
            )
            work_record = tool_context.state.get(MULTI_KPI_WORK_RECORD_STATE_KEY, {})
            rows = work_record.get("kpis", []) if isinstance(work_record, dict) else []
            if any(isinstance(row, dict) and row.get("kpi") == kpi for row in rows):
                completed.append(kpi)
            else:
                failed.append(
                    {
                        "kpi": kpi,
                        "error": (
                            result.get("error", result.get("status", "incomplete"))
                            if isinstance(result, dict)
                            else "incomplete"
                        ),
                    }
                )
        return {
            "status": "success" if not failed else "partial_success",
            "requested_count": len(requested),
            "completed_count": len(completed),
            "completed_kpis": completed,
            "skipped_structured_kpis": [kpi for kpi in requested if kpi in structured_kpis],
            "failed": failed,
            "next_action": "request coverage audit",
        }


__all__ = [
    "COMMON_TASK_AGENT_NAME",
    "KPI_AGENT_FACTORIES",
    "KPI_AGENT_NAME_PREFIX",
    "KPI_DISPATCH_TOOL_NAME",
    "KPI_SPECIALIST_MODEL_TURN_LIMIT",
    "KPI_SPECIALIST_SEARCH_LIMIT",
    "MULTI_KPI_COORDINATOR_NAME",
    "KpiSpecialistTool",
    "create_kpi_specialist_agent",
    "kpi_agent_name",
]
