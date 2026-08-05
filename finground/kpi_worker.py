"""The single ADK task-mode worker for KPI extraction."""

from __future__ import annotations

import os
from typing import Literal

from google.adk.agents import Agent
from google.adk.workflow import RetryConfig
from pydantic import BaseModel, Field, model_validator

from finground.kpi_catalog import get_kpi_knowledge_tool
from finground.report_tools import read_report_chunks_tool, search_report_tool

MODEL = os.getenv("FINGROUND_MODEL", "deepseek/deepseek-v4-flash")
WORKER_NAME = "kpi_worker"
WORKER_MAX_ATTEMPTS = 3
WORKER_RETRY_INITIAL_DELAY = 0.1


class KpiTaskInput(BaseModel):
    """One KPI assignment created by the root agent."""

    task_id: str
    report_ref: str
    target_year: int
    kpi_key: str


class Evidence(BaseModel):
    chunk_id: str
    page: int
    statement: str
    label: str
    text: str


class KpiTaskResult(BaseModel):
    """Auditable terminal result for one KPI task."""

    task_id: str
    report_ref: str
    target_year: int
    kpi_key: str
    status: Literal["found", "explicit_zero", "absent", "ambiguous"]
    value: float | None
    unit: str | None
    source_value: str | None
    source_unit: str | None
    evidence: Evidence | None
    notes: list[str] = Field(min_length=0)

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> KpiTaskResult:
        if self.status in {"found", "explicit_zero"}:
            if self.value is None or self.evidence is None:
                raise ValueError("found and explicit_zero require value and evidence")
            if self.status == "explicit_zero" and self.value != 0:
                raise ValueError("explicit_zero requires value=0")
        elif self.value is not None:
            raise ValueError("absent and ambiguous require value=null")
        if self.status == "absent" and not self.notes:
            raise ValueError("absent requires notes describing the searched scope")
        return self


WORKER_INSTRUCTION = """You extract exactly one financial KPI from one annual report.

Required workflow:
1. Read the KpiTaskInput. Call GetKpiKnowledge with exactly its kpi_key.
2. Follow that knowledge record. Search the complete report with SearchReport. The tool scans the
   entire artifact outside your context and returns bounded candidates. Use cursor pagination only
   when the current candidates are insufficient.
3. Call ReadReportChunks only for candidate chunk_ids returned by SearchReport. Read the smallest
   amount of evidence needed to resolve the target year, consolidated scope, value, sign, and unit.
4. Return found, explicit_zero, absent, or ambiguous. Never turn missing or uncertain evidence into
   zero. Never calculate an unprinted total.
5. Call finish_task by itself with a KpiTaskResult. Copy task_id, report_ref, target_year, and
   kpi_key exactly from the input.

If a report tool says its context budget is exhausted, finish as ambiguous and record that reason.
Do not process another KPI and do not call root task-management tools.
"""


def create_kpi_worker() -> Agent:
    return Agent(
        name=WORKER_NAME,
        model=MODEL,
        mode="task",
        description="Extracts one KPI from a large annual report with bounded, auditable evidence.",
        instruction=WORKER_INSTRUCTION,
        input_schema=KpiTaskInput,
        output_schema=KpiTaskResult,
        retry_config=RetryConfig(
            max_attempts=WORKER_MAX_ATTEMPTS,
            initial_delay=WORKER_RETRY_INITIAL_DELAY,
            max_delay=WORKER_RETRY_INITIAL_DELAY,
            backoff_factor=1.0,
            jitter=0.0,
            exceptions=["JSONDecodeError"],
        ),
        tools=[get_kpi_knowledge_tool, search_report_tool, read_report_chunks_tool],
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
