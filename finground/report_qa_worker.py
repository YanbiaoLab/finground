"""Task-mode worker for evidence-grounded questions about an annual report."""

from __future__ import annotations

from typing import Literal

from google.adk.agents import Agent
from pydantic import BaseModel, Field, model_validator

from finground.config import create_agent_model
from finground.kpi_worker import create_tool_call_retry_config
from finground.report_tools import (
    prepare_report_question_tool,
    read_report_chunks_tool,
    search_report_tool,
)

REPORT_QA_WORKER_NAME = "report_qa_worker"


class ReportQuestionInput(BaseModel):
    """One non-KPI question about the current annual report."""

    task_id: str
    report_ref: str
    question: str = Field(min_length=1, max_length=4_000)


class ReportQuestionEvidence(BaseModel):
    chunk_id: str
    page: int
    heading: str
    text: str = Field(min_length=1, max_length=2_000)


class ReportQuestionResult(BaseModel):
    """Evidence-grounded terminal result for one annual-report question."""

    task_id: str
    report_ref: str
    question: str = Field(min_length=1, max_length=4_000)
    status: Literal["answered", "not_found", "ambiguous"]
    answer: str | None = Field(max_length=12_000)
    evidence: list[ReportQuestionEvidence] = Field(default_factory=list, max_length=5)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answer_contract(self) -> ReportQuestionResult:
        if self.status == "answered":
            if not self.answer or not self.answer.strip() or not self.evidence:
                raise ValueError("answered requires a non-empty answer and evidence")
        else:
            if self.answer is not None:
                raise ValueError("not_found and ambiguous require answer=null")
            if not self.notes:
                raise ValueError("not_found and ambiguous require explanatory notes")
        return self


REPORT_QA_INSTRUCTION = """You answer exactly one question using the current annual report.

Required workflow:
1. Read the ReportQuestionInput. Call PrepareReportQuestion with exactly its report_ref and question.
2. Search the complete report with SearchReport. Use several precise queries when needed, including
   terminology likely used by the company. The tool scans the artifact outside your context.
3. Call ReadReportChunks only for candidate chunk_ids returned by SearchReport. Read the smallest
   amount of evidence needed for a complete and accurate answer.
4. Return answered only when the answer is directly supported by at least one quoted evidence
   record. Return not_found after a reasonable search finds no answer. Return ambiguous when
   relevant evidence conflicts, lacks scope, or the report-tool context budget is exhausted.
5. Call finish_task by itself with a ReportQuestionResult. Copy task_id, report_ref, and question
   exactly from the input.

Do not use outside knowledge, browse the web, infer undisclosed facts, or perform canonical KPI
extraction and normalization. A canonical KPI request belongs to kpi_worker; finish as ambiguous
and explain the routing issue if such a request reaches you. Do not call root task-management tools.
"""


def create_report_qa_worker() -> Agent:
    return Agent(
        name=REPORT_QA_WORKER_NAME,
        model=create_agent_model(),
        mode="task",
        description="Answers one non-KPI question from an annual report with bounded evidence.",
        instruction=REPORT_QA_INSTRUCTION,
        input_schema=ReportQuestionInput,
        output_schema=ReportQuestionResult,
        retry_config=create_tool_call_retry_config(),
        tools=[
            prepare_report_question_tool,
            search_report_tool,
            read_report_chunks_tool,
        ],
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
