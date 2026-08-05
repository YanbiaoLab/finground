"""Workflow-backed dispatch for a general annual-report question."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from google.adk import Context
from google.adk.agents import Agent
from google.adk.workflow import Workflow, node
from google.adk.workflow._errors import DynamicNodeFailError
from pydantic import BaseModel, ValidationError, model_validator

from finground.kpi_dispatcher import WorkerResultError, root_error_message
from finground.report_qa_worker import (
    ReportQuestionInput,
    ReportQuestionResult,
    create_report_qa_worker,
)

REPORT_QA_DISPATCHER_NAME = "answer_report_question"
REPORT_QA_NODE_NAME = "answer_report_question_item"


class ReportQuestionOutcome(BaseModel):
    """Dispatcher outcome for one annual-report question."""

    task_id: str
    status: Literal["succeeded", "failed"]
    result: ReportQuestionResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ReportQuestionOutcome:
        if self.status == "succeeded" and (self.result is None or self.error is not None):
            raise ValueError("succeeded outcome requires result and forbids error")
        if self.status == "failed" and (self.result is not None or not self.error):
            raise ValueError("failed outcome requires error and forbids result")
        return self


def _normalized_result(result: Any) -> ReportQuestionResult:
    if result is None:
        raise WorkerResultError("report_qa_worker returned no task result")
    if not isinstance(result, Mapping):
        raise WorkerResultError(
            f"report_qa_worker returned {type(result).__name__}, expected an object"
        )
    return ReportQuestionResult.model_validate(result)


def create_report_qa_dispatcher(worker: Agent | None = None) -> Workflow:
    """Create a workflow tool that runs one question in an isolated task worker."""
    task_worker = worker or create_report_qa_worker()

    async def run_question(
        node_input: ReportQuestionInput,
        ctx: Context,
    ) -> dict[str, Any]:
        isolation_scope = (
            f"{ctx.isolation_scope or REPORT_QA_DISPATCHER_NAME}/{node_input.task_id}"
        )
        try:
            raw_result = await ctx.run_node(
                task_worker,
                node_input=node_input,
                use_sub_branch=True,
                override_isolation_scope=isolation_scope,
            )
        except DynamicNodeFailError as error:
            return ReportQuestionOutcome(
                task_id=node_input.task_id,
                status="failed",
                error=root_error_message(error),
            ).model_dump()
        try:
            result = _normalized_result(raw_result)
        except (WorkerResultError, ValidationError) as error:
            return ReportQuestionOutcome(
                task_id=node_input.task_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            ).model_dump()
        return ReportQuestionOutcome(
            task_id=node_input.task_id,
            status="succeeded",
            result=result,
        ).model_dump()

    question_node = node(
        run_question,
        name=REPORT_QA_NODE_NAME,
        rerun_on_resume=True,
    )
    return Workflow(
        name=REPORT_QA_DISPATCHER_NAME,
        description="Answer one non-KPI question using evidence from the current annual report.",
        input_schema=ReportQuestionInput,
        # run_question validates the typed outcome before returning it. Expose only the outer JSON
        # shape to the coordinator model to avoid duplicating the full result schema in its tools.
        output_schema=dict[str, Any],
        edges=[("START", question_node)],
    )
