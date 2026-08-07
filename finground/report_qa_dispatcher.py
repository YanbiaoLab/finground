"""Workflow-backed dispatch for a general annual-report question."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from google.adk import Context
from google.adk.agents import Agent
from google.adk.workflow import Workflow, node
from google.adk.workflow._errors import DynamicNodeFailError
from pydantic import BaseModel, ValidationError, model_validator

from finground.kpi_dispatcher import WorkerResultError
from finground.kpi_worker import TOOL_CALL_MAX_ATTEMPTS
from finground.report_qa_worker import (
    ReportQuestionInput,
    ReportQuestionResult,
    create_report_qa_worker,
)

REPORT_QA_DISPATCHER_NAME = "answer_report_question"
REPORT_QA_NODE_NAME = "answer_report_question_item"
logger = logging.getLogger(__name__)

ReportQuestionErrorCode = Literal[
    "worker_invalid_json",
    "worker_invalid_result",
    "worker_execution_failed",
]


class ReportQuestionOutcome(BaseModel):
    """Dispatcher outcome for one annual-report question."""

    task_id: str
    status: Literal["succeeded", "failed"]
    result: ReportQuestionResult | None = None
    error: str | None = None
    error_code: ReportQuestionErrorCode | None = None
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ReportQuestionOutcome:
        failure_fields = (self.error, self.error_code, self.retryable)
        if self.status == "succeeded" and (
            self.result is None or any(value is not None for value in failure_fields)
        ):
            raise ValueError("succeeded outcome requires only result")
        if self.status == "failed" and (
            self.result is not None
            or not self.error
            or self.error_code is None
            or self.retryable is None
        ):
            raise ValueError("failed outcome requires structured failure guidance")
        return self


def _root_error(error: DynamicNodeFailError) -> Exception:
    root_error: Exception = error
    seen: set[int] = set()
    while isinstance(root_error, DynamicNodeFailError) and id(root_error) not in seen:
        seen.add(id(root_error))
        root_error = root_error.error
    return root_error


def _worker_failure(error: Exception) -> tuple[ReportQuestionErrorCode, str, bool]:
    if isinstance(error, json.JSONDecodeError):
        return (
            "worker_invalid_json",
            "Report QA worker did not return a valid structured response after "
            f"{TOOL_CALL_MAX_ATTEMPTS} attempts.",
            True,
        )
    if isinstance(error, (WorkerResultError, ValidationError)):
        return (
            "worker_invalid_result",
            "Report QA worker returned an incomplete or invalid structured result.",
            True,
        )
    return (
        "worker_execution_failed",
        "Report QA worker could not complete this question.",
        False,
    )


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
            root_error = _root_error(error)
            error_code, message, retryable = _worker_failure(root_error)
            logger.warning(
                "Report QA worker failed for task_id=%s: %s: %s",
                node_input.task_id,
                type(root_error).__name__,
                root_error,
            )
            return ReportQuestionOutcome(
                task_id=node_input.task_id,
                status="failed",
                error=message,
                error_code=error_code,
                retryable=retryable,
            ).model_dump()
        try:
            result = _normalized_result(raw_result)
        except (WorkerResultError, ValidationError) as error:
            error_code, message, retryable = _worker_failure(error)
            logger.warning(
                "Report QA worker returned an invalid result for task_id=%s: %s: %s",
                node_input.task_id,
                type(error).__name__,
                error,
            )
            return ReportQuestionOutcome(
                task_id=node_input.task_id,
                status="failed",
                error=message,
                error_code=error_code,
                retryable=retryable,
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
