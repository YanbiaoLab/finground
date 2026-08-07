"""Workflow-backed parallel dispatch for KPI task agents."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from google.adk import Context
from google.adk.agents import Agent
from google.adk.workflow import Workflow, node
from google.adk.workflow._errors import DynamicNodeFailError
from pydantic import BaseModel, Field, ValidationError, model_validator

from finground.kpi_catalog import KPI_CATALOG
from finground.kpi_worker import (
    TOOL_CALL_MAX_ATTEMPTS,
    KpiTaskInput,
    KpiTaskResult,
    create_kpi_worker,
)

DISPATCHER_NAME = "dispatch_kpi_tasks"
DISPATCH_ITEM_NAME = "dispatch_kpi_item"
MAX_PARALLEL_WORKERS = 4
logger = logging.getLogger(__name__)
SUPPORTED_KPI_KEYS = ", ".join(KPI_CATALOG)
DISPATCHER_DESCRIPTION = (
    "Execute independent canonical KPI extraction tasks concurrently. "
    f"Supported kpi_key values: {SUPPORTED_KPI_KEYS}. "
    "Do not use this tool for metrics outside this list; when such a request can be answered from "
    "the current annual report, assign it to the general report QA workflow instead."
)

KpiDispatchErrorCode = Literal[
    "worker_invalid_json",
    "worker_invalid_result",
    "worker_execution_failed",
]


class WorkerResultError(ValueError):
    """Raised when a task-mode worker finishes without a usable result."""


class KpiTaskBatch(BaseModel):
    """A batch of independent KPI tasks prepared by the root agent."""

    tasks: list[KpiTaskInput] = Field(min_length=1)


class KpiDispatchOutcome(BaseModel):
    """The isolated dispatcher outcome for one KPI task."""

    task_id: str
    kpi_key: str
    status: Literal["succeeded", "failed"]
    result: KpiTaskResult | None = None
    error: str | None = None
    error_code: KpiDispatchErrorCode | None = None
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> KpiDispatchOutcome:
        failure_fields = (self.error, self.error_code, self.retryable)
        if self.status == "succeeded" and (
            self.result is None or any(value is not None for value in failure_fields)
        ):
            raise ValueError("succeeded dispatch outcome requires only result")
        if self.status == "failed" and (
            self.result is not None
            or not self.error
            or self.error_code is None
            or self.retryable is None
        ):
            raise ValueError("failed dispatch outcome requires structured failure guidance")
        return self


def _task_inputs(node_input: KpiTaskBatch) -> list[KpiTaskInput]:
    return node_input.tasks


def _normalized_result(result: Any) -> KpiTaskResult:
    if result is None:
        raise WorkerResultError("kpi_worker returned no task result")
    if not isinstance(result, Mapping):
        raise WorkerResultError(
            f"kpi_worker returned {type(result).__name__}, expected an object"
        )
    nullable_fields = {
        "value": None,
        "unit": None,
        "source_value": None,
        "source_unit": None,
        "evidence": None,
    }
    return KpiTaskResult.model_validate({**nullable_fields, **result})


def _normalized_outcomes(node_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [KpiDispatchOutcome.model_validate(outcome).model_dump() for outcome in node_input]


def _root_error(error: DynamicNodeFailError) -> Exception:
    root_error: Exception = error
    seen: set[int] = set()
    while isinstance(root_error, DynamicNodeFailError) and id(root_error) not in seen:
        seen.add(id(root_error))
        root_error = root_error.error
    return root_error


def root_error_message(error: DynamicNodeFailError) -> str:
    """Return the technical nested error text for legacy non-KPI dispatchers."""
    root_error = _root_error(error)
    return f"{type(root_error).__name__}: {root_error}"


def _worker_failure(error: Exception) -> tuple[KpiDispatchErrorCode, str, bool]:
    if isinstance(error, json.JSONDecodeError):
        return (
            "worker_invalid_json",
            "KPI worker did not return a valid structured response after "
            f"{TOOL_CALL_MAX_ATTEMPTS} attempts. Retry only this KPI and keep successful "
            "sibling results.",
            True,
        )
    if isinstance(error, (WorkerResultError, ValidationError)):
        return (
            "worker_invalid_result",
            "KPI worker returned an incomplete or invalid structured result. Retry only this KPI "
            "and keep successful sibling results.",
            True,
        )
    return (
        "worker_execution_failed",
        "KPI worker could not complete this item. Keep successful sibling results and report this "
        "KPI as unfinished.",
        False,
    )


def create_kpi_dispatcher(worker: Agent | None = None) -> Workflow:
    """Create a workflow tool that fans KPI tasks out to isolated workers."""
    task_worker = worker or create_kpi_worker()

    async def run_task_worker(node_input: KpiTaskInput, ctx: Context) -> dict[str, Any]:
        isolation_scope = f"{ctx.isolation_scope or DISPATCHER_NAME}/{node_input.task_id}"
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
                "KPI worker failed for task_id=%s kpi_key=%s: %s: %s",
                node_input.task_id,
                node_input.kpi_key,
                type(root_error).__name__,
                root_error,
            )
            return KpiDispatchOutcome(
                task_id=node_input.task_id,
                kpi_key=node_input.kpi_key,
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
                "KPI worker returned an invalid result for task_id=%s kpi_key=%s: %s: %s",
                node_input.task_id,
                node_input.kpi_key,
                type(error).__name__,
                error,
            )
            return KpiDispatchOutcome(
                task_id=node_input.task_id,
                kpi_key=node_input.kpi_key,
                status="failed",
                error=message,
                error_code=error_code,
                retryable=retryable,
            ).model_dump()
        return KpiDispatchOutcome(
            task_id=node_input.task_id,
            kpi_key=node_input.kpi_key,
            status="succeeded",
            result=result,
        ).model_dump()

    parallel_worker = node(
        run_task_worker,
        name=DISPATCH_ITEM_NAME,
        rerun_on_resume=True,
        parallel_worker=True,
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )
    return Workflow(
        name=DISPATCHER_NAME,
        description=DISPATCHER_DESCRIPTION,
        input_schema=KpiTaskBatch,
        # Every item is already validated by _normalized_outcomes. Keep the provider-facing
        # response schema shallow so tool-call models do not receive the full nested result model.
        output_schema=list[dict[str, Any]],
        edges=[("START", _task_inputs, parallel_worker, _normalized_outcomes)],
    )
