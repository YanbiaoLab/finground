"""Workflow-backed parallel dispatch for KPI task agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from google.adk import Context
from google.adk.agents import Agent
from google.adk.workflow import Workflow, node
from google.adk.workflow._errors import DynamicNodeFailError
from pydantic import BaseModel, Field, ValidationError, model_validator

from finground.kpi_worker import KpiTaskInput, KpiTaskResult, create_kpi_worker

DISPATCHER_NAME = "dispatch_kpi_tasks"
DISPATCH_ITEM_NAME = "dispatch_kpi_item"
MAX_PARALLEL_WORKERS = 4


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

    @model_validator(mode="after")
    def validate_outcome(self) -> KpiDispatchOutcome:
        if self.status == "succeeded" and (self.result is None or self.error is not None):
            raise ValueError("succeeded dispatch outcome requires result and forbids error")
        if self.status == "failed" and (self.result is not None or not self.error):
            raise ValueError("failed dispatch outcome requires error and forbids result")
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


def _root_error_message(error: DynamicNodeFailError) -> str:
    root_error: Exception = error
    seen: set[int] = set()
    while isinstance(root_error, DynamicNodeFailError) and id(root_error) not in seen:
        seen.add(id(root_error))
        root_error = root_error.error
    return f"{type(root_error).__name__}: {root_error}"


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
            return KpiDispatchOutcome(
                task_id=node_input.task_id,
                kpi_key=node_input.kpi_key,
                status="failed",
                error=_root_error_message(error),
            ).model_dump()
        try:
            result = _normalized_result(raw_result)
        except (WorkerResultError, ValidationError) as error:
            return KpiDispatchOutcome(
                task_id=node_input.task_id,
                kpi_key=node_input.kpi_key,
                status="failed",
                error=f"{type(error).__name__}: {error}",
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
        description="Execute independent KPI extraction tasks concurrently.",
        input_schema=KpiTaskBatch,
        output_schema=list[KpiDispatchOutcome],
        edges=[("START", _task_inputs, parallel_worker, _normalized_outcomes)],
    )
