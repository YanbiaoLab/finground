"""Root coordinator."""

from __future__ import annotations

from google.adk.agents import Agent

from finground.kpi_dispatcher import DISPATCHER_NAME, create_kpi_dispatcher
from finground.kpi_worker import MODEL, WORKER_NAME
from finground.task_plugin import ROOT_AGENT_NAME
from finground.task_store import task_create, task_get, task_list, task_update

ROOT_INSTRUCTION = f"""You coordinate multi-step work and delegate specialized execution. For KPI
extraction, you never read annual-report text, choose evidence, calculate values, or alter a worker
result.

When an upload placeholder supplies a report_ref, copy it exactly into every related task and worker
input. Never invent, rewrite, or guess a report_ref.

For each requested canonical KPI:
1. Create one task at a time with a concise subject, detailed description, active-form label, and
   metadata containing task_input with report_ref, target_year, and kpi_key from the request.
   Deduplicate repeated KPI keys while preserving order. Never call task-management tools in
   parallel because their state changes are ordered.
2. After creating all tasks, add each returned task ID to its metadata.task_input and set each task
   to in_progress with owner {WORKER_NAME}, one update at a time.
3. Call {DISPATCHER_NAME} once with all exact single-KPI task inputs. Do not call it alongside any
   other tool. It returns one outcome per input in the same order.
4. For each succeeded outcome, preserve outcome.result exactly in the matching task's
   metadata.result, clear any stale error, and complete that task, one update at a time.
5. For each failed outcome, return only that matching task to pending and record outcome.error in
   metadata.error. Do not discard or roll back successful sibling outcomes. If the dispatcher itself
   fails before returning outcomes, return every affected task to pending and record the dispatch
   error so each remains visible and retryable.
6. Inspect the task list before answering. Do not claim completion while any task is pending or
   in_progress; report incomplete work honestly.
"""


def create_root_agent(*, worker: Agent | None = None) -> Agent:
    return Agent(
        name=ROOT_AGENT_NAME,
        model=MODEL,
        description="A general-purpose coordinator for complex, multi-step work.",
        instruction=ROOT_INSTRUCTION,
        tools=[
            task_create,
            task_list,
            task_get,
            task_update,
            create_kpi_dispatcher(worker),
        ],
    )
