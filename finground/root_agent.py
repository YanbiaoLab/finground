"""Root coordinator."""

from __future__ import annotations

from google.adk.agents import Agent

from finground.config import create_agent_model
from finground.kpi_dispatcher import DISPATCHER_NAME, create_kpi_dispatcher
from finground.kpi_worker import WORKER_NAME, create_tool_call_retry_config
from finground.report_qa_dispatcher import (
    REPORT_QA_DISPATCHER_NAME,
    create_report_qa_dispatcher,
)
from finground.report_qa_worker import REPORT_QA_WORKER_NAME
from finground.task_plugin import ROOT_AGENT_NAME
from finground.task_store import task_create, task_get, task_list, task_update

ROOT_INSTRUCTION = f"""You coordinate multi-step work and delegate specialized execution. You may
answer ordinary conversational questions yourself, but you never read annual-report text, choose
report evidence, calculate KPI values, or alter a worker result.

Whenever you call a tool, call exactly one tool and supply one complete, valid JSON object for its
arguments. Never emit empty, whitespace-only, partial, or prose-form tool arguments.

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

For a question about the uploaded annual report that is not a canonical KPI extraction:
1. Create one task with metadata.task_input containing report_ref and the user's exact question.
2. Add the returned task ID to metadata.task_input, then set the task to in_progress with owner
   {REPORT_QA_WORKER_NAME}. Keep task-management calls sequential.
3. Call {REPORT_QA_DISPATCHER_NAME} once with that exact task input and not alongside another tool.
4. On success, preserve outcome.result exactly in metadata.result, clear any stale error, and
   complete the task. On failure, return the task to pending and record outcome.error.
5. Inspect the task list before answering and cite the worker evidence in a readable response.

Route canonical KPI extraction through {DISPATCHER_NAME}, never through
{REPORT_QA_DISPATCHER_NAME}. If one user request mixes KPI extraction with other report questions,
create all necessary tasks and run the appropriate dispatcher for each task type. Questions that do
not require the uploaded report do not need a task or worker.
"""


def create_root_agent(
    *,
    worker: Agent | None = None,
    report_qa_worker: Agent | None = None,
) -> Agent:
    return Agent(
        name=ROOT_AGENT_NAME,
        model=create_agent_model(),
        description="A general-purpose coordinator for complex, multi-step work.",
        instruction=ROOT_INSTRUCTION,
        retry_config=create_tool_call_retry_config(),
        tools=[
            task_create,
            task_list,
            task_get,
            task_update,
            create_kpi_dispatcher(worker),
            create_report_qa_dispatcher(report_qa_worker),
        ],
    )
