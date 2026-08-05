"""ADK plugin that reminds and guards the root task workflow."""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import ToolContext
from google.genai import types

from finground.task_store import TASK_TOOL_NAMES, TASKS_STATE_KEY

ROOT_AGENT_NAME = "root_agent"
TASK_TOOL_CALL_COUNT_STATE_KEY = "task_tool_call_count"


def _progress(state: Any) -> dict[str, Any]:
    tasks = state.get(TASKS_STATE_KEY, {})
    values = list(tasks.values()) if isinstance(tasks, dict) else []
    counts = dict.fromkeys(("pending", "in_progress", "completed"), 0)
    for task in values:
        status = task.get("status")
        if status in counts:
            counts[status] += 1
    unfinished = []
    for task in values:
        if task["status"] == "completed":
            continue
        metadata = task.get("metadata", {})
        error = metadata.get("error") if isinstance(metadata, dict) else None
        unfinished.append(
            {
                "id": task["id"],
                "subject": task["subject"],
                "status": task["status"],
                "blockedBy": list(task.get("blockedBy", [])),
                **({"error": error} if error else {}),
            }
        )
    return {"counts": counts, "unfinished": unfinished}


class TaskProgressPlugin(BasePlugin):
    """Expose fresh progress after task tools and prevent premature root answers."""

    def __init__(self) -> None:
        super().__init__(name="task_progress")

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        del tool_args
        if tool.name not in TASK_TOOL_NAMES or not isinstance(result, dict):
            return None
        calls = int(tool_context.state.get(TASK_TOOL_CALL_COUNT_STATE_KEY, 0)) + 1
        tool_context.state[TASK_TOOL_CALL_COUNT_STATE_KEY] = calls
        progress = _progress(tool_context.state)
        return {
            **result,
            "progress_reminder": {
                **progress,
                "next_action": (
                    "continue active tasks; tasks with recorded errors may be reported incomplete"
                    if any("error" not in task for task in progress["unfinished"])
                    else "report the recorded incomplete tasks honestly"
                    if progress["unfinished"]
                    else "all tasks are completed"
                ),
            },
        }

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        if callback_context.agent_name != ROOT_AGENT_NAME or llm_response.get_function_calls():
            return None
        progress = _progress(callback_context.state)
        active = [task for task in progress["unfinished"] if "error" not in task]
        if not active:
            return None
        guarded = llm_response.model_copy(deep=True)
        guarded.content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="TaskList",
                        args={},
                    )
                )
            ],
        )
        return guarded
