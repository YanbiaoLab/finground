"""Claude Code-style, session-backed task graph tools."""

from __future__ import annotations

from typing import Any, ClassVar

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

TASKS_STATE_KEY = "tasks"
TASK_COUNTER_STATE_KEY = "task_counter"
TASK_TOOL_NAMES = ("TaskCreate", "TaskList", "TaskGet", "TaskUpdate")


def _load_tasks(tool_context: ToolContext) -> dict[str, dict[str, Any]]:
    value = tool_context.state.get(TASKS_STATE_KEY, {})
    if not isinstance(value, dict):
        return {}
    return {
        task_id: {
            **task,
            "blocks": list(task.get("blocks", [])),
            "blockedBy": list(task.get("blockedBy", [])),
            "metadata": dict(task.get("metadata", {})),
        }
        for task_id, task in value.items()
    }


def _save_tasks(tool_context: ToolContext, tasks: dict[str, dict[str, Any]]) -> None:
    tool_context.state[TASKS_STATE_KEY] = tasks


def _new_task_id(tool_context: ToolContext) -> str:
    counter = int(tool_context.state.get(TASK_COUNTER_STATE_KEY, 0)) + 1
    tool_context.state[TASK_COUNTER_STATE_KEY] = counter
    return str(counter)


def _task_get_output(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "subject": task["subject"],
        "description": task["description"],
        "status": task["status"],
        "blocks": list(task["blocks"]),
        "blockedBy": list(task["blockedBy"]),
    }


def _task_list_output(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "subject": task["subject"],
        "status": task["status"],
        **({"owner": task["owner"]} if task.get("owner") else {}),
        "blockedBy": list(task["blockedBy"]),
    }


class _TaskTool(BaseTool):
    parameters_json_schema: ClassVar[dict[str, Any]]

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.parameters_json_schema,
        )


class TaskCreateTool(_TaskTool):
    """Create a task with optional presentation text and arbitrary metadata."""

    parameters_json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "A brief title for the task."},
            "description": {"type": "string", "description": "What needs to be done."},
            "activeForm": {
                "type": "string",
                "description": "Present continuous form shown while the task is in progress.",
            },
            "metadata": {
                "type": "object",
                "description": "Arbitrary metadata to attach to the task.",
                "additionalProperties": True,
            },
        },
        "required": ["subject", "description"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="TaskCreate",
            description="Create a task and return its ID and subject.",
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        tasks = _load_tasks(tool_context)
        task_id = _new_task_id(tool_context)
        task = {
            "id": task_id,
            "subject": args["subject"],
            "description": args["description"],
            "activeForm": args.get("activeForm"),
            "status": "pending",
            "blocks": [],
            "blockedBy": [],
            "owner": None,
            "metadata": dict(args.get("metadata", {})),
        }
        tasks[task_id] = task
        _save_tasks(tool_context, tasks)
        return {"task": {"id": task_id, "subject": task["subject"]}}


class TaskListTool(_TaskTool):
    """List compact summaries of every current task."""

    parameters_json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(name="TaskList", description="List all current tasks.")

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        del args
        tasks = _load_tasks(tool_context)
        ordered = sorted(tasks.values(), key=lambda task: int(task["id"]))
        return {"tasks": [_task_list_output(task) for task in ordered]}


class TaskGetTool(_TaskTool):
    """Return the complete public record for one task."""

    parameters_json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to retrieve."}
        },
        "required": ["taskId"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(name="TaskGet", description="Retrieve one task by ID.")

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        task_id = args.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            return {"task": None, "error": "missing required argument: taskId"}
        task = _load_tasks(tool_context).get(task_id)
        return {"task": _task_get_output(task) if task is not None else None}


def _merge_metadata(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _link_tasks(
    tasks: dict[str, dict[str, Any]],
    *,
    task_id: str,
    related_ids: list[str],
    relation: str,
) -> str | None:
    unknown = [related_id for related_id in related_ids if related_id not in tasks]
    if unknown:
        return f"unknown related task IDs: {', '.join(unknown)}"
    if task_id in related_ids:
        return "a task cannot block itself"
    inverse = "blockedBy" if relation == "blocks" else "blocks"
    for related_id in related_ids:
        if related_id not in tasks[task_id][relation]:
            tasks[task_id][relation].append(related_id)
        if task_id not in tasks[related_id][inverse]:
            tasks[related_id][inverse].append(task_id)
    return None


def _delete_task(tasks: dict[str, dict[str, Any]], task_id: str) -> None:
    tasks.pop(task_id)
    for task in tasks.values():
        task["blocks"] = [value for value in task["blocks"] if value != task_id]
        task["blockedBy"] = [value for value in task["blockedBy"] if value != task_id]


class TaskUpdateTool(_TaskTool):
    """Update task fields, status, ownership, metadata, or graph dependencies."""

    parameters_json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to update."},
            "subject": {"type": "string", "description": "New subject for the task."},
            "description": {"type": "string", "description": "New description for the task."},
            "activeForm": {
                "type": "string",
                "description": "New present continuous form shown while in progress.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "deleted"],
                "description": "New status for the task.",
            },
            "addBlocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task IDs that this task blocks.",
            },
            "addBlockedBy": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task IDs that block this task.",
            },
            "owner": {"type": "string", "description": "New owner for the task."},
            "metadata": {
                "type": "object",
                "description": "Metadata keys to merge; null deletes a key.",
                "additionalProperties": True,
            },
        },
        "required": ["taskId"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        super().__init__(
            name="TaskUpdate",
            description="Update an existing task. Always include its required taskId.",
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        tasks = _load_tasks(tool_context)
        task_id = args.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            return {
                "success": False,
                "taskId": None,
                "updatedFields": [],
                "error": "missing required argument: taskId",
            }
        task = tasks.get(task_id)
        if task is None:
            return {
                "success": False,
                "taskId": task_id,
                "updatedFields": [],
                "error": "unknown task",
            }
        requested_fields = [key for key in args if key != "taskId"]
        if args.get("status") == "deleted":
            previous = task["status"]
            _delete_task(tasks, task_id)
            _save_tasks(tool_context, tasks)
            return {
                "success": True,
                "taskId": task_id,
                "updatedFields": ["status"],
                "statusChange": {"from": previous, "to": "deleted"},
            }
        for input_name, relation in (("addBlocks", "blocks"), ("addBlockedBy", "blockedBy")):
            if input_name in args and (
                error := _link_tasks(
                    tasks,
                    task_id=task_id,
                    related_ids=args[input_name],
                    relation=relation,
                )
            ):
                return {
                    "success": False,
                    "taskId": task_id,
                    "updatedFields": [],
                    "error": error,
                }
        previous_status = task["status"]
        for field in ("subject", "description", "activeForm", "owner"):
            if field in args:
                task[field] = args[field]
        if "status" in args:
            task["status"] = args["status"]
        if "metadata" in args:
            task["metadata"] = _merge_metadata(task["metadata"], args["metadata"])
        tasks[task_id] = task
        _save_tasks(tool_context, tasks)
        output: dict[str, Any] = {
            "success": True,
            "taskId": task_id,
            "updatedFields": requested_fields,
        }
        if task["status"] != previous_status:
            output["statusChange"] = {"from": previous_status, "to": task["status"]}
        return output


task_create = TaskCreateTool()
task_list = TaskListTool()
task_get = TaskGetTool()
task_update = TaskUpdateTool()
