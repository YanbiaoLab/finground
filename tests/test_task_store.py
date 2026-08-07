import asyncio
from copy import deepcopy
from types import SimpleNamespace

from finground.task_store import TASKS_STATE_KEY, task_create, task_get, task_list, task_update


def _call(tool: object, args: dict, context: SimpleNamespace) -> dict:
    return asyncio.run(tool.run_async(args=args, tool_context=context))


def _create(context: SimpleNamespace, subject: str = "Extract revenue") -> str:
    result = _call(
        task_create,
        {
            "subject": subject,
            "description": "Extract one KPI from the specified annual report.",
            "activeForm": "Extracting revenue",
            "metadata": {"task_input": {"kpi_key": "revenue"}},
        },
        context,
    )
    return result["task"]["id"]


def test_claude_style_create_get_and_list_contracts() -> None:
    context = SimpleNamespace(state={})
    task_id = _create(context)

    assert task_id == "1"
    assert _call(task_get, {"taskId": task_id}, context) == {
        "task": {
            "id": "1",
            "subject": "Extract revenue",
            "description": "Extract one KPI from the specified annual report.",
            "status": "pending",
            "blocks": [],
            "blockedBy": [],
        }
    }
    assert _call(task_list, {}, context) == {
        "tasks": [
            {
                "id": "1",
                "subject": "Extract revenue",
                "status": "pending",
                "blockedBy": [],
            }
        ]
    }


def test_update_merges_metadata_and_reports_status_change() -> None:
    context = SimpleNamespace(state={})
    task_id = _create(context)

    updated = _call(
        task_update,
        {
            "taskId": task_id,
            "status": "in_progress",
            "owner": "kpi_worker",
            "metadata": {"result": {"value": 123}, "task_input": None},
        },
        context,
    )

    assert updated == {
        "success": True,
        "taskId": "1",
        "updatedFields": ["status", "owner", "metadata"],
        "statusChange": {"from": "pending", "to": "in_progress"},
    }
    stored = context.state[TASKS_STATE_KEY][task_id]
    assert stored["owner"] == "kpi_worker"
    assert stored["metadata"] == {"result": {"value": 123}}


def test_dependencies_are_symmetric_and_invalid_update_is_atomic() -> None:
    context = SimpleNamespace(state={})
    first = _create(context, "First")
    second = _create(context, "Second")
    before = deepcopy(context.state[TASKS_STATE_KEY])

    failed = _call(
        task_update,
        {"taskId": first, "addBlocks": [second], "addBlockedBy": ["missing"]},
        context,
    )

    assert failed["success"] is False
    assert context.state[TASKS_STATE_KEY] == before

    linked = _call(task_update, {"taskId": first, "addBlocks": [second]}, context)
    assert linked["success"] is True
    assert context.state[TASKS_STATE_KEY][first]["blocks"] == [second]
    assert context.state[TASKS_STATE_KEY][second]["blockedBy"] == [first]


def test_deleted_task_is_removed_from_dependency_graph() -> None:
    context = SimpleNamespace(state={})
    first = _create(context, "First")
    second = _create(context, "Second")
    _call(task_update, {"taskId": first, "addBlocks": [second]}, context)

    deleted = _call(task_update, {"taskId": first, "status": "deleted"}, context)

    assert deleted["statusChange"] == {"from": "pending", "to": "deleted"}
    assert first not in context.state[TASKS_STATE_KEY]
    assert context.state[TASKS_STATE_KEY][second]["blockedBy"] == []


def test_unknown_task_returns_null_or_error_without_mutation() -> None:
    context = SimpleNamespace(state={})

    assert _call(task_get, {"taskId": "404"}, context) == {"task": None}
    assert _call(task_update, {"taskId": "404", "status": "completed"}, context) == {
        "success": False,
        "taskId": "404",
        "updatedFields": [],
        "error": "unknown task",
    }
    assert context.state == {}


def test_update_missing_task_id_returns_recoverable_error_without_mutation() -> None:
    context = SimpleNamespace(state={})
    _create(context)
    before = deepcopy(context.state)

    failed = _call(
        task_update,
        {"status": "completed", "metadata": {"result": {"status": "answered"}}},
        context,
    )

    assert failed == {
        "success": False,
        "taskId": None,
        "updatedFields": [],
        "error": "missing required argument: taskId",
    }
    assert context.state == before


def test_get_missing_task_id_returns_recoverable_error() -> None:
    context = SimpleNamespace(state={})

    assert _call(task_get, {}, context) == {
        "task": None,
        "error": "missing required argument: taskId",
    }
