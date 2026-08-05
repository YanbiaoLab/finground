import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import ClassVar

from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.kpi_dispatcher import DISPATCHER_NAME
from finground.kpi_worker import create_kpi_worker
from finground.root_agent import create_root_agent
from finground.task_plugin import TaskProgressPlugin
from finground.task_store import TASKS_STATE_KEY, task_create, task_list, task_update


class ScriptedLlm(BaseLlm):
    responses: list[types.Content]

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        if not self.responses:
            raise RuntimeError(f"no scripted response left for {self.model}")
        yield LlmResponse(content=self.responses.pop(0))


class ConcurrentWorkerLlm(BaseLlm):
    active_calls: ClassVar[int] = 0
    max_active_calls: ClassVar[int] = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        request_text = str(llm_request.contents[-1])
        is_net_income = "net_income" in request_text
        type(self).active_calls += 1
        type(self).max_active_calls = max(type(self).max_active_calls, type(self).active_calls)
        try:
            await asyncio.sleep(0.05)
        finally:
            type(self).active_calls -= 1
        yield LlmResponse(
            content=_tool_call(
                "finish_task",
                {
                    "task_id": "2" if is_net_income else "1",
                    "report_ref": "ACME_2025",
                    "target_year": 2025,
                    "kpi_key": "net_income" if is_net_income else "revenue",
                    "status": "ambiguous",
                    "value": None,
                    "unit": None,
                    "source_value": None,
                    "source_unit": None,
                    "evidence": None,
                    "notes": ["scripted concurrent result"],
                },
            )
        )


class PartiallyFailingWorkerLlm(BaseLlm):
    attempts: ClassVar[dict[str, int]] = {}

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        request_text = str(llm_request.contents)
        kpi_key = "capex" if "capex" in request_text else "revenue"
        type(self).attempts[kpi_key] = type(self).attempts.get(kpi_key, 0) + 1
        if kpi_key == "capex":
            raise json.JSONDecodeError("Unterminated string", '"broken', 0)
        yield LlmResponse(
            content=_tool_call(
                "finish_task",
                {
                    "task_id": "1",
                    "report_ref": "ACME_2025",
                    "target_year": 2025,
                    "kpi_key": "revenue",
                    "status": "ambiguous",
                    "value": None,
                    "unit": None,
                    "source_value": None,
                    "source_unit": None,
                    "evidence": None,
                    "notes": ["scripted successful sibling"],
                },
            )
        )


class EmptyWorkerResultLlm(BaseLlm):
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="I could not complete the task.")],
            )
        )


class MalformedRootToolCallLlm(BaseLlm):
    attempts: ClassVar[int] = 0

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        type(self).attempts += 1
        if type(self).attempts < 3:
            raise json.JSONDecodeError("Expecting value", " ", 0)
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Recovered after malformed tool call.")],
            )
        )


def _tool_call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


async def _call(tool: object, args: dict, context: SimpleNamespace) -> dict:
    return await tool.run_async(args=args, tool_context=context)


def test_progress_plugin_reminds_after_task_tools() -> None:
    context = SimpleNamespace(state={})
    plugin = TaskProgressPlugin()

    async def run() -> dict:
        await _call(
            task_create,
            {"subject": "Extract revenue", "description": "Extract the requested KPI."},
            context,
        )
        return await plugin.after_tool_callback(
            tool=SimpleNamespace(name="TaskList"),
            tool_args={},
            tool_context=context,
            result=await _call(task_list, {}, context),
        )

    result = asyncio.run(run())

    assert result["progress_reminder"]["counts"]["pending"] == 1
    assert result["progress_reminder"]["unfinished"][0]["subject"] == "Extract revenue"


def test_progress_plugin_replaces_premature_root_text_with_task_list() -> None:
    context = SimpleNamespace(state={}, agent_name="root_agent")
    plugin = TaskProgressPlugin()
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="All done")])
    )

    async def guard() -> LlmResponse:
        await _call(
            task_create,
            {"subject": "Extract revenue", "description": "Extract the requested KPI."},
            context,
        )
        return await plugin.after_model_callback(
            callback_context=context,
            llm_response=response,
        )

    guarded = asyncio.run(guard())

    assert guarded.get_function_calls()[0].name == "TaskList"
    assert guarded.get_function_calls()[0].args == {}


def test_progress_plugin_allows_an_explicitly_recorded_failure() -> None:
    context = SimpleNamespace(state={}, agent_name="root_agent")
    plugin = TaskProgressPlugin()
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="Task failed")])
    )

    async def guard() -> LlmResponse | None:
        created = await _call(
            task_create,
            {"subject": "Extract revenue", "description": "Extract the requested KPI."},
            context,
        )
        await _call(
            task_update,
            {
                "taskId": created["task"]["id"],
                "status": "pending",
                "metadata": {"error": "report artifact is unavailable"},
            },
            context,
        )
        return await plugin.after_model_callback(
            callback_context=context,
            llm_response=response,
        )

    assert asyncio.run(guard()) is None


def test_adk_task_mode_flow_completes_one_kpi() -> None:
    async def run() -> tuple[dict, list]:
        worker = create_kpi_worker()
        root = create_root_agent(worker=worker)
        result = {
            "task_id": "1",
            "report_ref": "ACME_2025",
            "target_year": 2025,
            "kpi_key": "revenue",
            "status": "found",
            "value": 123_000_000.0,
            "unit": "USD",
            "source_value": "123",
            "source_unit": "USD millions",
            "evidence": {
                "chunk_id": "ACME_2025:p72:c0",
                "page": 72,
                "statement": "Consolidated Statements of Operations",
                "label": "Revenue",
                "text": "Revenue 123",
            },
            "notes": [],
        }
        root.model = ScriptedLlm(
            model="root-script",
            responses=[
                _tool_call(
                    "TaskCreate",
                    {
                        "subject": "Extract revenue",
                        "description": "Extract revenue from ACME_2025 for fiscal year 2025.",
                        "activeForm": "Extracting revenue",
                        "metadata": {
                            "task_input": {
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            }
                        },
                    },
                ),
                _tool_call(
                    "TaskUpdate",
                    {
                        "taskId": "1",
                        "status": "in_progress",
                        "owner": "kpi_worker",
                        "metadata": {
                            "task_input": {
                                "task_id": "1",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            }
                        },
                    },
                ),
                _tool_call(
                    DISPATCHER_NAME,
                    {
                        "tasks": [
                            {
                                "task_id": "1",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            }
                        ]
                    },
                ),
                _tool_call(
                    "TaskUpdate",
                    {
                        "taskId": "1",
                        "status": "completed",
                        "metadata": {"result": result, "error": None},
                    },
                ),
                _tool_call("TaskList", {}),
                types.Content(
                    role="model", parts=[types.Part.from_text(text="Revenue extracted.")]
                ),
            ],
        )
        worker.model = ScriptedLlm(
            model="worker-script",
            responses=[
                _tool_call("GetKpiKnowledge", {"kpi_key": "revenue"}),
                _tool_call(
                    "SearchReport",
                    {
                        "report_ref": "ACME_2025",
                        "query": "revenue",
                        "cursor": "",
                        "limit": 8,
                    },
                ),
                _tool_call(
                    "ReadReportChunks",
                    {
                        "report_ref": "ACME_2025",
                        "chunk_ids": ["ACME_2025:p72:c0"],
                    },
                ),
                _tool_call("finish_task", result),
            ],
        )
        app = App(name="finground_test", root_agent=root, plugins=[TaskProgressPlugin()])
        sessions = InMemorySessionService()
        artifacts = InMemoryArtifactService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
            state={
                "report": {
                    "report_ref": "ACME_2025",
                    "artifact_name": "report.jsonl",
                }
            },
        )
        report = {
            "chunk_id": "ACME_2025:p72:c0",
            "page": 72,
            "heading": "Consolidated Statements of Operations",
            "text": "Revenue 123",
        }
        await artifacts.save_artifact(
            app_name=app.name,
            user_id="user",
            session_id="session",
            filename="report.jsonl",
            artifact=types.Part.from_bytes(
                data=json.dumps(report).encode(),
                mime_type="application/x-ndjson",
            ),
        )
        runner = Runner(app=app, session_service=sessions, artifact_service=artifacts)
        events = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Extract revenue")],
                ),
            )
        ]
        session = await sessions.get_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        return session.state[TASKS_STATE_KEY]["1"], events

    task, events = asyncio.run(run())

    assert task["status"] == "completed"
    assert task["metadata"]["result"]["value"] == 123_000_000.0
    assert events[-1].is_final_response()
    assert events[-1].content.parts[0].text == "Revenue extracted."


def test_dispatcher_runs_kpi_workers_concurrently() -> None:
    async def run() -> None:
        ConcurrentWorkerLlm.active_calls = 0
        ConcurrentWorkerLlm.max_active_calls = 0
        worker = create_kpi_worker()
        worker.model = ConcurrentWorkerLlm(model="concurrent-worker")
        root = create_root_agent(worker=worker)
        root.model = ScriptedLlm(
            model="root-script",
            responses=[
                _tool_call(
                    DISPATCHER_NAME,
                    {
                        "tasks": [
                            {
                                "task_id": "1",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            },
                            {
                                "task_id": "2",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "net_income",
                            },
                        ]
                    },
                ),
                types.Content(role="model", parts=[types.Part.from_text(text="Batch complete.")]),
            ],
        )
        app = App(name="parallel_dispatch_test", root_agent=root)
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        runner = Runner(app=app, session_service=sessions)

        events = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Extract two KPIs")],
                ),
            )
        ]

        assert events[-1].content.parts[0].text == "Batch complete."
        dispatcher_outputs = [
            event.output
            for event in events
            if event.author == DISPATCHER_NAME and isinstance(event.output, list)
        ]
        assert [result["task_id"] for result in dispatcher_outputs[-1]] == ["1", "2"]
        assert all(result["status"] == "succeeded" for result in dispatcher_outputs[-1])
        assert all(result["result"]["value"] is None for result in dispatcher_outputs[-1])

    asyncio.run(run())

    assert ConcurrentWorkerLlm.max_active_calls == 2


def test_dispatcher_retries_and_isolates_one_worker_failure() -> None:
    async def run() -> list[dict]:
        PartiallyFailingWorkerLlm.attempts = {}
        worker = create_kpi_worker()
        worker.model = PartiallyFailingWorkerLlm(model="partially-failing-worker")
        root = create_root_agent(worker=worker)
        root.model = ScriptedLlm(
            model="root-script",
            responses=[
                _tool_call(
                    DISPATCHER_NAME,
                    {
                        "tasks": [
                            {
                                "task_id": "1",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            },
                            {
                                "task_id": "2",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "capex",
                            },
                        ]
                    },
                ),
                types.Content(role="model", parts=[types.Part.from_text(text="Batch handled.")]),
            ],
        )
        app = App(name="partial_failure_test", root_agent=root)
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        runner = Runner(app=app, session_service=sessions)
        events = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Extract revenue and capex")],
                ),
            )
        ]
        return next(
            event.output
            for event in reversed(events)
            if event.author == DISPATCHER_NAME and isinstance(event.output, list)
        )

    outcomes = asyncio.run(run())

    assert PartiallyFailingWorkerLlm.attempts == {"revenue": 1, "capex": 3}
    assert outcomes[0]["status"] == "succeeded"
    assert outcomes[0]["result"]["kpi_key"] == "revenue"
    assert outcomes[1] == {
        "task_id": "2",
        "kpi_key": "capex",
        "status": "failed",
        "result": None,
        "error": "JSONDecodeError: Unterminated string: line 1 column 1 (char 0)",
    }


def test_dispatcher_converts_missing_worker_result_to_failed_outcome() -> None:
    async def run() -> list[dict]:
        worker = create_kpi_worker()
        worker.model = EmptyWorkerResultLlm(model="empty-worker-result")
        root = create_root_agent(worker=worker)
        root.model = ScriptedLlm(
            model="root-script",
            responses=[
                _tool_call(
                    DISPATCHER_NAME,
                    {
                        "tasks": [
                            {
                                "task_id": "1",
                                "report_ref": "ACME_2025",
                                "target_year": 2025,
                                "kpi_key": "revenue",
                            }
                        ]
                    },
                ),
                types.Content(role="model", parts=[types.Part.from_text(text="Handled.")]),
            ],
        )
        app = App(name="missing_worker_result_test", root_agent=root)
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        runner = Runner(app=app, session_service=sessions)
        events = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Extract revenue")],
                ),
            )
        ]
        return next(
            event.output
            for event in reversed(events)
            if event.author == DISPATCHER_NAME and isinstance(event.output, list)
        )

    assert asyncio.run(run()) == [
        {
            "task_id": "1",
            "kpi_key": "revenue",
            "status": "failed",
            "result": None,
            "error": "WorkerResultError: kpi_worker returned no task result",
        }
    ]


def test_root_retries_malformed_tool_call_json() -> None:
    async def run() -> list:
        MalformedRootToolCallLlm.attempts = 0
        root = create_root_agent()
        root.model = MalformedRootToolCallLlm(model="malformed-root-tool-call")
        app = App(name="root_json_retry_test", root_agent=root)
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        runner = Runner(app=app, session_service=sessions)
        return [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Hello")],
                ),
            )
        ]

    events = asyncio.run(run())

    assert MalformedRootToolCallLlm.attempts == 3
    assert events[-1].is_final_response()
    assert events[-1].content.parts[0].text == "Recovered after malformed tool call."
