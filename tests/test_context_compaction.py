import asyncio
from types import SimpleNamespace

from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.models.llm_request import LlmRequest
from google.genai import types

import finground.context_compaction as compaction
from finground.context_compaction import ScopedContextCompactionPlugin


class RecordingSummarizer:
    def __init__(self) -> None:
        self.calls: list[list[Event]] = []

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event:
        self.calls.append(events)
        return Event(
            author="user",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=f"summary-{len(self.calls)}")],
                    ),
                )
            ),
        )


def _text(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def _tool_call(call_id: str) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(function_call=types.FunctionCall(id=call_id, name="SearchReport", args={}))
        ],
    )


def _tool_response(call_id: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id,
                    name="SearchReport",
                    response={"result": "evidence"},
                )
            )
        ],
    )


def test_compacts_filtered_context_and_reuses_rolling_summary(monkeypatch) -> None:
    monkeypatch.setattr(compaction, "CONTEXT_TOKEN_THRESHOLD", 100)
    monkeypatch.setattr(compaction, "CONTEXT_RETENTION_TOKENS", 10)
    summarizer = RecordingSummarizer()
    plugin = ScopedContextCompactionPlugin(summarizer=summarizer)
    state: dict[str, object] = {}
    context = SimpleNamespace(agent_name="root_agent", isolation_scope=None, state=state)
    raw_contents = [
        _text("user", "old request " * 50),
        _tool_call("call-1"),
        _tool_response("call-1"),
        _text("user", "latest request"),
    ]

    async def run() -> tuple[LlmRequest, LlmRequest, types.Content]:
        first_request = LlmRequest(contents=raw_contents)
        await plugin.before_model_callback(callback_context=context, llm_request=first_request)
        short_answer = _text("model", "short answer")
        second_request = LlmRequest(contents=[*raw_contents, short_answer])
        await plugin.before_model_callback(callback_context=context, llm_request=second_request)
        return first_request, second_request, short_answer

    first_request, second_request, short_answer = asyncio.run(run())

    assert len(summarizer.calls) == 1
    assert len(summarizer.calls[0]) == 3
    assert first_request.contents[0].parts[0].text == "summary-1"
    assert first_request.contents[1:] == raw_contents[3:]

    assert len(summarizer.calls) == 1
    assert second_request.contents[0].parts[0].text == "summary-1"
    assert second_request.contents[1:] == [raw_contents[3], short_answer]


def test_task_scope_cache_is_isolated_and_cleared(monkeypatch) -> None:
    monkeypatch.setattr(compaction, "CONTEXT_TOKEN_THRESHOLD", 20)
    monkeypatch.setattr(compaction, "CONTEXT_RETENTION_TOKENS", 5)
    summarizer = RecordingSummarizer()
    plugin = ScopedContextCompactionPlugin(summarizer=summarizer)
    state: dict[str, object] = {}
    context = SimpleNamespace(agent_name="kpi_worker", isolation_scope="task/1", state=state)
    request = LlmRequest(contents=[_text("user", "old " * 80), _text("model", "recent evidence")])
    other_scope = SimpleNamespace(agent_name="kpi_worker", isolation_scope="task/2", state=state)

    assert plugin._state_key(context) != plugin._state_key(other_scope)

    async def run() -> str:
        await plugin.before_model_callback(callback_context=context, llm_request=request)
        state_key = plugin._state_key(context)
        assert state[state_key] is not None
        await plugin.after_agent_callback(
            agent=SimpleNamespace(mode="task"), callback_context=context
        )
        return state_key

    state_key = asyncio.run(run())
    assert state[state_key] is None


def test_split_never_orphans_a_tool_response(monkeypatch) -> None:
    monkeypatch.setattr(compaction, "CONTEXT_RETENTION_TOKENS", 1)
    contents = [
        _text("user", "old"),
        _tool_call("call-1"),
        _tool_response("call-1"),
        _text("model", "recent"),
    ]

    assert compaction._split_index(contents) == 3
