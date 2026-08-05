"""Scope-safe rolling context compaction for every LLM agent."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

CONTEXT_TOKEN_THRESHOLD = 64_000
CONTEXT_RETENTION_TOKENS = 16_000
_APPROX_CHARS_PER_TOKEN = 4
_MAX_SUMMARY_CHARS = 16_000
_ROOT_SCOPE = "root"
_STATE_KEY_PREFIX = "context_compaction:"

logger = logging.getLogger(__name__)


def _serialized(content: types.Content) -> str:
    return content.model_dump_json(exclude_none=True)


def _estimated_tokens(contents: list[types.Content]) -> int:
    total_chars = sum(len(_serialized(content)) for content in contents)
    return total_chars // _APPROX_CHARS_PER_TOKEN


def _digest(contents: list[types.Content]) -> str:
    digest = sha256()
    for content in contents:
        digest.update(_serialized(content).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _call_ids(content: types.Content) -> set[str]:
    return {
        part.function_call.id
        for part in content.parts or []
        if part.function_call is not None and part.function_call.id
    }


def _response_ids(content: types.Content) -> set[str]:
    return {
        part.function_response.id
        for part in content.parts or []
        if part.function_response is not None and part.function_response.id
    }


def _safe_prefix_boundaries(contents: list[types.Content]) -> list[int]:
    open_calls: set[str] = set()
    boundaries: list[int] = []
    for index, content in enumerate(contents, start=1):
        open_calls -= _response_ids(content)
        open_calls |= _call_ids(content)
        if not open_calls:
            boundaries.append(index)
    return boundaries


def _split_index(contents: list[types.Content]) -> int:
    retained_chars = 0
    desired_split = len(contents)
    retention_chars = CONTEXT_RETENTION_TOKENS * _APPROX_CHARS_PER_TOKEN
    for index in range(len(contents) - 1, -1, -1):
        retained_chars += len(_serialized(contents[index]))
        desired_split = index
        if retained_chars >= retention_chars:
            break

    safe_boundaries = _safe_prefix_boundaries(contents)
    return max((boundary for boundary in safe_boundaries if boundary <= desired_split), default=0)


def _summary_content(record: dict[str, Any] | None) -> types.Content | None:
    if not isinstance(record, dict) or not isinstance(record.get("summary"), dict):
        return None
    try:
        return types.Content.model_validate(record["summary"])
    except (TypeError, ValueError):
        return None


def _valid_record(record: Any, contents: list[types.Content]) -> bool:
    if not isinstance(record, dict):
        return False
    prefix_count = record.get("prefix_count")
    if not isinstance(prefix_count, int) or not 0 < prefix_count <= len(contents):
        return False
    return _summary_content(record) is not None and record.get("prefix_digest") == _digest(
        contents[:prefix_count]
    )


def _bounded_summary(content: types.Content) -> types.Content:
    remaining = _MAX_SUMMARY_CHARS
    parts: list[types.Part] = []
    for part in content.parts or []:
        if not part.text or remaining <= 0:
            continue
        text = part.text[:remaining]
        parts.append(types.Part.from_text(text=text))
        remaining -= len(text)
    if not parts:
        raise ValueError("context summarizer returned no text")
    return types.Content(role="model", parts=parts)


def _events(contents: list[types.Content]) -> list[Event]:
    return [
        Event(author=content.role or "unknown", content=content, timestamp=float(index))
        for index, content in enumerate(contents, start=1)
    ]


class ScopedContextCompactionPlugin(BasePlugin):
    """Compact only the already-filtered context visible to the current agent."""

    def __init__(self, summarizer: BaseEventsSummarizer) -> None:
        super().__init__(name="scoped_context_compaction")
        self._summarizer = summarizer

    @staticmethod
    def _state_key(callback_context: CallbackContext) -> str:
        scope = callback_context.isolation_scope or _ROOT_SCOPE
        scope_hash = sha256(scope.encode()).hexdigest()[:16]
        return f"{_STATE_KEY_PREFIX}{callback_context.agent_name}:{scope_hash}"

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        contents = list(llm_request.contents)
        state_key = self._state_key(callback_context)
        record = callback_context.state.get(state_key)
        valid_record = _valid_record(record, contents)

        if valid_record:
            prefix_count = record["prefix_count"]
            summary = _summary_content(record)
            assert summary is not None
            effective_contents = [summary, *contents[prefix_count:]]
            if _estimated_tokens(effective_contents) < CONTEXT_TOKEN_THRESHOLD:
                llm_request.contents = effective_contents
                return None
        elif _estimated_tokens(contents) < CONTEXT_TOKEN_THRESHOLD:
            return None

        split_index = _split_index(contents)
        if split_index == 0:
            return None
        if valid_record and split_index <= record["prefix_count"]:
            llm_request.contents = effective_contents
            return None

        if valid_record:
            previous_summary = _summary_content(record)
            assert previous_summary is not None
            source_contents = [previous_summary, *contents[record["prefix_count"] : split_index]]
        else:
            source_contents = contents[:split_index]

        try:
            summary_event = await self._summarizer.maybe_summarize_events(
                events=_events(source_contents)
            )
            if (
                summary_event is None
                or summary_event.actions.compaction is None
                or summary_event.actions.compaction.compacted_content is None
            ):
                return None
            summary = _bounded_summary(summary_event.actions.compaction.compacted_content)
        except Exception:
            logger.exception("Context compaction failed for %s", callback_context.agent_name)
            return None

        callback_context.state[state_key] = {
            "prefix_count": split_index,
            "prefix_digest": _digest(contents[:split_index]),
            "summary": summary.model_dump(mode="json", exclude_none=True),
        }
        llm_request.contents = [summary, *contents[split_index:]]
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        if getattr(agent, "mode", None) == "task":
            callback_context.state[self._state_key(callback_context)] = None
        return None
