"""Full local ADK lifecycle and event trajectory recording."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=False, exclude_none=True)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bytes):
        return {"byte_count": len(value)}
    return str(value)


def _dump_model(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json", by_alias=False, exclude_none=True)


def _callback_metadata(callback_context: CallbackContext | ToolContext) -> dict[str, Any]:
    return {
        "invocation_id": callback_context.invocation_id,
        "agent_name": callback_context.agent_name,
        "function_call_id": callback_context.function_call_id,
    }


def _invocation_metadata(invocation_context: InvocationContext) -> dict[str, Any]:
    session = invocation_context.session
    return {
        "invocation_id": invocation_context.invocation_id,
        "agent_name": invocation_context.agent.name,
        "session_id": session.id,
        "user_id": session.user_id,
        "app_name": session.app_name,
    }


def _error_payload(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


class AdkTrajectoryPlugin(BasePlugin):
    """Write every ADK lifecycle hook and yielded event to one JSONL file."""

    def __init__(self, path: Path) -> None:
        super().__init__(name=f"finground_adk_trajectory_{path.stem}")
        self.path = path
        self.partial_path = path.with_suffix(f"{path.suffix}.partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.partial_path.open("w", encoding="utf-8")
        self._started = time.monotonic()
        self._sequence = 0
        self._write_error: str | None = None
        self._complete = False

    def _record(
        self,
        kind: str,
        *,
        context: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._stream.closed or self._write_error is not None:
            return
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "timestamp_unix": time.time(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1_000, 3),
            "kind": kind,
            "context": context or {},
            "payload": payload or {},
        }
        try:
            self._stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=_json_default,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._stream.flush()
        except OSError as error:
            self._write_error = f"{type(error).__name__}: {error}"

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        self._record(
            "user_message",
            context=_invocation_metadata(invocation_context),
            payload={"content": _dump_model(user_message)},
        )
        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        self._record(
            "before_run",
            context=_invocation_metadata(invocation_context),
        )
        return None

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Event | None:
        self._record(
            "event",
            context=_invocation_metadata(invocation_context),
            payload={"event": _dump_model(event)},
        )
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._record(
            "after_run",
            context=_invocation_metadata(invocation_context),
        )

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        self._record(
            "before_agent",
            context=_callback_metadata(callback_context),
            payload={"agent": agent.name},
        )
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        self._record(
            "after_agent",
            context=_callback_metadata(callback_context),
            payload={"agent": agent.name},
        )
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        self._record(
            "before_model",
            context=_callback_metadata(callback_context),
            payload={"request": _dump_model(llm_request)},
        )
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        self._record(
            "after_model",
            context=_callback_metadata(callback_context),
            payload={"response": _dump_model(llm_response)},
        )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        self._record(
            "model_error",
            context=_callback_metadata(callback_context),
            payload={
                "request": _dump_model(llm_request),
                "error": _error_payload(error),
            },
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        self._record(
            "before_tool",
            context=_callback_metadata(tool_context),
            payload={"tool": tool.name, "args": tool_args},
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        self._record(
            "after_tool",
            context=_callback_metadata(tool_context),
            payload={"tool": tool.name, "args": tool_args, "result": result},
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict | None:
        self._record(
            "tool_error",
            context=_callback_metadata(tool_context),
            payload={
                "tool": tool.name,
                "args": tool_args,
                "error": _error_payload(error),
            },
        )
        return None

    def finish(self, *, outcome: str, summary: dict[str, Any]) -> None:
        """Finalize the partial trace after benchmark result construction."""
        if self._complete:
            return
        self._record(
            "trajectory_finished",
            payload={"outcome": outcome, "summary": summary},
        )
        self._stream.close()
        try:
            self.partial_path.replace(self.path)
            self._complete = True
        except OSError as error:
            self._write_error = f"{type(error).__name__}: {error}"

    def snapshot(self) -> dict[str, Any]:
        """Return compact trajectory output metadata for the raw benchmark record."""
        return {
            "path": str(self.path),
            "record_count": self._sequence,
            "write_error": self._write_error,
            "complete": self._complete,
        }
