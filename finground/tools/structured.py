"""ADK function tool with an explicit nested JSON parameter schema."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.tools import FunctionTool
from google.genai import types


class JsonSchemaFunctionTool(FunctionTool):
    """Use a supplied JSON schema when ADK cannot inline nested Pydantic models."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        parameters_json_schema: dict[str, Any],
    ) -> None:
        super().__init__(func)
        self._parameters_json_schema = parameters_json_schema

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._parameters_json_schema,
        )
