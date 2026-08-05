"""ADK function tool with a protocol-facing name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

from google.adk.tools import FunctionTool
from google.genai import types


class NamedFunctionTool(FunctionTool):
    """Keep Python callable names separate from external tool names."""

    def __init__(self, func: Callable[..., Any], *, name: str) -> None:
        super().__init__(func)
        self.name = name

    @override
    def _get_declaration(self) -> types.FunctionDeclaration | None:
        declaration = super()._get_declaration()
        if declaration is not None:
            declaration.name = self.name
        return declaration
