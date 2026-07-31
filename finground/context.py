"""ADK-native context reduction for state-backed Multi-KPI extraction."""

from __future__ import annotations

import copy

from google.genai import types

MULTI_KPI_RECORD_TOOL = "record_multi_kpi_progress"
_RETRIEVAL_TOOL_NAMES = {
    "inspect_primary_statements",
    "read_report_pages",
    "search_report",
}
_COMPACTABLE_TOOL_NAMES = {
    "inspect_primary_statements",
    "query_multi_kpi_progress",
    "read_report_pages",
    "search_report",
    MULTI_KPI_RECORD_TOOL,
}
_COMPACTED_RESPONSE = {
    "status": "compacted",
    "message": "Older tool payload removed after a newer retrieval became active.",
}


def _active_retrieval_boundary(contents: list[types.Content]) -> int | None:
    latest_response_index: int | None = None
    latest_response_id: str | None = None
    latest_response_name: str | None = None
    for index, content in enumerate(contents):
        for part in content.parts or []:
            response = part.function_response
            if (
                response is not None
                and response.name in _RETRIEVAL_TOOL_NAMES
                and isinstance(response.response, dict)
                and response.response.get("status") == "success"
            ):
                latest_response_index = index
                latest_response_id = response.id
                latest_response_name = response.name
    if latest_response_index is None:
        return None

    boundary = latest_response_index
    for index in range(latest_response_index - 1, -1, -1):
        for part in contents[index].parts or []:
            call = part.function_call
            if (
                call is not None
                and call.name == latest_response_name
                and (latest_response_id is None or call.id == latest_response_id)
            ):
                boundary = index
                break
        if boundary != latest_response_index:
            break

    has_older_tool_payload = any(
        (part.function_call is not None and part.function_call.name in _COMPACTABLE_TOOL_NAMES)
        or (
            part.function_response is not None
            and part.function_response.name in _COMPACTABLE_TOOL_NAMES
        )
        for content in contents[:boundary]
        for part in content.parts or []
    )
    return boundary if has_older_tool_payload else None


def filter_recorded_multi_kpi_context(
    contents: list[types.Content],
) -> list[types.Content]:
    """Compact completed retrieval batches while retaining the active source pages.

    This function is supplied to ADK's ``ContextFilterPlugin``. It changes only
    the model request copy: immutable report text in session state and the
    session event history remain untouched. Function call/response identities
    are retained so OpenAI-compatible tool history remains valid. A retrieval
    remains visible across every progress-record batch that consumes it and is
    compacted whenever a newer retrieval starts. This also bounds context after
    validation failures, when no successful record checkpoint exists yet.
    """
    boundary = _active_retrieval_boundary(contents)
    if boundary is None:
        return contents

    filtered = copy.deepcopy(contents)
    for content in filtered[:boundary]:
        for part in content.parts or []:
            call = part.function_call
            if call is not None and call.name in _COMPACTABLE_TOOL_NAMES:
                call.args = {"compacted": True}
            response = part.function_response
            if response is not None and response.name in _COMPACTABLE_TOOL_NAMES:
                response.response = dict(_COMPACTED_RESPONSE)
    return filtered
