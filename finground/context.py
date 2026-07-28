"""ADK-native context reduction for state-backed Multi-KPI extraction."""

from __future__ import annotations

import copy

from google.genai import types

MULTI_KPI_RECORD_TOOL = "record_multi_kpi_progress"
_RETRIEVAL_TOOL_NAMES = {"read_report_pages", "search_report"}
_COMPACTABLE_TOOL_NAMES = {
    "get_report_info",
    "query_multi_kpi_progress",
    "read_report_pages",
    "search_report",
    MULTI_KPI_RECORD_TOOL,
}
_COMPACTED_RESPONSE = {
    "status": "compacted",
    "message": "Older tool payload removed after progress was recorded in session state.",
}


def _is_successful_record(content: types.Content) -> bool:
    return any(
        response is not None
        and response.name == MULTI_KPI_RECORD_TOOL
        and isinstance(response.response, dict)
        and response.response.get("status") in {"success", "partial_success"}
        for part in content.parts or []
        for response in [part.function_response]
    )


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
    if latest_response_index is None or not any(
        _is_successful_record(content)
        for content in contents[:latest_response_index]
    ):
        return None

    for index in range(latest_response_index - 1, -1, -1):
        for part in contents[index].parts or []:
            call = part.function_call
            if (
                call is not None
                and call.name == latest_response_name
                and (latest_response_id is None or call.id == latest_response_id)
            ):
                return index
    return latest_response_index


def filter_recorded_multi_kpi_context(
    contents: list[types.Content],
) -> list[types.Content]:
    """Compact completed retrieval batches while retaining the active source pages.

    This function is supplied to ADK's ``ContextFilterPlugin``. It changes only
    the model request copy: immutable report text in session state and the
    session event history remain untouched. Function call/response identities
    are retained so OpenAI-compatible tool history remains valid. A retrieval
    remains visible across every progress-record batch that consumes it and is
    compacted only after a newer retrieval starts.
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
