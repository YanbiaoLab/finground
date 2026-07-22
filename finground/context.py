"""ADK-native context reduction for state-backed Multi-KPI extraction."""

from __future__ import annotations

import copy

from google.genai import types

MULTI_KPI_RECORD_TOOL = "record_multi_kpi_progress"
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


def _successful_record_index(contents: list[types.Content]) -> int | None:
    latest: int | None = None
    for index, content in enumerate(contents):
        for part in content.parts or []:
            response = part.function_response
            if (
                response is not None
                and response.name == MULTI_KPI_RECORD_TOOL
                and isinstance(response.response, dict)
                and response.response.get("status") == "success"
            ):
                latest = index
    return latest


def filter_recorded_multi_kpi_context(
    contents: list[types.Content],
) -> list[types.Content]:
    """Remove bulky tool payloads preceding the latest successful progress record.

    This function is supplied to ADK's ``ContextFilterPlugin``. It changes only
    the model request copy: immutable report text in session state and the
    session event history remain untouched. Function call/response identities
    are retained so OpenAI-compatible tool history remains valid.
    """
    record_index = _successful_record_index(contents)
    if record_index is None:
        return contents

    filtered = copy.deepcopy(contents)
    for content in filtered[:record_index]:
        for part in content.parts or []:
            call = part.function_call
            if call is not None and call.name in _COMPACTABLE_TOOL_NAMES:
                call.args = {"compacted": True}
            response = part.function_response
            if response is not None and response.name in _COMPACTABLE_TOOL_NAMES:
                response.response = dict(_COMPACTED_RESPONSE)
    return filtered
