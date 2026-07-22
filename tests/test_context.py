from google.genai import types

from finground.context import filter_recorded_multi_kpi_context


def _call(name: str, call_id: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))],
    )


def _response(name: str, call_id: str, response: dict) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response=response,
                )
            )
        ],
    )


def test_context_filter_removes_old_tool_payloads_after_successful_record() -> None:
    large_page = "financial table " * 5_000
    contents = [
        types.Content(role="user", parts=[types.Part(text="Extract KPIs")]),
        _call("read_report_pages", "read-1", {"page_numbers": [10, 11]}),
        _response(
            "read_report_pages",
            "read-1",
            {"status": "success", "pages": [{"page": 10, "text": large_page}]},
        ),
        _call(
            "record_multi_kpi_progress",
            "record-1",
            {
                "reporting_currency": "USD",
                "units_note": "millions",
                "kpis": [{"kpi": "revenue", "fiscal_year": 2023, "value": 1.0}],
                "notes": [{"category": "evidence", "text": "Revenue table", "pages": [10]}],
            },
        ),
        _response(
            "record_multi_kpi_progress",
            "record-1",
            {"status": "success", "kpi_count": 1, "note_count": 1},
        ),
        _call("read_report_pages", "read-2", {"page_numbers": [20]}),
        _response(
            "read_report_pages",
            "read-2",
            {"status": "success", "pages": [{"page": 20, "text": "current table"}]},
        ),
    ]

    filtered = filter_recorded_multi_kpi_context(contents)

    assert large_page in str(contents[2].parts[0].function_response.response)
    assert large_page not in str(filtered)
    assert filtered[1].parts[0].function_call.args == {"compacted": True}
    assert filtered[2].parts[0].function_response.response["status"] == "compacted"
    assert filtered[6].parts[0].function_response.response["pages"][0]["text"] == "current table"


def test_context_filter_keeps_history_until_progress_is_recorded() -> None:
    contents = [
        types.Content(role="user", parts=[types.Part(text="Extract KPIs")]),
        _call("read_report_pages", "read-1", {"page_numbers": [10]}),
        _response(
            "read_report_pages",
            "read-1",
            {"status": "success", "pages": [{"page": 10, "text": "table"}]},
        ),
    ]

    assert filter_recorded_multi_kpi_context(contents) is contents
