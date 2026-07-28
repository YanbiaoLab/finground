import asyncio
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin
from finground.tools import MULTI_KPI_WORK_RECORD_STATE_KEY, REPORT_STATE_KEY


def _request_with_tools() -> LlmRequest:
    return LlmRequest(
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=name)
                        for name in (
                            "get_report_info",
                            "search_report",
                            "read_report_pages",
                            "record_multi_kpi_progress",
                            "query_multi_kpi_progress",
                            "submit_multi_kpi_extraction",
                        )
                    ]
                )
            ]
        )
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        state={
            REPORT_STATE_KEY: {
                "report_id": "NYSE_ACME_2023",
                "ticker": "ACME",
                "year": 2023,
            }
        }
    )


def _allowed_tools(request: LlmRequest) -> list[str] | None:
    return request.config.tool_config.function_calling_config.allowed_function_names


def test_multi_kpi_budget_plugin_adds_supplemental_messages_at_thresholds() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=30)
    requests = [LlmRequest() for _ in range(24)]

    async def run_callbacks() -> None:
        for request in requests:
            await plugin.before_model_callback(
                callback_context=None,
                llm_request=request,
            )

    asyncio.run(run_callbacks())

    assert all(not request.contents for request in requests[:17])
    first_message = requests[17].contents[-1].parts[0].text
    assert "60% USED" in first_message
    assert "record_multi_kpi_progress" in first_message
    assert "Before any more retrieval" in first_message
    assert all(not request.contents for request in requests[18:23])
    final_message = requests[23].contents[-1].parts[0].text
    assert "80% USED" in final_message
    assert "closure-only" in final_message
    assert "no new retrieval" in final_message
    assert "call 30" in final_message


def test_multi_kpi_budget_plugin_rounds_thresholds_up() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=31)

    assert plugin.first_reminder_call == 19
    assert plugin.final_warning_call == 25


def test_multi_kpi_guard_requires_report_info_on_first_call() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50)
    request = _request_with_tools()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(),
            llm_request=request,
        )
    )

    assert _allowed_tools(request) == ["get_report_info"]
    assert "phase=metadata" in request.contents[-1].parts[0].text


def test_multi_kpi_guard_requires_checkpoint_after_reading_pages() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50)
    context = _context()

    async def run_callbacks() -> LlmRequest:
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="get_report_info"),
            tool_args={},
            tool_context=context,
            result={"status": "success"},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args={"page_numbers": [10], "focus_phrases": []},
            tool_context=context,
            result={"status": "success", "pages": [{"page": 10, "text": "table"}]},
        )
        request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=request,
        )
        return request

    request = asyncio.run(run_callbacks())

    assert _allowed_tools(request) == ["record_multi_kpi_progress"]
    assert "phase=checkpoint" in request.contents[-1].parts[0].text
    assert "pages=[10]" in request.contents[-1].parts[0].text


def test_multi_kpi_guard_blocks_duplicate_read_until_coverage_progresses() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50)
    context = _context()
    args = {"page_numbers": [10], "focus_phrases": []}

    async def run_callbacks() -> tuple[dict | None, LlmRequest, dict | None]:
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="get_report_info"),
            tool_args={},
            tool_context=context,
            result={"status": "success"},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args=args,
            tool_context=context,
            result={"status": "success", "pages": [{"page": 10, "text": "table"}]},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args={},
            tool_context=context,
            result={"status": "success", "coverage_count": 0, "added_kpi_count": 0},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="search_report"),
            tool_args={"query": "balance sheet"},
            tool_context=context,
            result={"status": "success", "results": [{"page": 10}]},
        )
        blocked = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args=args,
            tool_context=context,
        )
        recovery_request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=recovery_request,
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args={},
            tool_context=context,
            result={"status": "success", "coverage_count": 8, "added_kpi_count": 8},
        )
        allowed = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args=args,
            tool_context=context,
        )
        return blocked, recovery_request, allowed

    blocked, recovery_request, allowed = asyncio.run(run_callbacks())

    assert blocked["status"] == "error"
    assert blocked["error"] == "duplicate retrieval blocked"
    assert blocked["next_action"] == "choose a new source or query_multi_kpi_progress"
    assert _allowed_tools(recovery_request) == [
        "search_report",
        "record_multi_kpi_progress",
        "query_multi_kpi_progress",
    ]
    assert any(
        "phase=duplicate_recovery" in content.parts[0].text for content in recovery_request.contents
    )
    assert allowed is None


def test_multi_kpi_guard_releases_checkpoint_after_one_failed_repair() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50)
    context = _context()
    repair_result = {
        "status": "partial_success",
        "coverage_count": 8,
        "added_kpi_count": 0,
        "repair_queue": [
            {
                "index": 0,
                "kpi": "revenue",
                "validation_errors": [
                    {
                        "field": "kpis.0.line_label",
                        "message": "number was not found on the cited labelled row",
                    }
                ],
            }
        ],
    }

    async def run_callbacks() -> tuple[LlmRequest, LlmRequest]:
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="get_report_info"),
            tool_args={},
            tool_context=context,
            result={"status": "success"},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args={"page_numbers": [10], "focus_phrases": []},
            tool_context=context,
            result={"status": "success", "pages": [{"page": 10, "text": "table"}]},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args={},
            tool_context=context,
            result=repair_result,
        )
        repair_request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=repair_request,
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args={},
            tool_context=context,
            result=repair_result,
        )
        released_request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=released_request,
        )
        return repair_request, released_request

    repair_request, released_request = asyncio.run(run_callbacks())

    assert _allowed_tools(repair_request) == ["record_multi_kpi_progress"]
    assert _allowed_tools(released_request) == [
        "search_report",
        "record_multi_kpi_progress",
        "query_multi_kpi_progress",
    ]
    assert any(
        "phase=repair_exhausted" in content.parts[0].text
        and "repair_exhausted" in content.parts[0].text
        for content in released_request.contents
    )


def test_multi_kpi_guard_rotates_away_from_duplicate_progress_record() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50)
    context = _context()
    args = {
        "reporting_currency": "USD",
        "units_note": None,
        "kpis": [
            {
                "kpi": "total_liabilities",
                "fiscal_year": 2023,
                "status": "ambiguous",
            }
        ],
        "notes": [],
    }

    async def run_callbacks() -> tuple[dict | None, LlmRequest, dict | None]:
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="get_report_info"),
            tool_args={},
            tool_context=context,
            result={"status": "success"},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="read_report_pages"),
            tool_args={"page_numbers": [10], "focus_phrases": []},
            tool_context=context,
            result={"status": "success", "pages": [{"page": 10, "text": "table"}]},
        )
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args=args,
            tool_context=context,
            result={
                "status": "partial_success",
                "coverage_count": 1,
                "added_kpi_count": 1,
                "repair_queue": [
                    {
                        "index": 0,
                        "kpi": "total_liabilities",
                        "validation_errors": [
                            {
                                "field": "kpis.0.line_label",
                                "message": "label mismatch",
                            }
                        ],
                    }
                ],
            },
        )
        blocked = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="record_multi_kpi_progress"),
            tool_args=args,
            tool_context=context,
        )
        recovery_request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=recovery_request,
        )
        search_allowed = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="search_report"),
            tool_args={"query": "cash flow statement"},
            tool_context=context,
        )
        return blocked, recovery_request, search_allowed

    blocked, recovery_request, search_allowed = asyncio.run(run_callbacks())

    assert blocked["status"] == "error"
    assert blocked["error"] == "duplicate progress action blocked"
    assert _allowed_tools(recovery_request) == [
        "search_report",
        "query_multi_kpi_progress",
    ]
    assert search_allowed is None


def test_multi_kpi_guard_enforces_total_search_budget() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=50, max_searches=2)
    context = _context()

    async def run_callbacks() -> tuple[dict | None, LlmRequest]:
        await plugin.after_tool_callback(
            tool=SimpleNamespace(name="get_report_info"),
            tool_args={},
            tool_context=context,
            result={"status": "success"},
        )
        for index in range(2):
            await plugin.after_tool_callback(
                tool=SimpleNamespace(name="search_report"),
                tool_args={"query": f"source {index}"},
                tool_context=context,
                result={"status": "success", "results": []},
            )
        blocked = await plugin.before_tool_callback(
            tool=SimpleNamespace(name="search_report"),
            tool_args={"query": "one more source"},
            tool_context=context,
        )
        recovery_request = _request_with_tools()
        await plugin.before_model_callback(
            callback_context=context,
            llm_request=recovery_request,
        )
        return blocked, recovery_request

    blocked, recovery_request = asyncio.run(run_callbacks())

    assert blocked["error"] == "search budget exhausted"
    assert blocked["search_count"] == 2
    assert _allowed_tools(recovery_request) == [
        "record_multi_kpi_progress",
        "query_multi_kpi_progress",
    ]


def test_multi_kpi_guard_restricts_closure_to_progress_and_submission_tools() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=10)
    plugin.call_count = 7
    request = _request_with_tools()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(),
            llm_request=request,
        )
    )

    assert plugin.call_count == 8
    assert _allowed_tools(request) == [
        "record_multi_kpi_progress",
        "query_multi_kpi_progress",
        "submit_multi_kpi_extraction",
    ]
    assert any("phase=closure" in content.parts[0].text for content in request.contents)


def test_multi_kpi_guard_replaces_premature_text_with_progress_query() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=30)
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="I have finished the analysis.")],
        )
    )

    guarded = asyncio.run(plugin.after_model_callback(callback_context=None, llm_response=response))

    assert guarded is not None
    function_call = guarded.get_function_calls()[0]
    assert function_call.name == "query_multi_kpi_progress"
    assert function_call.args == {"view": "kpis"}
    assert plugin.prevented_early_stops == 1


def test_multi_kpi_guard_preserves_model_tool_calls() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=30)
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="search_report",
                        args={"query": "revenue"},
                    )
                )
            ],
        )
    )

    guarded = asyncio.run(plugin.after_model_callback(callback_context=None, llm_response=response))

    assert guarded is None
    assert plugin.prevented_early_stops == 0


def test_multi_kpi_guard_forces_state_backed_submission_on_last_call() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=30)
    plugin.call_count = 30
    context = SimpleNamespace(
        state={
            REPORT_STATE_KEY: {"ticker": "ACME"},
            MULTI_KPI_WORK_RECORD_STATE_KEY: {
                "reporting_currency": "USD",
                "units_note": "(in millions)",
            },
        }
    )
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="I have finished the analysis.")],
        )
    )

    guarded = asyncio.run(
        plugin.after_model_callback(callback_context=context, llm_response=response)
    )

    function_call = guarded.get_function_calls()[0]
    assert function_call.name == "submit_multi_kpi_extraction"
    assert function_call.args == {
        "ticker": "ACME",
        "reporting_currency": "USD",
        "units_note": "(in millions)",
        "kpis": [],
    }


def test_multi_kpi_guard_injects_exact_state_backed_args_on_last_request() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=3)
    context = SimpleNamespace(
        state={
            REPORT_STATE_KEY: {"ticker": "ACME"},
            MULTI_KPI_WORK_RECORD_STATE_KEY: {
                "reporting_currency": "USD",
                "units_note": None,
            },
        }
    )
    requests = [LlmRequest() for _ in range(3)]

    async def run_callbacks() -> None:
        for request in requests:
            await plugin.before_model_callback(
                callback_context=context,
                llm_request=request,
            )

    asyncio.run(run_callbacks())

    final_message = requests[-1].contents[-1].parts[0].text
    assert "[FINAL CALL]" in final_message
    assert '"ticker": "ACME"' in final_message
    assert '"reporting_currency": "USD"' in final_message
    assert '"kpis": []' in final_message
