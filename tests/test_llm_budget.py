import asyncio

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin


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
    assert 'query_multi_kpi_progress(view="kpis")' in final_message
    assert "next model call must call submit_multi_kpi_extraction" in final_message
    assert "persisted as incomplete" in final_message


def test_multi_kpi_budget_plugin_rounds_thresholds_up() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=31)

    assert plugin.first_reminder_call == 19
    assert plugin.final_warning_call == 25


def test_multi_kpi_guard_replaces_premature_text_with_progress_query() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=30)
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="I have finished the analysis.")],
        )
    )

    guarded = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=response)
    )

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

    guarded = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=response)
    )

    assert guarded is None
    assert plugin.prevented_early_stops == 0
