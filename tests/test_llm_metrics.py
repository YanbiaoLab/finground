import asyncio
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from finground.benchmark.llm_metrics import LlmCallCounterPlugin, MultiKpiRunMetricsPlugin
from finground.tools import MULTI_KPI_ALLOW_PARTIAL_STATE_KEY


def test_llm_call_counter_counts_each_model_request_attempt() -> None:
    counter = LlmCallCounterPlugin()

    async def count() -> None:
        await counter.before_model_callback(callback_context=None, llm_request=None)
        await counter.before_model_callback(callback_context=None, llm_request=None)

    asyncio.run(count())

    assert counter.count == 2


def test_llm_call_counter_keeps_actual_count_beyond_configured_limit() -> None:
    counter = LlmCallCounterPlugin(max_calls=2)

    async def count() -> None:
        for _ in range(3):
            await counter.before_model_callback(callback_context=None, llm_request=None)

    asyncio.run(count())

    assert counter.count == 3


def test_llm_call_counter_forces_final_tool_at_submission_deadline() -> None:
    counter = LlmCallCounterPlugin(
        max_calls=3,
        force_tool_at_call=2,
        forced_tool_name="submit_multi_kpi_extraction",
    )
    first_request = LlmRequest()
    deadline_request = LlmRequest(
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="search_report"),
                        types.FunctionDeclaration(name="submit_multi_kpi_extraction"),
                    ]
                )
            ]
        )
    )
    context = SimpleNamespace(state={})

    async def count() -> None:
        await counter.before_model_callback(
            callback_context=context,
            llm_request=first_request,
        )
        await counter.before_model_callback(
            callback_context=context,
            llm_request=deadline_request,
        )

    asyncio.run(count())

    assert first_request.config.tool_config is None
    function_config = deadline_request.config.tool_config.function_calling_config
    assert function_config.mode == "ANY"
    assert function_config.allowed_function_names == ["submit_multi_kpi_extraction"]
    declarations = deadline_request.config.tools[0].function_declarations
    assert [declaration.name for declaration in declarations] == ["submit_multi_kpi_extraction"]
    assert context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] is True


def test_multi_kpi_run_metrics_tracks_tool_validation_and_tokens() -> None:
    metrics = MultiKpiRunMetricsPlugin()
    tool = SimpleNamespace(name="record_multi_kpi_progress")
    response = LlmResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12_000,
            candidates_token_count=500,
            total_token_count=12_500,
        )
    )

    async def collect() -> None:
        await metrics.after_model_callback(callback_context=None, llm_response=response)
        await metrics.after_tool_callback(
            tool=tool,
            tool_args={"kpis": [{}, {}]},
            tool_context=None,
            result={
                "status": "partial_success",
                "added_kpi_count": 1,
                "updated_kpi_count": 0,
                "pending_count": 30,
                "validation_errors": [{"field": "kpis.1.value_verbatim", "message": "wrong year"}],
            },
        )
        await metrics.after_tool_callback(
            tool=tool,
            tool_args={"kpis": [{}]},
            tool_context=None,
            result={
                "status": "error",
                "retryable": True,
                "validation_errors": [{"field": "kpis.0.value_verbatim", "message": "wrong year"}],
            },
        )

    asyncio.run(collect())

    assert metrics.snapshot() == {
        "model_tokens": {
            "prompt_total": 12_000,
            "prompt_max": 12_000,
            "candidate_total": 500,
            "total": 12_500,
        },
        "tool_calls": {"record_multi_kpi_progress": 2},
        "tool_statuses": {
            "record_multi_kpi_progress:partial_success": 1,
            "record_multi_kpi_progress:error": 1,
        },
        "validation_error_count": 2,
        "retryable_error_calls": 1,
        "partial_success_calls": 1,
        "repeated_validation_error_calls": 1,
        "saved_kpi_rows": 1,
        "latest_pending_count": 30,
        "tool_exception_count": 0,
    }


def test_multi_kpi_run_metrics_accepts_agent_tool_text_results() -> None:
    metrics = MultiKpiRunMetricsPlugin()

    asyncio.run(
        metrics.after_tool_callback(
            tool=SimpleNamespace(name="extract_revenue"),
            tool_args={"request": "find revenue"},
            tool_context=None,
            result='{"status":"success","coverage_count":1}',
        )
    )

    assert metrics.snapshot()["tool_calls"] == {"extract_revenue": 1}
    assert metrics.snapshot()["tool_statuses"] == {"extract_revenue:completed": 1}
