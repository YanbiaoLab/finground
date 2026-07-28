import asyncio
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from finground.benchmark.llm_metrics import LlmCallCounterPlugin
from finground.tools import MULTI_KPI_ALLOW_PARTIAL_STATE_KEY


def test_llm_call_counter_counts_each_model_request_attempt() -> None:
    counter = LlmCallCounterPlugin()

    async def count() -> None:
        await counter.before_model_callback(callback_context=None, llm_request=None)
        await counter.before_model_callback(callback_context=None, llm_request=None)

    asyncio.run(count())

    assert counter.count == 2


def test_llm_call_counter_does_not_count_calls_rejected_by_adk_limit() -> None:
    counter = LlmCallCounterPlugin(max_calls=2)

    async def count() -> None:
        for _ in range(3):
            await counter.before_model_callback(callback_context=None, llm_request=None)

    asyncio.run(count())

    assert counter.count == 2


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
    assert [declaration.name for declaration in declarations] == [
        "submit_multi_kpi_extraction"
    ]
    assert context.state[MULTI_KPI_ALLOW_PARTIAL_STATE_KEY] is True
