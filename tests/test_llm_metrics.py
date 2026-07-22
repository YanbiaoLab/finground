import asyncio

from google.adk.models.llm_request import LlmRequest

from finground.benchmark.llm_metrics import LlmCallCounterPlugin


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
    deadline_request = LlmRequest()

    async def count() -> None:
        await counter.before_model_callback(
            callback_context=None,
            llm_request=first_request,
        )
        await counter.before_model_callback(
            callback_context=None,
            llm_request=deadline_request,
        )

    asyncio.run(count())

    assert first_request.config.tool_config is None
    function_config = deadline_request.config.tool_config.function_calling_config
    assert function_config.mode == "ANY"
    assert function_config.allowed_function_names == ["submit_multi_kpi_extraction"]
