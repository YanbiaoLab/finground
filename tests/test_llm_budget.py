import asyncio
from types import SimpleNamespace

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from finground.agents.kpi_specialists import (
    COMMON_TASK_AGENT_NAME,
    KPI_DISPATCH_TOOL_NAME,
    MULTI_KPI_COORDINATOR_NAME,
    kpi_agent_name,
)
from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin
from finground.kpis import KPI_KEYS
from finground.tools import (
    MULTI_KPI_PREPARED_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    REPORT_STATE_KEY,
)


def _request_with_agent_tools() -> LlmRequest:
    return LlmRequest(
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=name)
                        for name in (
                            COMMON_TASK_AGENT_NAME,
                            *(kpi_agent_name(kpi) for kpi in KPI_KEYS),
                        )
                    ]
                )
            ]
        )
    )


def _context(
    *,
    agent_name: str = MULTI_KPI_COORDINATOR_NAME,
    prepared: bool = False,
    covered: tuple[str, ...] = (),
) -> SimpleNamespace:
    state = {
        REPORT_STATE_KEY: {
            "report_id": "NYSE_ACME_2023",
            "ticker": "ACME",
            "year": 2023,
        },
        MULTI_KPI_WORK_RECORD_STATE_KEY: {
            "kpis": [{"kpi": kpi, "fiscal_year": 2023, "status": "absent"} for kpi in covered]
        },
    }
    if prepared:
        state[MULTI_KPI_PREPARED_STATE_KEY] = {"status": "success"}
    return SimpleNamespace(state=state, agent_name=agent_name)


def _allowed_tools(request: LlmRequest) -> list[str] | None:
    if request.config.tool_config is None:
        return None
    return request.config.tool_config.function_calling_config.allowed_function_names


def test_guard_routes_first_coordinator_call_to_common_preparation() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    request = _request_with_agent_tools()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(),
            llm_request=request,
        )
    )

    assert _allowed_tools(request) == [COMMON_TASK_AGENT_NAME]
    assert "prepared=False" in request.contents[-1].parts[0].text
    assert f"next_action={COMMON_TASK_AGENT_NAME} prepare" in request.contents[-1].parts[0].text


def test_guard_exposes_only_pending_specialists_after_preparation() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    request = _request_with_agent_tools()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(prepared=True, covered=("revenue",)),
            llm_request=request,
        )
    )

    allowed = _allowed_tools(request)
    assert kpi_agent_name("revenue") not in allowed
    assert allowed == [KPI_DISPATCH_TOOL_NAME]
    assert COMMON_TASK_AGENT_NAME not in allowed


def test_guard_routes_complete_coverage_to_common_finalization() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    request = _request_with_agent_tools()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(prepared=True, covered=KPI_KEYS),
            llm_request=request,
        )
    )

    assert _allowed_tools(request) == [COMMON_TASK_AGENT_NAME]
    assert f"next_action={COMMON_TASK_AGENT_NAME} finalize" in request.contents[-1].parts[0].text


def test_guard_does_not_restrict_specialist_model_tools() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    request = LlmRequest()

    asyncio.run(
        plugin.before_model_callback(
            callback_context=_context(agent_name=kpi_agent_name("revenue")),
            llm_request=request,
        )
    )

    assert request.config.tool_config is None
    assert request.contents == []


def test_guard_adds_budget_reminders_on_next_coordinator_turn() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=10)
    context = _context(prepared=True)
    requests = [_request_with_agent_tools() for _ in range(8)]

    async def run_callbacks() -> None:
        for request in requests:
            await plugin.before_model_callback(
                callback_context=context,
                llm_request=request,
            )

    asyncio.run(run_callbacks())

    assert "60% USED" in requests[5].contents[-1].parts[0].text
    assert "80% USED" in requests[7].contents[-1].parts[0].text


def test_guard_replaces_premature_coordinator_text_with_pending_specialist() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    context = _context(prepared=True, covered=("revenue",))
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Finished.")],
        )
    )

    guarded = asyncio.run(
        plugin.after_model_callback(
            callback_context=context,
            llm_response=response,
        )
    )

    function_call = guarded.get_function_calls()[0]
    assert function_call.name == KPI_DISPATCH_TOOL_NAME
    assert function_call.args["kpis"] == list(KPI_KEYS[1:])
    assert function_call.args == {
        "kpis": list(KPI_KEYS[1:]),
        "request": "Find, validate, and checkpoint every supplied pending KPI.",
    }
    assert plugin.prevented_early_stops == 1


def test_guard_preserves_specialist_final_text() -> None:
    plugin = MultiKpiExecutionGuardPlugin(max_calls=200)
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Specialist result.")],
        )
    )

    guarded = asyncio.run(
        plugin.after_model_callback(
            callback_context=_context(agent_name=kpi_agent_name("revenue")),
            llm_response=response,
        )
    )

    assert guarded is None
    assert plugin.prevented_early_stops == 0
