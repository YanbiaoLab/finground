import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.agents.kpi_specialists import (
    COMMON_TASK_AGENT_NAME,
    KPI_DISPATCH_TOOL_NAME,
    create_kpi_specialist_agent,
)
from finground.agents.multi_kpi import MULTI_KPI_APP_NAME, create_multi_kpi_app
from finground.benchmark.llm_budget import MultiKpiExecutionGuardPlugin
from finground.benchmark.llm_metrics import (
    LlmCallCounterPlugin,
    MultiKpiRunMetricsPlugin,
)
from finground.documents import Report
from finground.kpis import KPI_KEYS
from finground.tools import (
    MULTI_KPI_RESULT_STATE_KEY,
    MULTI_KPI_WORK_RECORD_STATE_KEY,
    REPORT_STATE_KEY,
    build_report_state,
)

REPORT = Path(__file__).parent / "fixtures" / "ledger" / "report.mmd"
pytestmark = pytest.mark.filterwarnings(
    "ignore:The `plugins` argument is deprecated.*:DeprecationWarning"
)


class ScriptedLlm(BaseLlm):
    responses: list[types.Content]

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        if not self.responses:
            raise RuntimeError(f"no scripted response left for {self.model}")
        yield LlmResponse(content=self.responses.pop(0))


def _tool_call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name=name,
                    args=args,
                )
            )
        ],
    )


def test_isolated_agent_tools_forward_state_and_finalize_parent_run() -> None:
    async def run() -> tuple[dict, list, int, dict]:
        counter = LlmCallCounterPlugin(max_calls=200)
        guard = MultiKpiExecutionGuardPlugin(max_calls=200)
        metrics = MultiKpiRunMetricsPlugin()
        app = create_multi_kpi_app(plugins=[counter, guard, metrics])
        root = app.root_agent
        tools = {tool.name: tool for tool in root.tools}
        root.model = ScriptedLlm(
            model="root-script",
            responses=[
                _tool_call(COMMON_TASK_AGENT_NAME, {"request": "prepare"}),
                _tool_call(
                    KPI_DISPATCH_TOOL_NAME,
                    {"kpis": ["revenue"], "request": "find and record revenue"},
                ),
                _tool_call(COMMON_TASK_AGENT_NAME, {"request": "finalize"}),
            ],
        )
        tools[COMMON_TASK_AGENT_NAME].agent.model = ScriptedLlm(
            model="common-script",
            responses=[
                _tool_call("prepare_multi_kpi_report", {}),
                _tool_call("finalize_multi_kpi_report", {}),
            ],
        )
        tools[KPI_DISPATCH_TOOL_NAME]._specialists["revenue"].agent.model = ScriptedLlm(
            model="revenue-script",
            responses=[
                _tool_call("find_revenue_candidates", {}),
                _tool_call(
                    "record_multi_kpi_progress",
                    {
                        "reporting_currency": "USD",
                        "units_note": "(in millions, except per-share amounts)",
                        "kpis": [
                            {
                                "kpi": "revenue",
                                "fiscal_year": 2023,
                                "status": "found",
                                "source_id": "p3:t0:r1:c1",
                            }
                        ],
                        "notes": [],
                    },
                ),
            ],
        )

        report = Report(
            "NYSE_ACME_2023",
            "NYSE",
            "ACME",
            2023,
            REPORT.read_text(),
        )
        state = {
            REPORT_STATE_KEY: build_report_state(report),
            MULTI_KPI_WORK_RECORD_STATE_KEY: {
                "ticker": "ACME",
                "reporting_currency": None,
                "units_note": None,
                "kpis": [
                    {
                        "kpi": kpi,
                        "fiscal_year": 2023,
                        "status": "absent",
                        "value": None,
                        "normalization": None,
                    }
                    for kpi in KPI_KEYS
                    if kpi != "revenue"
                ],
                "notes": [],
            },
        }
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=MULTI_KPI_APP_NAME,
            user_id="test",
            session_id="runtime",
            state=state,
        )
        runner = Runner(
            app=app,
            session_service=sessions,
        )
        events = [
            event
            async for event in runner.run_async(
                user_id="test",
                session_id="runtime",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="extract")],
                ),
            )
        ]
        session = await sessions.get_session(
            app_name=MULTI_KPI_APP_NAME,
            user_id="test",
            session_id="runtime",
        )
        return (
            session.state[MULTI_KPI_RESULT_STATE_KEY],
            events,
            counter.count,
            metrics.snapshot(),
        )

    result, events, llm_calls, metrics = asyncio.run(run())

    assert result["kpis"] == [
        {
            "kpi": "revenue",
            "fiscal_year": 2023,
            "value": 1_234_000_000.0,
        }
    ]
    assert events[-1].is_final_response()
    assert events[-1].actions.skip_summarization is True
    assert llm_calls == 7
    assert metrics["tool_calls"]["manage_report_workflow"] == 2
    assert metrics["tool_calls"][KPI_DISPATCH_TOOL_NAME] == 1


def test_specialist_repeated_budget_exhaustion_records_ambiguous_and_stops() -> None:
    specialist = create_kpi_specialist_agent("revenue", max_output_tokens=4_096)
    callback = specialist.canonical_before_tool_callbacks[0]
    report = Report(
        "NYSE_ACME_2023",
        "NYSE",
        "ACME",
        2023,
        REPORT.read_text(),
    )
    context = SimpleNamespace(
        state={REPORT_STATE_KEY: build_report_state(report)},
        actions=SimpleNamespace(skip_summarization=False),
    )
    tool = SimpleNamespace(name="find_revenue_candidates")

    async def exhaust() -> list[dict | None]:
        return [await callback(tool=tool, args={}, tool_context=context) for _index in range(3)]

    results = asyncio.run(exhaust())

    assert results[0] is None
    assert results[1]["status"] == "error"
    assert results[2]["status"] == "success"
    assert "repeated find_revenue_candidates" in results[2]["fallback_reason"]
    assert context.actions.skip_summarization is True
    assert context.state[MULTI_KPI_WORK_RECORD_STATE_KEY]["kpis"][0]["status"] == ("ambiguous")


def test_specialist_eighth_non_record_response_is_forced_to_ambiguous_checkpoint() -> None:
    specialist = create_kpi_specialist_agent("revenue", max_output_tokens=4_096)
    callback = specialist.canonical_after_model_callbacks[0]
    context = SimpleNamespace(
        state={
            REPORT_STATE_KEY: {
                "report_id": "NYSE_ACME_2023",
                "ticker": "ACME",
                "year": 2023,
            }
        }
    )
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="I need more evidence.")],
        )
    )

    async def exhaust() -> list[LlmResponse | None]:
        return [
            await callback(callback_context=context, llm_response=response) for _index in range(8)
        ]

    results = asyncio.run(exhaust())
    forced_call = results[-1].get_function_calls()[0]

    assert results[:3] == [None, None, None]
    assert forced_call.name == "record_multi_kpi_progress"
    assert forced_call.args["kpis"] == [
        {
            "kpi": "revenue",
            "fiscal_year": 2023,
            "status": "ambiguous",
        }
    ]
