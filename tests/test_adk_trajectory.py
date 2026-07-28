import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from finground.benchmark.adk_trajectory import AdkTrajectoryPlugin

TRACE_SET = Path(__file__).parent / "fixtures" / "ledger" / "eval-2017-trace-20.txt"


def test_2017_trace_set_contains_twenty_fixed_reports() -> None:
    report_ids = [
        line.strip()
        for line in TRACE_SET.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(report_ids) == 20
    assert len(set(report_ids)) == 20
    assert all(report_id.endswith("_2017") for report_id in report_ids)
    assert sum(report_id.startswith("NYSE_") for report_id in report_ids) == 11
    assert sum(report_id.startswith("NASDAQ_") for report_id in report_ids) == 8
    assert sum(report_id.startswith("AMEX_") for report_id in report_ids) == 1


def test_adk_trajectory_plugin_records_full_lifecycle(tmp_path: Path) -> None:
    trace_path = tmp_path / "NYSE_ACME_2017.jsonl"
    plugin = AdkTrajectoryPlugin(trace_path)
    callback_context = SimpleNamespace(
        invocation_id="inv-1",
        agent_name="multi_kpi",
        function_call_id="call-1",
    )
    invocation_context = SimpleNamespace(
        invocation_id="inv-1",
        agent=SimpleNamespace(name="multi_kpi"),
        session=SimpleNamespace(id="session-1", user_id="benchmark", app_name="app"),
    )
    request = LlmRequest(
        model="qwen",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Extract all KPIs")],
            )
        ],
    )
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-1",
                        name="get_report_info",
                        args={},
                    )
                )
            ],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=10,
            total_token_count=110,
        ),
    )
    event = Event(author="multi_kpi", content=response.content)
    tool = SimpleNamespace(name="get_report_info")

    async def record() -> None:
        await plugin.before_run_callback(invocation_context=invocation_context)
        await plugin.before_model_callback(
            callback_context=callback_context,
            llm_request=request,
        )
        await plugin.after_model_callback(
            callback_context=callback_context,
            llm_response=response,
        )
        await plugin.before_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=callback_context,
        )
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=callback_context,
            result={"status": "success", "page_count": 80},
        )
        await plugin.on_event_callback(
            invocation_context=invocation_context,
            event=event,
        )
        await plugin.after_run_callback(invocation_context=invocation_context)

    asyncio.run(record())
    plugin.finish(
        outcome="ok",
        summary={"coverage_count": 31, "llm_calls": 20},
    )

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == list(range(1, len(records) + 1))
    assert [record["kind"] for record in records] == [
        "before_run",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "event",
        "after_run",
        "trajectory_finished",
    ]
    assert records[1]["payload"]["request"]["contents"][0]["parts"][0]["text"] == (
        "Extract all KPIs"
    )
    assert records[2]["payload"]["response"]["usage_metadata"]["prompt_token_count"] == 100
    assert records[4]["payload"]["result"] == {"status": "success", "page_count": 80}
    assert records[5]["payload"]["event"]["author"] == "multi_kpi"
    assert records[-1]["payload"]["summary"]["coverage_count"] == 31
    assert plugin.snapshot() == {
        "path": str(trace_path),
        "record_count": 8,
        "write_error": None,
        "complete": True,
    }


def test_adk_trajectory_plugin_preserves_partial_file_until_finished(tmp_path: Path) -> None:
    trace_path = tmp_path / "NYSE_ACME_2017.jsonl"
    plugin = AdkTrajectoryPlugin(trace_path)

    asyncio.run(
        plugin.on_model_error_callback(
            callback_context=SimpleNamespace(
                invocation_id="inv-1",
                agent_name="multi_kpi",
                function_call_id=None,
            ),
            llm_request=LlmRequest(model="qwen"),
            error=RuntimeError("remote disconnected"),
        )
    )

    assert not trace_path.exists()
    assert plugin.partial_path.exists()
    partial_record = json.loads(plugin.partial_path.read_text(encoding="utf-8"))
    assert partial_record["kind"] == "model_error"
    assert partial_record["payload"]["error"] == {
        "type": "RuntimeError",
        "message": "remote disconnected",
    }

    plugin.finish(outcome="failed", summary={"error": "remote disconnected"})

    assert trace_path.exists()
    assert not plugin.partial_path.exists()
