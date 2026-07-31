import ast
import json
from pathlib import Path
from types import SimpleNamespace

from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool

import finground.agents as agents_package
from finground.agents.common import create_adk_model
from finground.agents.kpi_specialists import (
    COMMON_TASK_AGENT_NAME,
    KPI_AGENT_SPECS,
    KPI_DISPATCH_TOOL_NAME,
    MULTI_KPI_COORDINATOR_NAME,
    KpiSpecialistTool,
    kpi_agent_name,
)
from finground.agents.multi_kpi import (
    KPI_AGENT_NAMES,
    MULTI_KPI_CONTEXT_WINDOW_TOKENS,
    MULTI_KPI_FINAL_WARNING_CALL,
    MULTI_KPI_LLM_CALL_LIMIT,
    MULTI_KPI_MAX_OUTPUT_TOKENS,
    MULTI_KPI_PROGRESS_REMINDER_CALL,
    MULTI_KPI_SEARCH_LIMIT,
    MULTI_KPI_SUBMISSION_DEADLINE,
    create_multi_kpi_agent,
    create_multi_kpi_app,
    resolve_requested_kpis,
)
from finground.config import load_settings
from finground.kpis import KPI_KEYS


def test_agent_core_does_not_import_benchmark_package() -> None:
    package_dir = Path(agents_package.__file__).parent
    for path in package_dir.rglob("*.py"):
        if path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(module.startswith("finground.benchmark") for module in imported_modules), (
            path
        )


def test_default_model_is_qwen36_27b_fp8(monkeypatch) -> None:
    monkeypatch.delenv("FINGROUND_MODEL", raising=False)

    assert load_settings().model == "qwen36-27b-fp8"


def test_default_vllm_endpoint_is_remote(monkeypatch) -> None:
    monkeypatch.delenv("FINGROUND_VLLM_BASE_URL", raising=False)

    assert load_settings().vllm_base_url == "http://60.171.65.125:30845/v1"


def test_deepseek_model_uses_litellm_provider_prefix() -> None:
    model = create_adk_model("deepseek-v4-flash")

    assert isinstance(model, LiteLlm)
    assert model.model == "deepseek/deepseek-v4-flash"


def test_deepseek_json_output_uses_supported_json_object_mode() -> None:
    model = create_adk_model("deepseek-v4-flash", json_output=True)

    assert isinstance(model, LiteLlm)
    assert model._additional_args["response_format"] == {"type": "json_object"}


def test_non_deepseek_model_remains_native_adk_name() -> None:
    assert create_adk_model("gemini-3-flash-preview") == "gemini-3-flash-preview"


def test_qwen_model_uses_local_vllm_openai_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "finground.agents.common.SETTINGS",
        SimpleNamespace(
            vllm_base_url="http://vllm.test:8000/v1",
            vllm_api_key="test-key",
        ),
    )

    model = create_adk_model("Qwen/Qwen3.6-27B-FP8")

    assert isinstance(model, LiteLlm)
    assert model.model == "openai/Qwen/Qwen3.6-27B-FP8"
    assert model._additional_args == {
        "api_base": "http://vllm.test:8000/v1",
        "api_key": "test-key",
        "drop_params": True,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_qwen_vllm_json_output_uses_json_object_mode() -> None:
    model = create_adk_model("qwen-local", json_output=True)

    assert isinstance(model, LiteLlm)
    assert model.model == "openai/qwen-local"
    assert model._additional_args["response_format"] == {"type": "json_object"}
    assert "tool_choice" not in model._additional_args


def test_multi_kpi_agent_exposes_one_compact_dispatcher_and_one_common_agent() -> None:
    agent = create_multi_kpi_agent()

    assert agent.name == MULTI_KPI_COORDINATOR_NAME
    assert agent.description == (
        "Coordinates context-isolated specialists to complete a structured extraction workflow."
    )
    assert len(agent.tools) == 2
    assert isinstance(agent.tools[0], AgentTool)
    assert isinstance(agent.tools[1], KpiSpecialistTool)
    declaration = agent.tools[1]._get_declaration()
    assert declaration.name == KPI_DISPATCH_TOOL_NAME
    assert declaration.parameters_json_schema["properties"]["kpis"]["items"]["enum"] == list(
        KPI_KEYS
    )
    assert [tool.name for tool in agent.tools] == [
        COMMON_TASK_AGENT_NAME,
        KPI_DISPATCH_TOOL_NAME,
    ]
    root_tool_declarations = [
        tool._get_declaration().model_dump(mode="json", exclude_none=True) for tool in agent.tools
    ]
    serialized_root_tools = json.dumps(
        root_tool_declarations,
        separators=(",", ":"),
    )
    assert len(serialized_root_tools) <= 2_000
    assert "Source priority" not in serialized_root_tools
    assert "Reject:" not in serialized_root_tools
    assert tuple(kpi_agent_name(kpi) for kpi in KPI_KEYS) == KPI_AGENT_NAMES
    assert tuple(KPI_AGENT_SPECS) == KPI_KEYS
    assert "Do not inspect report text" in agent.instruction
    assert "Context isolation is intentional" in agent.instruction
    assert MULTI_KPI_MAX_OUTPUT_TOKENS == 4_096
    assert agent.generate_content_config.max_output_tokens == 4_096
    function_config = agent.generate_content_config.tool_config.function_calling_config
    assert function_config.mode == "ANY"

    common_agent = agent.tools[0].agent
    assert common_agent.include_contents == "none"
    assert "never decides an individual KPI value" in common_agent.description
    assert {tool.__name__ for tool in common_agent.tools} == {
        "prepare_multi_kpi_report",
        "query_multi_kpi_progress",
        "finalize_multi_kpi_report",
    }

    dispatcher = agent.tools[1]
    assert tuple(dispatcher._specialists) == KPI_KEYS
    for kpi, agent_tool in dispatcher._specialists.items():
        specialist = agent_tool.agent
        assert specialist.name == kpi_agent_name(kpi)
        assert specialist.include_contents == "none"
        assert f"only {kpi}" in specialist.description
        assert KPI_AGENT_SPECS[kpi].source_priority in specialist.description
        assert f"exactly one canonical KPI: {kpi}" in specialist.instruction
        assert "Do not find, judge, or record any other KPI" in specialist.instruction
        specialist_tool_names = {
            getattr(tool, "name", None) or tool.__name__ for tool in specialist.tools
        }
        assert specialist_tool_names == {
            f"find_{kpi}_candidates",
            "search_report",
            "read_report_pages",
            "record_multi_kpi_progress",
        }
        record_tool = next(
            tool
            for tool in specialist.tools
            if getattr(tool, "name", None) == "record_multi_kpi_progress"
        )
        evidence_schema = record_tool._get_declaration().parameters_json_schema["properties"][
            "kpis"
        ]["items"]
        assert evidence_schema["properties"]["kpi"]["enum"] == [kpi]
        assert evidence_schema["properties"]["status"]["enum"] == [
            "found",
            "explicit_zero",
            "absent",
            "ambiguous",
        ]
        assert "value" not in evidence_schema["properties"]


def test_multi_kpi_budget_matches_multi_agent_topology() -> None:
    assert MULTI_KPI_LLM_CALL_LIMIT == 200
    assert MULTI_KPI_PROGRESS_REMINDER_CALL == 120
    assert MULTI_KPI_SEARCH_LIMIT == 2
    assert MULTI_KPI_FINAL_WARNING_CALL == 160
    assert MULTI_KPI_SUBMISSION_DEADLINE == 200


def test_requested_kpi_scope_supports_single_multiple_and_all() -> None:
    assert resolve_requested_kpis("Find revenue") == ["revenue"]
    assert resolve_requested_kpis("Find net income and capital expenditures") == [
        "net_income",
        "capex",
    ]
    assert resolve_requested_kpis("Extract all 31 KPIs") == list(KPI_KEYS)


def test_multi_kpi_app_uses_deterministic_context_filter_without_llm_compaction() -> None:
    app = create_multi_kpi_app()

    assert app.name == "finground_multi_kpi"
    assert len(app.plugins) == 1
    assert app.plugins[0].name == "context_filter_plugin"
    assert app.events_compaction_config is None
    assert MULTI_KPI_CONTEXT_WINDOW_TOKENS == 131_072
