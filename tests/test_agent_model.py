import ast
from pathlib import Path
from types import SimpleNamespace

from google.adk.models.lite_llm import LiteLlm

import finground.agent as agent_module
from finground.agent import (
    MULTI_KPI_LLM_CALL_LIMIT,
    create_adk_model,
    create_multi_kpi_agent,
    create_multi_kpi_app,
    create_needle_agent,
)
from finground.config import load_settings


def test_agent_core_does_not_import_benchmark_package() -> None:
    package_dir = Path(agent_module.__file__).parent
    for path in package_dir.glob("*.py"):
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


def test_default_vllm_endpoint_is_local(monkeypatch) -> None:
    monkeypatch.delenv("FINGROUND_VLLM_BASE_URL", raising=False)

    assert load_settings().vllm_base_url == "http://localhost:8000/v1"


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
        "finground.agent.SETTINGS",
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
    }


def test_qwen_vllm_json_output_uses_json_object_mode() -> None:
    model = create_adk_model("qwen-local", json_output=True)

    assert isinstance(model, LiteLlm)
    assert model.model == "openai/qwen-local"
    assert model._additional_args["response_format"] == {"type": "json_object"}


def test_needle_agent_exposes_state_and_submission_tools() -> None:
    agent = create_needle_agent()

    assert {tool.__name__ for tool in agent.tools} == {
        "get_report_info",
        "read_report_pages",
        "search_report",
        "submit_needle_extraction",
    }
    assert "correct every" in agent.instruction


def test_multi_kpi_agent_exposes_state_and_submission_tools() -> None:
    agent = create_multi_kpi_agent()

    tool_names = {getattr(tool, "name", None) or tool.__name__ for tool in agent.tools}
    assert tool_names == {
        "get_report_info",
        "query_multi_kpi_progress",
        "read_report_pages",
        "record_multi_kpi_progress",
        "search_report",
        "submit_multi_kpi_extraction",
    }
    assert "correct every" in agent.instruction
    assert MULTI_KPI_LLM_CALL_LIMIT == 30
    assert "At model call 18" in agent.instruction
    assert "At call 24" in agent.instruction
    assert "Call 25 is restricted" in agent.instruction
    assert "Missing means omitting" in agent.instruction
    function_config = agent.generate_content_config.tool_config.function_calling_config
    assert function_config.mode == "ANY"
    record_tool = next(
        tool for tool in agent.tools if getattr(tool, "name", None) == "record_multi_kpi_progress"
    )
    evidence_schema = record_tool._get_declaration().parameters_json_schema["properties"][
        "kpis"
    ]["items"]
    assert evidence_schema["properties"]["status"]["enum"] == [
        "found",
        "explicit_zero",
        "absent",
        "ambiguous",
    ]
    assert "value" not in evidence_schema["properties"]


def test_multi_kpi_app_enables_adk_context_filter_and_compaction() -> None:
    app = create_multi_kpi_app()

    assert app.name == "finground_multi_kpi"
    assert len(app.plugins) == 1
    assert app.plugins[0].name == "context_filter_plugin"
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.token_threshold == 18_000
    assert app.events_compaction_config.event_retention_size == 6
