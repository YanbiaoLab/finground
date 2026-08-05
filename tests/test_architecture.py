import os
from pathlib import Path

import pytest

from finground.agent import app, root_agent
from finground.config import ModelConfig, load_project_env
from finground.kpi_dispatcher import (
    DISPATCH_ITEM_NAME,
    DISPATCHER_NAME,
    MAX_PARALLEL_WORKERS,
    KpiDispatchOutcome,
    create_kpi_dispatcher,
)
from finground.kpi_worker import TOOL_CALL_MAX_ATTEMPTS, WORKER_NAME, create_kpi_worker
from finground.root_agent import create_root_agent
from finground.task_store import TASK_TOOL_NAMES


def _tool_name(tool: object) -> str:
    return getattr(tool, "name", getattr(tool, "__name__", ""))


def test_app_has_one_root_and_one_task_worker() -> None:
    assert app.root_agent is root_agent
    assert root_agent.sub_agents == []
    dispatcher = create_kpi_dispatcher()
    parallel_workers = [node for node in dispatcher.graph.nodes if node.name == DISPATCH_ITEM_NAME]

    assert len(parallel_workers) == 1
    assert DISPATCH_ITEM_NAME != WORKER_NAME
    assert parallel_workers[0].max_parallel_workers == MAX_PARALLEL_WORKERS
    assert dispatcher.output_schema == list[KpiDispatchOutcome]
    worker_retry = create_kpi_worker().retry_config
    root_retry = create_root_agent().retry_config
    assert worker_retry is not None
    assert root_retry is not None
    assert worker_retry.max_attempts == TOOL_CALL_MAX_ATTEMPTS
    assert root_retry.max_attempts == TOOL_CALL_MAX_ATTEMPTS
    assert worker_retry.exceptions == ["JSONDecodeError"]
    assert root_retry.exceptions == ["JSONDecodeError"]


def test_agents_expose_only_their_scoped_tools() -> None:
    worker = create_kpi_worker()

    assert [_tool_name(tool) for tool in root_agent.tools] == [*TASK_TOOL_NAMES, DISPATCHER_NAME]
    assert [_tool_name(tool) for tool in worker.tools] == [
        "GetKpiKnowledge",
        "SearchReport",
        "ReadReportChunks",
        "finish_task",
    ]
    assert [plugin.name for plugin in app.plugins] == [
        "report_upload",
        "task_progress",
        "scoped_context_compaction",
    ]


def test_root_description_is_domain_and_tool_agnostic() -> None:
    description = create_root_agent().description

    assert description == "A general-purpose coordinator for complex, multi-step work."
    assert all(term not in description.lower() for term in ("tool", "taskcreate", "kpi", "report"))


def test_model_config_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGROUND_MODEL", "openai/configured-model")
    monkeypatch.setenv("FINGROUND_MODEL_BASE_URL", "https://model.example/v1/")
    monkeypatch.setenv("FINGROUND_MODEL_API_KEY", "secret")

    config = ModelConfig.from_env()

    assert config.name == "openai/configured-model"
    assert config.base_url == "https://model.example/v1"
    assert config.api_key == "secret"
    assert "secret" not in repr(config)


def test_model_config_rejects_missing_required_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINGROUND_MODEL_API_KEY")

    with pytest.raises(RuntimeError, match="FINGROUND_MODEL_API_KEY"):
        ModelConfig.from_env()


def test_model_config_rejects_example_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGROUND_MODEL_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("FINGROUND_MODEL_API_KEY", "secret")

    with pytest.raises(RuntimeError, match="example placeholder"):
        ModelConfig.from_env()


def test_project_dotenv_loads_without_overriding_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FROM_DOTENV=file-value\nPROCESS_WINS=file-value\n")
    monkeypatch.delenv("FROM_DOTENV", raising=False)
    monkeypatch.setenv("PROCESS_WINS", "process-value")

    assert load_project_env(env_file) is True
    assert os.environ["FROM_DOTENV"] == "file-value"
    assert os.environ["PROCESS_WINS"] == "process-value"
