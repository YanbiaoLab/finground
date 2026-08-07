import json
import os
from pathlib import Path
from typing import Any

import pytest

from finground.agent import app, root_agent
from finground.config import ModelConfig, load_project_env
from finground.kpi_catalog import KPI_CATALOG
from finground.kpi_dispatcher import (
    DISPATCH_ITEM_NAME,
    DISPATCHER_DESCRIPTION,
    DISPATCHER_NAME,
    MAX_PARALLEL_WORKERS,
    create_kpi_dispatcher,
)
from finground.kpi_worker import TOOL_CALL_MAX_ATTEMPTS, WORKER_NAME, create_kpi_worker
from finground.report_qa_dispatcher import (
    REPORT_QA_DISPATCHER_NAME,
    REPORT_QA_NODE_NAME,
    create_report_qa_dispatcher,
)
from finground.report_qa_worker import REPORT_QA_WORKER_NAME, create_report_qa_worker
from finground.root_agent import ROOT_INSTRUCTION, create_root_agent
from finground.task_store import TASK_TOOL_NAMES


def _tool_name(tool: object) -> str:
    return getattr(tool, "name", getattr(tool, "__name__", ""))


def test_app_has_one_root_and_two_task_workers() -> None:
    assert app.root_agent is root_agent
    assert root_agent.sub_agents == []
    dispatcher = create_kpi_dispatcher()
    parallel_workers = [node for node in dispatcher.graph.nodes if node.name == DISPATCH_ITEM_NAME]

    assert len(parallel_workers) == 1
    assert DISPATCH_ITEM_NAME != WORKER_NAME
    assert parallel_workers[0].max_parallel_workers == MAX_PARALLEL_WORKERS
    assert dispatcher.output_schema == list[dict[str, Any]]
    worker_retry = create_kpi_worker().retry_config
    root_retry = create_root_agent().retry_config
    assert worker_retry is not None
    assert root_retry is not None
    assert worker_retry.max_attempts == TOOL_CALL_MAX_ATTEMPTS
    assert root_retry.max_attempts == TOOL_CALL_MAX_ATTEMPTS
    assert worker_retry.exceptions == ["JSONDecodeError"]
    assert root_retry.exceptions == ["JSONDecodeError"]
    report_qa_dispatcher = create_report_qa_dispatcher()
    assert [node.name for node in report_qa_dispatcher.graph.nodes] == [
        "__START__",
        REPORT_QA_NODE_NAME,
    ]
    assert report_qa_dispatcher.output_schema is not None
    assert create_report_qa_worker().mode == "task"


def test_root_workflow_tool_schemas_stay_shallow() -> None:
    declarations = {
        tool.name: tool._get_declaration().model_dump(mode="json", exclude_none=True)
        for tool in root_agent.tools
        if tool.name in {DISPATCHER_NAME, REPORT_QA_DISPATCHER_NAME}
    }

    assert declarations[DISPATCHER_NAME]["response_json_schema"] == {
        "items": {"additionalProperties": True, "type": "object"},
        "type": "array",
    }
    assert declarations[REPORT_QA_DISPATCHER_NAME]["response_json_schema"] == {
        "additionalProperties": True,
        "type": "object",
    }
    assert sum(len(json.dumps(value)) for value in declarations.values()) < 2_500


def test_agents_expose_only_their_scoped_tools() -> None:
    worker = create_kpi_worker()
    report_qa_worker = create_report_qa_worker()

    assert [_tool_name(tool) for tool in root_agent.tools] == [
        *TASK_TOOL_NAMES,
        DISPATCHER_NAME,
        REPORT_QA_DISPATCHER_NAME,
    ]
    assert [_tool_name(tool) for tool in worker.tools] == [
        "GetKpiKnowledge",
        "SearchReport",
        "ReadReportChunks",
        "finish_task",
    ]
    assert report_qa_worker.name == REPORT_QA_WORKER_NAME
    assert [_tool_name(tool) for tool in report_qa_worker.tools] == [
        "PrepareReportQuestion",
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


def test_root_retries_only_retryable_failed_kpis_once() -> None:
    assert "retryable=true" in ROOT_INSTRUCTION
    assert "at most one more time" in ROOT_INSTRUCTION
    assert "Never rerun a succeeded task" in ROOT_INSTRUCTION
    assert "make a third dispatcher call" in ROOT_INSTRUCTION
    assert "Never expose exception class names" in ROOT_INSTRUCTION


def test_root_can_see_supported_kpis_and_routes_other_metrics_to_report_qa() -> None:
    dispatcher_tool = next(tool for tool in root_agent.tools if tool.name == DISPATCHER_NAME)
    declaration = dispatcher_tool._get_declaration().model_dump(mode="json", exclude_none=True)

    assert declaration["description"] == DISPATCHER_DESCRIPTION
    assert all(kpi_key in declaration["description"] for kpi_key in KPI_CATALOG)
    assert "metrics outside this list" in declaration["description"]
    assert REPORT_QA_DISPATCHER_NAME in ROOT_INSTRUCTION
    assert "outside that list" in ROOT_INSTRUCTION
    assert "non-canonical annual-report" in ROOT_INSTRUCTION
    assert "preserve" in ROOT_INSTRUCTION
    assert "user's exact wording" in ROOT_INSTRUCTION


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
