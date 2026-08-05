from finground.agent import app, root_agent
from finground.kpi_dispatcher import (
    DISPATCH_ITEM_NAME,
    DISPATCHER_NAME,
    MAX_PARALLEL_WORKERS,
    KpiDispatchOutcome,
    create_kpi_dispatcher,
)
from finground.kpi_worker import WORKER_MAX_ATTEMPTS, WORKER_NAME, create_kpi_worker
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
    assert worker_retry is not None
    assert worker_retry.max_attempts == WORKER_MAX_ATTEMPTS
    assert worker_retry.exceptions == ["JSONDecodeError"]


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
