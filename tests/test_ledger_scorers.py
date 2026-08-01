import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from finground.benchmark.cli import build_parser
from finground.benchmark.multi_kpi_scorer import score_kpi, score_multi_kpi
from finground.documents import Report
from finground.kpis import KPI_KEYS
from finground.tools import (
    MULTI_KPI_AUDIT_STATE_KEY,
    MULTI_KPI_RESULT_STATE_KEY,
    build_report_state,
    submit_multi_kpi_extraction,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ledger"


def _write_multi_ground_truth(path: Path) -> None:
    values: dict[str, list[object]] = {
        "ticker": ["ACME", "OTHER"],
        "exchange": ["NYSE", "NYSE"],
        "company_name": ["ACME Corp", "Other Corp"],
        "industry": ["Fixture", "Fixture"],
        "year": [2023, 2023],
    }
    values.update({kpi: [None, None] for kpi in KPI_KEYS})
    values["revenue"] = [1_234_000_000.0, 999.0]
    pq.write_table(pa.table(values), path)


def _write_long_ground_truth(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "ticker": ["ACME"],
                "exchange": ["NYSE"],
                "company_name": ["ACME Corp"],
                "industry": ["Fixture"],
                "year": [2023],
                "kpi": ["revenue"],
                "value": [1_234_000_000.0],
                "mmd_text": ["unused by scorer"],
            }
        ),
        path,
    )


def _write_multi_scope(output_dir: Path) -> None:
    (output_dir / "run_meta.json").write_text(
        json.dumps({"report_ids": ["NYSE_ACME_2023"]}),
        encoding="utf-8",
    )


def test_cli_exposes_only_supported_ledger_commands() -> None:
    subparsers = next(action for action in build_parser()._actions if action.dest == "command")
    assert set(subparsers.choices) == {
        "ledger-kpi",
        "ledger-multi",
        "ledger-select-kpi-samples",
        "ledger-score-kpi",
        "ledger-score-multi",
    }


def test_every_cli_option_has_help_text() -> None:
    subparsers = next(action for action in build_parser()._actions if action.dest == "command")
    for command, command_parser in subparsers.choices.items():
        missing_help = [
            action.dest
            for action in command_parser._actions
            if action.option_strings and action.dest != "help" and not action.help
        ]
        assert not missing_help, f"{command} options missing help: {missing_help}"


def test_prediction_commands_only_expose_required_parquet_input() -> None:
    subparsers = next(action for action in build_parser()._actions if action.dest == "command")
    for command in ("ledger-kpi", "ledger-multi"):
        command_parser = subparsers.choices[command]
        option_actions = {
            option: action for action in command_parser._actions for option in action.option_strings
        }
        assert option_actions["--parquet"].required is True
        assert "--ocr-root" not in option_actions
        assert "--queries" not in option_actions
        assert "--qrels" not in option_actions
        assert "--candidate-pages" not in option_actions
        assert "--retrieval-mode" not in option_actions
        assert "--concurrency" in option_actions


def test_multi_kpi_defaults_to_twenty_concurrent_runs() -> None:
    args = build_parser().parse_args(["ledger-multi", "--parquet", "multi-kpi.parquet"])

    assert args.concurrency == 20


def test_single_kpi_command_requires_a_canonical_kpi() -> None:
    args = build_parser().parse_args(
        ["ledger-kpi", "--kpi", "revenue", "--parquet", "multi-kpi.parquet"]
    )

    assert args.kpi == "revenue"
    assert args.output_dir == Path(__file__).resolve().parents[1] / "outputs" / "ledger" / "kpi"


def test_score_commands_only_require_output_and_original_parquet() -> None:
    subparsers = next(action for action in build_parser()._actions if action.dest == "command")
    for command in ("ledger-score-kpi", "ledger-score-multi"):
        option_actions = {
            option: action
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert option_actions["--output-dir"].required is True
        assert option_actions["--parquet"].required is True
        assert "--ledger-root" not in option_actions
        assert "--kpis-long" not in option_actions
        assert "--test-set-reports" not in option_actions


def test_internal_multi_scorer_reads_ground_truth_from_parquet(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    (predictions / "raw").mkdir(parents=True)
    source = FIXTURE / "multi" / "raw" / "NYSE_ACME_2023.json"
    (predictions / "raw" / source.name).write_text(source.read_text())
    stale = json.loads(source.read_text())
    stale.update(
        ticker="OTHER",
        report_name="NYSE_OTHER_2023",
        extraction=stale["extraction"] | {"ticker": "OTHER"},
    )
    (predictions / "raw" / "NYSE_OTHER_2023.json").write_text(json.dumps(stale))
    _write_multi_scope(predictions)
    parquet_path = tmp_path / "multi.parquet"
    _write_multi_ground_truth(parquet_path)

    result = score_multi_kpi(
        output_dir=predictions,
        parquet_path=parquet_path,
    )

    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["n_gt"] == 1
    assert result["quality_gate"]["passed"] is False
    assert sum(gate["passed"] for gate in result["per_kpi_quality_gates"]) == 1
    assert result["reports_scored"] == 1
    summary = (predictions / "summary.md").read_text()
    assert "1.0000" in summary


def test_single_kpi_scorer_projects_only_requested_ledger_column(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    (predictions / "raw").mkdir(parents=True)
    source = FIXTURE / "multi" / "raw" / "NYSE_ACME_2023.json"
    (predictions / "raw" / source.name).write_text(source.read_text())
    (predictions / "run_meta.json").write_text(
        json.dumps(
            {
                "report_ids": ["NYSE_ACME_2023"],
                "requested_kpis": ["revenue"],
            }
        ),
        encoding="utf-8",
    )
    parquet_path = tmp_path / "multi.parquet"
    _write_multi_ground_truth(parquet_path)

    result = score_kpi(
        kpi="revenue",
        output_dir=predictions,
        parquet_path=parquet_path,
    )

    assert result["n_gt"] == 1
    assert result["matched"] == 1
    assert result["missing"] == 0
    assert result["quality_gate"]["passed"] is True


def test_single_kpi_scorer_accepts_official_ledger_long_parquet(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    (predictions / "raw").mkdir(parents=True)
    source = FIXTURE / "multi" / "raw" / "NYSE_ACME_2023.json"
    (predictions / "raw" / source.name).write_text(source.read_text())
    (predictions / "run_meta.json").write_text(
        json.dumps(
            {
                "report_ids": ["NYSE_ACME_2023"],
                "requested_kpis": ["revenue"],
            }
        ),
        encoding="utf-8",
    )
    parquet_path = tmp_path / "ledger-long.parquet"
    _write_long_ground_truth(parquet_path)

    result = score_kpi(
        kpi="revenue",
        output_dir=predictions,
        parquet_path=parquet_path,
    )

    assert result["quality_gate"]["passed"] is True


def test_internal_multi_scorer_scores_incomplete_partial_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    raw_dir = predictions / "raw"
    raw_dir.mkdir(parents=True)
    source = json.loads((FIXTURE / "multi" / "raw" / "NYSE_ACME_2023.json").read_text())
    source["status"] = "incomplete"
    (raw_dir / "NYSE_ACME_2023.json").write_text(json.dumps(source))
    _write_multi_scope(predictions)
    parquet_path = tmp_path / "multi.parquet"
    _write_multi_ground_truth(parquet_path)

    result = score_multi_kpi(output_dir=predictions, parquet_path=parquet_path)

    assert result["recall"] == 1.0
    assert result["quality_gate"]["passed"] is False
    assert "incomplete=1" in (predictions / "summary.md").read_text()


def test_internal_multi_scorer_accepts_submission_tool_output(tmp_path: Path) -> None:
    report_path = FIXTURE / "report.mmd"
    report = Report("NYSE_ACME_2023", "NYSE", "ACME", 2023, report_path.read_text())
    predictions = tmp_path / "predictions"
    context = SimpleNamespace(state={"report": build_report_state(report)})
    result = submit_multi_kpi_extraction(
        "ACME",
        "USD",
        "Values reported in millions.",
        [
            {
                "kpi": "revenue",
                "fiscal_year": 2023,
                "status": "found",
                "value_verbatim": "1,234",
                "unit_scale": "millions",
                "unit_text": "(in millions, except per-share amounts)",
                "unit_page": 3,
                "page": 3,
                "statement": "Consolidated Statements of Operations",
                "line_label": "Revenue",
                "year_label": "2023",
                "scope": "consolidated total company",
            },
            *[
                {"kpi": kpi, "fiscal_year": 2023, "status": "absent"}
                for kpi in KPI_KEYS
                if kpi != "revenue"
            ],
        ],
        context,
    )

    assert result["status"] == "success"
    raw_dir = predictions / "raw"
    raw_dir.mkdir(parents=True)
    record = {
        "ticker": "ACME",
        "year": 2023,
        "exchange": "NYSE",
        "report_name": "NYSE_ACME_2023",
        "model": "fixture",
        "status": "ok",
        "extraction": context.state[MULTI_KPI_RESULT_STATE_KEY],
        "audit": context.state[MULTI_KPI_AUDIT_STATE_KEY],
        "error": None,
    }
    (raw_dir / "NYSE_ACME_2023.json").write_text(json.dumps(record))
    _write_multi_scope(predictions)
    parquet_path = tmp_path / "multi.parquet"
    _write_multi_ground_truth(parquet_path)
    score_multi_kpi(
        output_dir=predictions,
        parquet_path=parquet_path,
    )
    assert "1.0000" in (predictions / "summary.md").read_text()
