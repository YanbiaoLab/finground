"""Command-line interface for FinGround's LEDGER benchmark tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finground.benchmark.answer_extractor import extract_output_answers_sync
from finground.benchmark.kpi_samples import (
    write_grounded_multi_parquet,
    write_grounded_report_ids,
)
from finground.benchmark.multi_kpi_runner import run_kpi_sync, run_multi_kpi_sync
from finground.benchmark.multi_kpi_scorer import score_kpi, score_multi_kpi
from finground.kpis import KPI_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ledger"
DEFAULT_KPI_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "kpi"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finground")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_options(
        command_parser: argparse.ArgumentParser,
        *,
        default_output: Path,
    ) -> None:
        command_parser.add_argument(
            "--parquet",
            type=_path,
            required=True,
            help="Multi-KPI Parquet file or directory; reads report metadata and mmd_text",
        )
        command_parser.add_argument(
            "--output-dir",
            type=_path,
            default=default_output,
            help=f"directory for raw report JSON and run_meta.json (default: {default_output})",
        )
        command_parser.add_argument(
            "--limit-reports",
            type=_non_negative_int,
            default=None,
            help="maximum number of reports to process; omit to process all reports",
        )
        command_parser.add_argument(
            "--reports-file",
            type=_path,
            default=None,
            help="optional text file listing exact report IDs to include, one per line",
        )
        command_parser.add_argument(
            "--resume",
            action="store_true",
            help="skip reports whose existing raw JSON result has status=ok",
        )
        command_parser.add_argument(
            "--concurrency",
            type=_positive_int,
            default=20,
            help="maximum number of concurrent Multi-KPI agent runs (default: 20)",
        )

    single_parser = commands.add_parser("ledger-kpi", help="run one ADK KPI specialist")
    single_parser.add_argument(
        "--kpi", required=True, choices=KPI_KEYS, help="KPI specialist to test"
    )
    add_run_options(single_parser, default_output=DEFAULT_KPI_OUTPUT_ROOT)

    multi_parser = commands.add_parser("ledger-multi", help="run all 31 KPI specialists")
    add_run_options(multi_parser, default_output=DEFAULT_OUTPUT_ROOT / "multi")

    sample_parser = commands.add_parser(
        "ledger-select-kpi-samples",
        help="select reports where a KPI ground-truth value is visible in report text",
    )
    sample_parser.add_argument("--kpi", required=True, choices=KPI_KEYS, help="KPI to sample")
    sample_parser.add_argument(
        "--parquet", type=_path, required=True, help="official wide LEDGER Parquet"
    )
    sample_parser.add_argument(
        "--limit", type=_positive_int, default=5, help="maximum reports to select (default: 5)"
    )
    sample_parser.add_argument(
        "--max-per-ticker",
        type=_positive_int,
        default=1,
        help="maximum selected years for one ticker (default: 1)",
    )
    sample_parser.add_argument(
        "--output-file", type=_path, required=True, help="reports-file to write for ledger-kpi"
    )

    multi_sample_parser = commands.add_parser(
        "ledger-select-multi-samples",
        help="write a long Parquet slice containing only report-grounded KPI answers",
    )
    multi_sample_parser.add_argument(
        "--parquet", type=_path, required=True, help="official wide LEDGER Parquet"
    )
    multi_sample_parser.add_argument(
        "--min-per-kpi",
        type=_positive_int,
        default=5,
        help="minimum grounded answers selected for every KPI (default: 5)",
    )
    multi_sample_parser.add_argument(
        "--max-reports",
        type=_positive_int,
        default=50,
        help="maximum reports in the grounded multi slice (default: 50)",
    )
    multi_sample_parser.add_argument(
        "--max-per-ticker",
        type=_positive_int,
        default=1,
        help="maximum selected years for one ticker (default: 1)",
    )
    multi_sample_parser.add_argument(
        "--output-file", type=_path, required=True, help="long Parquet file to write"
    )

    score_multi_parser = commands.add_parser(
        "ledger-score-multi",
        help="score Multi-KPI output against its original wide Parquet",
    )
    score_single_parser = commands.add_parser(
        "ledger-score-kpi",
        help="score one KPI specialist against LEDGER Parquet ground truth",
    )
    score_single_parser.add_argument(
        "--kpi", required=True, choices=KPI_KEYS, help="KPI specialist that produced the run"
    )
    score_single_parser.add_argument(
        "--baseline-dir",
        type=_path,
        default=None,
        help="optional previous scored run for per-KPI improvement comparison",
    )
    score_single_parser.add_argument(
        "--output-dir", type=_path, required=True, help="ledger-kpi output directory"
    )
    score_single_parser.add_argument(
        "--parquet", type=_path, required=True, help="original LEDGER evaluation Parquet"
    )
    score_single_parser.add_argument(
        "--tolerance",
        type=_non_negative_float,
        default=0.01,
        help="relative tolerance for a matched prediction (default: 0.01)",
    )
    score_single_parser.add_argument(
        "--zero-eps",
        type=_non_negative_float,
        default=0.5,
        help="absolute tolerance when ground truth is approximately zero (default: 0.5)",
    )
    score_multi_parser.add_argument(
        "--output-dir",
        type=_path,
        required=True,
        help="ledger-multi output directory containing raw report JSON files",
    )
    score_multi_parser.add_argument(
        "--baseline-dir",
        type=_path,
        default=None,
        help="optional previous scored run for per-KPI improvement comparison",
    )
    score_multi_parser.add_argument(
        "--parquet",
        type=_path,
        required=True,
        help="original Multi-KPI evaluation Parquet file or shard directory",
    )
    score_multi_parser.add_argument(
        "--tolerance",
        type=_non_negative_float,
        default=0.01,
        help="relative tolerance for a matched prediction (default: 0.01)",
    )
    score_multi_parser.add_argument(
        "--zero-eps",
        type=_non_negative_float,
        default=0.5,
        help="absolute tolerance when ground truth is approximately zero (default: 0.5)",
    )
    score_multi_parser.add_argument(
        "--llm-extract-answers",
        action="store_true",
        help=(
            "first use an isolated schema-constrained LLM judge to extract LEDGER rows "
            "from each saved user-visible answer"
        ),
    )
    score_multi_parser.add_argument(
        "--judge-model",
        default=None,
        help="optional model name for --llm-extract-answers (default: FINGROUND_MODEL)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"ledger-kpi", "ledger-multi"}:
        run = run_kpi_sync if args.command == "ledger-kpi" else run_multi_kpi_sync
        kwargs = {"kpi": args.kpi} if args.command == "ledger-kpi" else {}
        output_dir = (
            args.output_dir / args.kpi
            if args.command == "ledger-kpi" and args.output_dir == DEFAULT_KPI_OUTPUT_ROOT
            else args.output_dir
        )
        result = run(
            parquet_path=args.parquet,
            output_dir=output_dir,
            limit_reports=args.limit_reports,
            reports_file=args.reports_file,
            resume=args.resume,
            concurrency=args.concurrency,
            **kwargs,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-select-kpi-samples":
        result = write_grounded_report_ids(
            args.parquet,
            kpi=args.kpi,
            limit=args.limit,
            max_per_ticker=args.max_per_ticker,
            output_file=args.output_file,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-select-multi-samples":
        result = write_grounded_multi_parquet(
            args.parquet,
            min_per_kpi=args.min_per_kpi,
            max_reports=args.max_reports,
            max_per_ticker=args.max_per_ticker,
            output_file=args.output_file,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-score-multi":
        prediction_dir_name = "raw"
        if args.llm_extract_answers:
            extract_output_answers_sync(
                output_dir=args.output_dir,
                model_name=args.judge_model,
            )
            prediction_dir_name = "judged"
        result = score_multi_kpi(
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            tolerance=args.tolerance,
            zero_eps=args.zero_eps,
            prediction_dir_name=prediction_dir_name,
            baseline_dir=args.baseline_dir,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-score-kpi":
        result = score_kpi(
            kpi=args.kpi,
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            tolerance=args.tolerance,
            zero_eps=args.zero_eps,
            baseline_dir=args.baseline_dir,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
