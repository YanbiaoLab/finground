"""Command-line interface for FinGround's LEDGER benchmark tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finground.benchmark.multi_kpi_runner import run_multi_kpi_sync
from finground.benchmark.multi_kpi_scorer import score_multi_kpi
from finground.benchmark.needle_runner import run_needle_sync
from finground.benchmark.needle_scorer import score_needle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ledger"


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

    needle_parser = commands.add_parser("ledger-needle", help="run ADK single-KPI predictions")
    needle_parser.add_argument(
        "--parquet",
        type=_path,
        required=True,
        help="KPI-QA Parquet file or directory of Parquet shards",
    )
    needle_parser.add_argument(
        "--output-dir",
        type=_path,
        default=DEFAULT_OUTPUT_ROOT / "needle",
        help="directory for responses.jsonl and run_meta.json (default: outputs/ledger/needle)",
    )
    needle_parser.add_argument(
        "--limit-queries",
        type=_non_negative_int,
        default=None,
        help="maximum number of queries to process; omit to process all queries",
    )
    needle_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=4,
        help="maximum number of concurrent Needle agent runs (default: 4)",
    )
    multi_parser = commands.add_parser("ledger-multi", help="run ADK Multi-KPI predictions")
    multi_parser.add_argument(
        "--parquet",
        type=_path,
        required=True,
        help="Multi-KPI Parquet file or directory; reads report metadata and mmd_text",
    )
    multi_parser.add_argument(
        "--output-dir",
        type=_path,
        default=DEFAULT_OUTPUT_ROOT / "multi",
        help="directory for raw report JSON and run_meta.json (default: outputs/ledger/multi)",
    )
    multi_parser.add_argument(
        "--limit-reports",
        type=_non_negative_int,
        default=None,
        help="maximum number of reports to process; omit to process all reports",
    )
    multi_parser.add_argument(
        "--reports-file",
        type=_path,
        default=None,
        help="optional text file listing exact report IDs to include, one per line",
    )
    multi_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip reports whose existing raw JSON result has status=ok",
    )
    multi_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=4,
        help="maximum number of concurrent Multi-KPI agent runs (default: 4)",
    )

    score_needle_parser = commands.add_parser(
        "ledger-score-needle",
        help="score Needle output against its original KPI-QA Parquet",
    )
    score_needle_parser.add_argument(
        "--output-dir",
        type=_path,
        required=True,
        help="ledger-needle output directory containing responses.jsonl",
    )
    score_needle_parser.add_argument(
        "--parquet",
        type=_path,
        required=True,
        help="original KPI-QA evaluation Parquet file or shard directory",
    )
    score_needle_parser.add_argument(
        "--tolerance",
        type=_non_negative_float,
        default=0.01,
        help="relative tolerance for a matched prediction (default: 0.01)",
    )
    score_needle_parser.add_argument(
        "--strict-tolerance",
        type=_non_negative_float,
        default=0.0005,
        help="relative tolerance for strict accuracy (default: 0.0005)",
    )
    score_needle_parser.add_argument(
        "--zero-eps",
        type=_non_negative_float,
        default=0.5,
        help="absolute tolerance when ground truth is approximately zero (default: 0.5)",
    )

    score_multi_parser = commands.add_parser(
        "ledger-score-multi",
        help="score Multi-KPI output against its original wide Parquet",
    )
    score_multi_parser.add_argument(
        "--output-dir",
        type=_path,
        required=True,
        help="ledger-multi output directory containing raw report JSON files",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ledger-needle":
        result = run_needle_sync(
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            limit_queries=args.limit_queries,
            concurrency=args.concurrency,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-multi":
        result = run_multi_kpi_sync(
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            limit_reports=args.limit_reports,
            reports_file=args.reports_file,
            resume=args.resume,
            concurrency=args.concurrency,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-score-needle":
        result = score_needle(
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            tolerance=args.tolerance,
            strict_tolerance=args.strict_tolerance,
            zero_eps=args.zero_eps,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "ledger-score-multi":
        result = score_multi_kpi(
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            tolerance=args.tolerance,
            zero_eps=args.zero_eps,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
