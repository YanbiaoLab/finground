# FinGround

FinGround runs Google ADK agents against the two LEDGER financial-report
extraction tasks and scores their outputs locally. It supports exactly four
commands:

| Command | Purpose |
| --- | --- |
| `ledger-needle` | Extract one requested KPI from each report/query pair |
| `ledger-score-needle` | Score Needle predictions |
| `ledger-multi` | Extract all supported KPIs from each report |
| `ledger-score-multi` | Score Multi-KPI predictions |

Prediction and scoring inputs must be Parquet files or directories containing
Parquet shards. CSV and standalone OCR directories are not supported.

## Install

FinGround requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

## Configure the model

The default configuration expects a Qwen model behind a local vLLM
OpenAI-compatible endpoint. Set the variables before invoking FinGround:

```bash
export FINGROUND_MODEL=qwen36-27b-fp8
export FINGROUND_VLLM_BASE_URL=http://localhost:8000/v1
export FINGROUND_VLLM_API_KEY=EMPTY
```

Any model name beginning with `qwen` (case-insensitive) uses the configured
vLLM endpoint. A model name beginning with `deepseek-` uses LiteLLM's DeepSeek
provider. Other model names are passed directly to Google ADK.

## Download the datasets

The official datasets are hosted on Hugging Face:

- [LEDGER Long-Context KPI-QA](https://huggingface.co/datasets/artefactory/ledger-long-context-KPI-QA)
- [LEDGER Long-Context Multi-KPI](https://huggingface.co/datasets/artefactory/ledger-long-context-multi-kpi)

Use the Hugging Face CLI through `uvx`; no additional project dependency is
needed. The following commands preserve the dataset directory structure under
the exact paths used in the examples below.

Download the ten Needle evaluation Parquet shards. The include pattern excludes
raw MMD assets and the separate large `eval/data.parquet` file:

```bash
uvx --from huggingface-hub hf download \
  artefactory/ledger-long-context-KPI-QA \
  --repo-type dataset \
  --include 'eval/data-*.parquet' \
  --local-dir data/ledger/raw/kpi_qa
```

Download both Multi-KPI configurations:

```bash
uvx --from huggingface-hub hf download \
  artefactory/ledger-long-context-multi-kpi \
  eval/data.parquet no_eval/data.parquet \
  --repo-type dataset \
  --local-dir data/ledger/raw/multi_kpi
```

The resulting files should have this layout:

```text
data/ledger/raw/
├── kpi_qa/
│   └── eval/
│       ├── data-00000-of-00010.parquet
│       ├── ...
│       └── data-00009-of-00010.parquet
└── multi_kpi/
    ├── eval/data.parquet
    └── no_eval/data.parquet
```

`eval` is the benchmark-evaluation split. `no_eval` is intended for development
and training; it is useful for smoke tests without repeatedly selecting from the
official Multi-KPI evaluation reports. Dataset files are ignored by Git.

## Run Multi-KPI extraction

Start with a small development run:

```bash
uv run finground ledger-multi \
  --parquet data/ledger/raw/multi_kpi/no_eval/data.parquet \
  --output-dir outputs/ledger/multi-no-eval \
  --limit-reports 10 \
  --concurrency 4
```

For an evaluation run, point both prediction and scoring at the `eval` file:

```bash
uv run finground ledger-multi \
  --parquet data/ledger/raw/multi_kpi/eval/data.parquet \
  --output-dir outputs/ledger/multi-eval \
  --limit-reports 10 \
  --concurrency 4

uv run finground ledger-score-multi \
  --output-dir outputs/ledger/multi-eval \
  --parquet data/ledger/raw/multi_kpi/eval/data.parquet
```

Omit `--limit-reports` to process every report. Use
`--reports-file reports.txt` to select exact report IDs, one per line. Add
`--resume` to skip existing `raw/*.json` records whose status is `ok`.

Each report produces `raw/<report_id>.json`. `run_meta.json` records the exact
run scope, model, call counts, concurrency, and failures. Scoring adds
`summary.md`, `predictions_long.csv`, and per-dimension metric CSV files to the
same output directory.

## Run Needle extraction

Run a small query sample, then score the generated responses against the same
Parquet shard directory:

```bash
uv run finground ledger-needle \
  --parquet data/ledger/raw/kpi_qa/eval \
  --output-dir outputs/ledger/needle-eval \
  --limit-queries 75 \
  --concurrency 4

uv run finground ledger-score-needle \
  --output-dir outputs/ledger/needle-eval \
  --parquet data/ledger/raw/kpi_qa/eval
```

Omit `--limit-queries` to process every query. Prediction writes
`responses.jsonl` and `run_meta.json`; scoring adds `summary.md`, `scored.csv`,
and per-dimension metric CSV files.

## Inspect command options

Every CLI parameter has built-in help:

```bash
uv run finground --help
uv run finground ledger-multi --help
uv run finground ledger-score-multi --help
uv run finground ledger-needle --help
uv run finground ledger-score-needle --help
```

`--concurrency N` limits the number of active ADK sessions. Lower it when the
model server cannot sustain the request volume. Scoring defaults to a 1%
relative match tolerance; use each scoring command's help to inspect or override
its tolerance options.

## Code layout

```text
finground/             agent construction, models, tools, report access, and retrieval
finground/tools/       state-backed report, work-record, and submission tools
finground/benchmark/   Parquet readers, ADK runners, CLI, and local scorers
tests/                 unit and end-to-end scorer tests
outputs/ledger/        generated run artifacts (Git-ignored)
```

Agent construction and tools remain under `finground/`. The benchmark package
owns dataset adaptation, ADK runner/session lifecycle, output files, and scoring.

## Verify the repository

```bash
uv run pytest -q
uv run ruff check finground tests
```
