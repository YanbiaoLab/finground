# finground

Financial document retrieval, extraction and QA pipeline built around an agent-style workflow.

## Highlights
- Parses financial documents and feeds structured extraction workflows.
- Supports report/KPI processing and orchestration logic in `finground` package.
- Provides a web frontend under `web/` for interaction and result viewing.
- Includes tests and evaluation datasets in `tests/`.

## Requirements
- Python `>=3.12` (project requirement in `pyproject.toml`)
- uv (optional, for dependency management)

## Installation
```bash
# from project root
uv sync
```

Or using pip:

```bash
python -m pip install -e .
```

## Usage
Run the web entrypoint:

```bash
finground-web
```

Or from source:

```bash
python -m finground.web
```

## Configuration
Copy environment sample if present and configure values for your environment:

```bash
cp .env.example .env
```

## Development
Run the test suite:

```bash
pytest
```

## Project layout
- `finground/` core application code
- `web/` frontend assets and app entry
- `tests/` unit tests and evaluation assets
- `pyproject.toml` project metadata and dependencies
- `uv.lock` pinned dependency lockfile
