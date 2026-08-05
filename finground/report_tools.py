"""Artifact-backed full-report search with hard context budgets."""

from __future__ import annotations

import json
import re
import secrets
from typing import Any

from google.adk.tools import ToolContext

from finground.kpi_catalog import KNOWLEDGE_STATE_KEY
from finground.named_function_tool import NamedFunctionTool

REPORT_STATE_KEY = "report"
REPORT_BUDGET_STATE_KEY = "temp:report_budget"
SEARCH_RESULTS_STATE_KEY = "temp:report_search_results"
CURSORS_STATE_KEY = "temp:report_cursors"
REPORT_QUESTION_STATE_KEY = "temp:report_question"

MAX_CHUNK_CHARS = 6_000
MAX_SEARCH_CALLS = 60
MAX_READ_CALLS = 40
MAX_SEARCH_RESULTS = 8
MAX_SNIPPET_CHARS = 600
MAX_READ_CHUNKS = 3
MAX_READ_CHARS = 18_000
MAX_TOTAL_OUTPUT_CHARS = 480_000
PAGE_BREAK_PATTERN = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)


def _error(message: str, *, budget_exhausted: bool = False) -> dict[str, Any]:
    return {"status": "error", "error": message, "budget_exhausted": budget_exhausted}


def _state_dict(tool_context: ToolContext, key: str) -> dict[str, Any]:
    value = tool_context.state.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _budget(tool_context: ToolContext) -> dict[str, int]:
    value = _state_dict(tool_context, REPORT_BUDGET_STATE_KEY)
    return {
        "search_calls": int(value.get("search_calls", 0)),
        "read_calls": int(value.get("read_calls", 0)),
        "output_chars": int(value.get("output_chars", 0)),
    }


def _charge(
    tool_context: ToolContext,
    budget: dict[str, int],
    *,
    call_type: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if budget["output_chars"] + size > MAX_TOTAL_OUTPUT_CHARS:
        return _error("report tool cumulative context budget exhausted", budget_exhausted=True)
    budget[f"{call_type}_calls"] += 1
    budget["output_chars"] += size
    tool_context.state[REPORT_BUDGET_STATE_KEY] = budget
    return None


def _manifest(
    report_ref: str, tool_context: ToolContext
) -> tuple[dict[str, Any] | None, dict | None]:
    manifest = tool_context.state.get(REPORT_STATE_KEY)
    if not isinstance(manifest, dict):
        return None, _error("report manifest is missing from session state")
    if manifest.get("report_ref") != report_ref:
        return None, _error("report_ref does not match the current session report")
    if not isinstance(manifest.get("artifact_name"), str):
        return None, _error("report manifest has no artifact_name")
    return manifest, None


def prepare_report_question(
    report_ref: str,
    question: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Authorize bounded report access for one non-KPI annual-report question."""
    _, error = _manifest(report_ref, tool_context)
    if error is not None:
        return error
    normalized_question = " ".join(question.split())
    if not normalized_question:
        return _error("question must not be empty")
    tool_context.state[REPORT_QUESTION_STATE_KEY] = {
        "report_ref": report_ref,
        "question": normalized_question,
    }
    tool_context.state[REPORT_BUDGET_STATE_KEY] = {}
    tool_context.state[SEARCH_RESULTS_STATE_KEY] = []
    tool_context.state[CURSORS_STATE_KEY] = {}
    return {
        "status": "success",
        "report_ref": report_ref,
        "question": normalized_question,
        "guidance": "Search using exact report terms, then read only the strongest evidence chunks.",
    }


def _has_report_access(report_ref: str, tool_context: ToolContext) -> bool:
    question_context = tool_context.state.get(REPORT_QUESTION_STATE_KEY)
    has_question_context = (
        isinstance(question_context, dict) and question_context.get("report_ref") == report_ref
    )
    has_kpi_knowledge = tool_context.state.get(KNOWLEDGE_STATE_KEY) is not None
    agent_name = getattr(tool_context, "agent_name", None)
    if agent_name == "kpi_worker":
        return has_kpi_knowledge
    if agent_name == "report_qa_worker":
        return has_question_context
    return has_kpi_knowledge or has_question_context


async def _load_chunks(
    report_ref: str, tool_context: ToolContext
) -> tuple[list[dict], dict | None]:
    manifest, error = _manifest(report_ref, tool_context)
    if error is not None:
        return [], error
    part = await tool_context.load_artifact(manifest["artifact_name"])
    if part is None or part.inline_data is None:
        return [], _error("report artifact is missing or is not binary data")
    try:
        text = part.inline_data.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], _error(f"report artifact is not valid UTF-8: {exc}")
    artifact_name = manifest["artifact_name"]
    mime_type = part.inline_data.mime_type or manifest.get("mime_type", "")
    if mime_type in {"text/markdown", "text/x-markdown"} or artifact_name.lower().endswith(
        (".md", ".markdown")
    ):
        chunks = _markdown_chunks(report_ref, text)
    else:
        try:
            chunks = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            return [], _error(f"report artifact is invalid JSONL: {exc}")
    for chunk in chunks:
        if not isinstance(chunk, dict) or not {
            "chunk_id",
            "page",
            "heading",
            "text",
        }.issubset(chunk):
            return [], _error("report artifact contains an invalid chunk")
        if len(str(chunk["text"])) > MAX_CHUNK_CHARS:
            return [], _error(f"chunk exceeds {MAX_CHUNK_CHARS} characters")
    return chunks, None


def _bounded_text(text: str) -> list[str]:
    remaining = text.strip()
    pieces: list[str] = []
    while remaining:
        if len(remaining) <= MAX_CHUNK_CHARS:
            pieces.append(remaining)
            break
        boundary = remaining.rfind("\n\n", 0, MAX_CHUNK_CHARS + 1)
        if boundary < MAX_CHUNK_CHARS // 2:
            boundary = MAX_CHUNK_CHARS
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    return pieces


def _markdown_chunks(report_ref: str, text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page, page_text in enumerate(PAGE_BREAK_PATTERN.split(text), start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        heading_match = re.search(r"(?m)^#{1,6}\s+(.+?)\s*$", page_text)
        heading = heading_match.group(1)[:200] if heading_match else "Annual Report"
        for chunk_index, piece in enumerate(_bounded_text(page_text)):
            chunks.append(
                {
                    "chunk_id": f"{report_ref}:p{page}:c{chunk_index}",
                    "page": page,
                    "heading": heading,
                    "text": piece,
                }
            )
    return chunks


def _terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.casefold())


def _ranked_matches(chunks: list[dict], query: str) -> list[tuple[int, dict, int]]:
    terms = _terms(query)
    phrase = " ".join(query.casefold().split())
    matches: list[tuple[int, dict, int]] = []
    for chunk in chunks:
        text = str(chunk["text"])
        folded_text = text.casefold()
        haystack = f"{chunk['heading']}\n{folded_text}".casefold()
        term_score = sum(haystack.count(term) for term in terms)
        phrase_score = haystack.count(phrase) * 10 if phrase else 0
        if term_score + phrase_score:
            first = min(
                (folded_text.find(term) for term in terms if term in folded_text),
                default=0,
            )
            matches.append((term_score + phrase_score, chunk, max(0, first - 150)))
    return sorted(matches, key=lambda item: (-item[0], int(item[1]["page"]), item[1]["chunk_id"]))


async def search_report(
    report_ref: str,
    query: str,
    cursor: str,
    limit: int,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search every report chunk while returning only bounded candidate snippets."""
    if not _has_report_access(report_ref, tool_context):
        return _error(
            "call GetKpiKnowledge or PrepareReportQuestion before searching the report"
        )
    if not query.strip():
        return _error("query must not be empty")
    if isinstance(limit, bool) or not 1 <= limit <= MAX_SEARCH_RESULTS:
        return _error(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")
    budget = _budget(tool_context)
    if budget["search_calls"] >= MAX_SEARCH_CALLS:
        return _error("report search-call budget exhausted", budget_exhausted=True)
    offset = 0
    cursors = _state_dict(tool_context, CURSORS_STATE_KEY)
    if cursor:
        cursor_data = cursors.get(cursor)
        if not isinstance(cursor_data, dict):
            return _error("invalid or expired search cursor")
        if cursor_data.get("report_ref") != report_ref or cursor_data.get("query") != query:
            return _error("search cursor does not match report_ref and query")
        offset = int(cursor_data["offset"])
    chunks, error = await _load_chunks(report_ref, tool_context)
    if error is not None:
        return error
    matches = _ranked_matches(chunks, query)
    page = matches[offset : offset + limit]
    items = []
    for score, chunk, start in page:
        text = str(chunk["text"])
        items.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "heading": chunk["heading"],
                "score": score,
                "snippet": text[start : start + MAX_SNIPPET_CHARS],
            }
        )
    next_cursor = None
    if offset + limit < len(matches):
        next_cursor = secrets.token_urlsafe(18)
        cursors[next_cursor] = {
            "report_ref": report_ref,
            "query": query,
            "offset": offset + limit,
        }
        tool_context.state[CURSORS_STATE_KEY] = cursors
    authorized = set(tool_context.state.get(SEARCH_RESULTS_STATE_KEY, []))
    authorized.update(item["chunk_id"] for item in items)
    tool_context.state[SEARCH_RESULTS_STATE_KEY] = sorted(authorized)
    response = {
        "status": "success",
        "scanned_chunks": len(chunks),
        "total_matches": len(matches),
        "items": items,
        "next_cursor": next_cursor,
    }
    return _charge(tool_context, budget, call_type="search", payload=response) or response


async def read_report_chunks(
    report_ref: str,
    chunk_ids: list[str],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Read at most three chunks previously returned by SearchReport."""
    if not _has_report_access(report_ref, tool_context):
        return _error(
            "call GetKpiKnowledge or PrepareReportQuestion before reading the report"
        )
    if not chunk_ids or len(chunk_ids) > MAX_READ_CHUNKS or len(set(chunk_ids)) != len(chunk_ids):
        return _error(f"chunk_ids must contain 1 to {MAX_READ_CHUNKS} unique ids")
    authorized = set(tool_context.state.get(SEARCH_RESULTS_STATE_KEY, []))
    unauthorized = [chunk_id for chunk_id in chunk_ids if chunk_id not in authorized]
    if unauthorized:
        return _error("chunks were not returned by SearchReport")
    budget = _budget(tool_context)
    if budget["read_calls"] >= MAX_READ_CALLS:
        return _error("report read-call budget exhausted", budget_exhausted=True)
    chunks, error = await _load_chunks(report_ref, tool_context)
    if error is not None:
        return error
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    if any(chunk_id not in by_id for chunk_id in chunk_ids):
        return _error("authorized chunk no longer exists in the report artifact")
    records = [by_id[chunk_id] for chunk_id in chunk_ids]
    body_chars = sum(len(str(record["text"])) for record in records)
    if body_chars > MAX_READ_CHARS:
        return _error(
            "selected chunks exceed the single-read context budget", budget_exhausted=True
        )
    response = {"status": "success", "chunks": records, "body_chars": body_chars}
    return _charge(tool_context, budget, call_type="read", payload=response) or response


search_report_tool = NamedFunctionTool(search_report, name="SearchReport")
read_report_chunks_tool = NamedFunctionTool(read_report_chunks, name="ReadReportChunks")
prepare_report_question_tool = NamedFunctionTool(
    prepare_report_question,
    name="PrepareReportQuestion",
)
