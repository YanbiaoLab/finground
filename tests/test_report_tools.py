import asyncio
import json
from types import SimpleNamespace

from google.genai import types

from finground.kpi_catalog import get_kpi_knowledge
from finground.report_tools import (
    MAX_READ_CALLS,
    MAX_SEARCH_RESULTS,
    MAX_SNIPPET_CHARS,
    prepare_report_question,
    read_report_chunks,
    search_report,
)


class FakeToolContext(SimpleNamespace):
    async def load_artifact(self, filename: str):
        assert filename == self.state["report"]["artifact_name"]
        return self.artifact


def _context(chunks: list[dict]) -> FakeToolContext:
    body = "\n".join(json.dumps(chunk) for chunk in chunks).encode()
    return FakeToolContext(
        state={
            "report": {
                "report_ref": "ACME_2025",
                "artifact_name": "reports/ACME_2025.jsonl",
            }
        },
        artifact=types.Part.from_bytes(data=body, mime_type="application/x-ndjson"),
    )


def _chunk(index: int, text: str) -> dict:
    return {
        "chunk_id": f"ACME_2025:p{index}:c0",
        "page": index,
        "heading": "Annual Report",
        "text": text,
    }


def test_search_requires_knowledge_and_scans_evidence_at_report_end() -> None:
    chunks = [_chunk(index, "ordinary disclosure") for index in range(2_000)]
    chunks.append(_chunk(2_000, "Consolidated revenue for 2025 was USD 123 million."))
    context = _context(chunks)

    blocked = asyncio.run(search_report("ACME_2025", "revenue", "", 8, context))
    get_kpi_knowledge("revenue", context)
    result = asyncio.run(search_report("ACME_2025", "revenue", "", 8, context))

    assert blocked["status"] == "error"
    assert result["scanned_chunks"] == 2_001
    assert result["items"][0]["chunk_id"] == "ACME_2025:p2000:c0"
    assert len(result["items"]) <= MAX_SEARCH_RESULTS
    assert len(result["items"][0]["snippet"]) <= MAX_SNIPPET_CHARS


def test_general_question_requires_preparation_before_report_access() -> None:
    context = _context([_chunk(10, "The company depends on a limited number of suppliers.")])
    context.agent_name = "report_qa_worker"

    blocked = asyncio.run(search_report("ACME_2025", "suppliers", "", 8, context))
    prepared = prepare_report_question(
        "ACME_2025",
        "  What supplier risks does the company disclose?  ",
        context,
    )
    result = asyncio.run(search_report("ACME_2025", "suppliers", "", 8, context))

    assert blocked["status"] == "error"
    assert "PrepareReportQuestion" in blocked["error"]
    assert prepared["question"] == "What supplier risks does the company disclose?"
    assert result["status"] == "success"
    assert result["items"][0]["chunk_id"] == "ACME_2025:p10:c0"


def test_invalid_jsonl_returns_stable_error_without_parser_details() -> None:
    context = _context([])
    context.artifact = types.Part.from_bytes(
        data=b'{"chunk_id":', mime_type="application/x-ndjson"
    )
    get_kpi_knowledge("revenue", context)

    result = asyncio.run(search_report("ACME_2025", "revenue", "", 8, context))

    assert result["status"] == "error"
    assert result["error"] == "report artifact is invalid JSONL"


def test_kpi_knowledge_does_not_replace_general_question_preparation() -> None:
    context = _context([_chunk(10, "The company depends on a limited number of suppliers.")])
    context.agent_name = "report_qa_worker"
    get_kpi_knowledge("revenue", context)

    result = asyncio.run(search_report("ACME_2025", "suppliers", "", 8, context))

    assert result["status"] == "error"
    assert "PrepareReportQuestion" in result["error"]


def test_general_question_preparation_rejects_another_report() -> None:
    context = _context([_chunk(1, "ordinary disclosure")])

    result = prepare_report_question("OTHER_2025", "What are the risks?", context)

    assert result["status"] == "error"
    assert "report_ref" in result["error"]


def test_cursor_is_opaque_and_chunks_must_be_authorized() -> None:
    context = _context([_chunk(index, f"revenue row {index}") for index in range(12)])
    get_kpi_knowledge("revenue", context)
    first = asyncio.run(search_report("ACME_2025", "revenue", "", 3, context))

    invalid = asyncio.run(search_report("ACME_2025", "revenue", "invented", 3, context))
    second = asyncio.run(search_report("ACME_2025", "revenue", first["next_cursor"], 3, context))
    unauthorized = asyncio.run(read_report_chunks("ACME_2025", ["unknown"], context))
    allowed = asyncio.run(
        read_report_chunks("ACME_2025", [second["items"][0]["chunk_id"]], context)
    )

    assert invalid["status"] == "error"
    assert first["items"][0]["chunk_id"] != second["items"][0]["chunk_id"]
    assert unauthorized["status"] == "error"
    assert allowed["status"] == "success"


def test_cumulative_report_output_budget_is_enforced() -> None:
    context = _context([_chunk(index, "revenue " + "x" * 5_980) for index in range(3)])
    get_kpi_knowledge("revenue", context)
    found = asyncio.run(search_report("ACME_2025", "revenue", "", 3, context))
    chunk_ids = [item["chunk_id"] for item in found["items"]]

    responses = []
    for _ in range(MAX_READ_CALLS):
        response = asyncio.run(read_report_chunks("ACME_2025", chunk_ids, context))
        responses.append(response)
        if response["status"] == "error":
            break

    exhausted = responses[-1]
    assert responses[0]["status"] == "success"
    assert exhausted["status"] == "error"
    assert exhausted["budget_exhausted"] is True
    assert "cumulative context budget" in exhausted["error"]


def test_markdown_artifact_is_split_and_searchable_outside_model_context() -> None:
    pages = ["# Cover\n\nordinary disclosure", "# Financial Statements\n\n" + "x" * 6_500]
    pages.append("# Consolidated Statements of Operations\n\nRevenue was USD 123 million.")
    body = "\n\n<--- Page Split --->\n\n".join(pages).encode()
    context = FakeToolContext(
        state={
            "report": {
                "report_ref": "AAP_2017",
                "artifact_name": "AAP_2017.md",
            }
        },
        artifact=types.Part.from_bytes(data=body, mime_type="text/markdown"),
    )
    get_kpi_knowledge("revenue", context)

    result = asyncio.run(search_report("AAP_2017", "revenue", "", 8, context))

    assert result["status"] == "success"
    assert result["scanned_chunks"] == 4
    assert result["items"][0]["chunk_id"] == "AAP_2017:p3:c0"
