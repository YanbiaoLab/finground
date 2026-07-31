"""LLM judge that converts a user-visible agent answer to LEDGER's schema."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.agents.common import SETTINGS, create_adk_model
from finground.kpis import KPI_KEYS
from finground.models import ReportExtraction

ANSWER_EXTRACTOR_APP_NAME = "finground_answer_extractor"
ANSWER_EXTRACTOR_NAME = "ledger_answer_extractor"
ANSWER_EXTRACTOR_PROMPT_VERSION = "ledger-answer-extractor-v1"

ANSWER_EXTRACTOR_INSTRUCTION = f"""Convert the supplied assistant answer into the LEDGER
ReportExtraction schema. This is an evaluator, not a financial analyst.

Rules:
- Extract only values explicitly asserted in the assistant answer.
- Never infer, calculate, repair, or add a missing KPI.
- Preserve signs and already-normalized raw units exactly.
- Use the supplied ticker when the answer does not state one.
- Keep only canonical KPI keys: {", ".join(KPI_KEYS)}.
- If the answer is empty or unusable, return an empty kpis list.
"""


def render_agent_answer(extraction: dict) -> str:
    """Render the state-backed result as the answer exposed to an evaluator/user."""
    validated = ReportExtraction.model_validate(extraction)
    return validated.model_dump_json(exclude_none=False)


def create_answer_extractor_agent(*, model_name: str | None = None) -> Agent:
    """Create a schema-constrained judge with no tools or report access."""
    return Agent(
        name=ANSWER_EXTRACTOR_NAME,
        model=create_adk_model(
            model_name or SETTINGS.model,
            json_output=True,
            json_schema=ReportExtraction.model_json_schema(),
        ),
        instruction=ANSWER_EXTRACTOR_INSTRUCTION,
        include_contents="none",
        output_schema=ReportExtraction,
        generate_content_config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=8_192,
        ),
    )


async def extract_ledger_answer(
    answer: str,
    *,
    ticker: str,
    model_name: str | None = None,
) -> ReportExtraction:
    """Use an isolated LLM call to parse one answer into LEDGER's exact contract."""
    session_service = InMemorySessionService()
    session_id = "judge-" + hashlib.sha256(answer.encode()).hexdigest()[:16]
    await session_service.create_session(
        app_name=ANSWER_EXTRACTOR_APP_NAME,
        user_id="benchmark",
        session_id=session_id,
    )
    runner = Runner(
        agent=create_answer_extractor_agent(model_name=model_name),
        app_name=ANSWER_EXTRACTOR_APP_NAME,
        session_service=session_service,
    )
    final_text: str | None = None
    prompt = f"Ticker: {ticker}\n\nAssistant answer:\n{answer}"
    async for event in runner.run_async(
        user_id="benchmark",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "\n".join(
                part.text for part in event.content.parts if part.text is not None
            )
    await runner.close()
    if not final_text:
        raise RuntimeError("answer extractor returned no final response")
    return ReportExtraction.model_validate(json.loads(final_text))


def extract_ledger_answer_sync(
    answer: str,
    *,
    ticker: str,
    model_name: str | None = None,
) -> ReportExtraction:
    """Synchronous wrapper used by benchmark commands."""
    return asyncio.run(extract_ledger_answer(answer, ticker=ticker, model_name=model_name))


async def extract_output_answers(
    *,
    output_dir: Path,
    model_name: str | None = None,
) -> Path:
    """Judge every saved user-visible answer and write isolated prediction records."""
    raw_dir = output_dir / "raw"
    judged_dir = output_dir / "judged"
    judged_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        answer = record.get("answer")
        if not isinstance(answer, str):
            extraction = record.get("extraction")
            if not isinstance(extraction, dict):
                continue
            answer = render_agent_answer(extraction)
        judged = await extract_ledger_answer(
            answer,
            ticker=str(record.get("ticker") or ""),
            model_name=model_name,
        )
        judged_record = {
            **record,
            "extraction": judged.model_dump(mode="json"),
            "answer_extractor": {
                "model": model_name or SETTINGS.model,
                "prompt_version": ANSWER_EXTRACTOR_PROMPT_VERSION,
                "source": "answer",
            },
        }
        temporary = (judged_dir / path.name).with_suffix(".json.tmp")
        temporary.write_text(json.dumps(judged_record, indent=2), encoding="utf-8")
        temporary.replace(judged_dir / path.name)
    return judged_dir


def extract_output_answers_sync(
    *,
    output_dir: Path,
    model_name: str | None = None,
) -> Path:
    """Synchronous batch wrapper for the scoring CLI."""
    return asyncio.run(extract_output_answers(output_dir=output_dir, model_name=model_name))
