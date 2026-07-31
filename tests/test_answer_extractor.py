import json

from finground.benchmark.answer_extractor import (
    ANSWER_EXTRACTOR_INSTRUCTION,
    create_answer_extractor_agent,
    render_agent_answer,
)
from finground.models import ReportExtraction


def test_rendered_answer_round_trips_through_ledger_schema() -> None:
    extraction = {
        "ticker": "ACME",
        "reporting_currency": "USD",
        "units_note": "raw units",
        "kpis": [
            {"kpi": "revenue", "fiscal_year": 2023, "value": 123.0},
            {"kpi": "gross_profit", "fiscal_year": 2023, "value": None},
        ],
    }

    rendered = render_agent_answer(extraction)

    assert ReportExtraction.model_validate(json.loads(rendered)).model_dump() == extraction


def test_answer_extractor_is_schema_constrained_and_has_no_tools() -> None:
    agent = create_answer_extractor_agent(model_name="gemini-3-flash-preview")

    assert agent.output_schema is ReportExtraction
    assert agent.tools == []
    assert "Never infer, calculate, repair" in ANSWER_EXTRACTOR_INSTRUCTION


def test_qwen_answer_extractor_uses_strict_json_schema_without_tool_choice() -> None:
    agent = create_answer_extractor_agent(model_name="qwen-local")
    args = agent.model._additional_args

    assert args["response_format"]["type"] == "json_schema"
    assert args["response_format"]["json_schema"]["strict"] is True
    assert "tool_choice" not in args
