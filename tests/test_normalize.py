import pytest

from finground.models import NeedleAnswer
from finground.normalize import (
    detect_scale,
    normalize_needle_answer,
    normalize_value,
    parse_financial_number,
    validate_needle_evidence,
)


def test_accounting_parentheses_are_negative() -> None:
    assert parse_financial_number("$ (1,505)") == -1505


def test_scale_and_positive_outflow_rules() -> None:
    assert detect_scale("(in millions)", "revenue") == "millions"
    assert detect_scale("Amounts in AUD $'000", "revenue") == "thousands"
    assert detect_scale("EPS in cents", "eps_basic") == "currency_subunits_per_share"
    assert normalize_value("(25)", "millions", "capex") == 25_000_000
    assert normalize_value("1.08", "per_share", "eps_diluted") == pytest.approx(1.08)
    assert normalize_value("(0.19)", "currency_subunits_per_share", "eps_basic") == (
        pytest.approx(-0.0019)
    )


def test_needle_answer_normalizes_provider_format_variants() -> None:
    answer = NeedleAnswer.model_validate(
        {
            "found": True,
            "value": 123_000,
            "value_verbatim": "123",
            "unit_scale": "thousands of dollars",
            "page": "[Page 46]",
        }
    )

    assert answer.unit_scale == "thousands"
    assert answer.page == 46


@pytest.mark.parametrize(
    ("kpi", "value", "value_verbatim", "unit_scale", "expected"),
    [
        ("accounts_receivable", 2_104_555, "2,104,555", "thousands", 2_104_555_000),
        ("capex", 222.8, "$222.8 million", "millions", 222_800_000),
        ("cash_and_equivalents", 204_213, "$ 204,213", "thousands", 204_213_000),
        ("eps_basic", 19.08, "19.08", "per_share", 19.08),
        (
            "eps_basic",
            -0.19,
            "(0.19)",
            "currency_subunits_per_share",
            -0.0019,
        ),
        ("dividends_paid", -79_117, "(79,117)", "thousands", 79_117_000),
    ],
)
def test_normalize_needle_answer_recomputes_single_source_number(
    kpi: str,
    value: float,
    value_verbatim: str,
    unit_scale: str,
    expected: float,
) -> None:
    answer = NeedleAnswer(
        found=True,
        value=value,
        value_verbatim=value_verbatim,
        unit_scale=unit_scale,
        page=10,
    )

    normalized, trace = normalize_needle_answer(answer, kpi)

    assert normalized.value == pytest.approx(expected)
    assert trace["status"] in {"verified", "corrected"}
    assert trace["computed_value"] == pytest.approx(expected)


def test_normalize_needle_answer_abstains_on_multiple_source_numbers() -> None:
    answer = NeedleAnswer(
        found=True,
        value=491_761_000,
        value_verbatim="Depreciation 284,997 and amortization 206,764",
        unit_scale="thousands",
        page=20,
    )

    normalized, trace = normalize_needle_answer(answer, "depreciation_amortization")

    assert normalized.found is False
    assert normalized.value is None
    assert normalized.value_verbatim is None
    assert trace["status"] == "unverified"
    assert trace["reason"] == "value_verbatim_not_single_number"


def test_explicit_scale_in_verbatim_wins_over_provider_scale() -> None:
    answer = NeedleAnswer(
        found=True,
        value=222.8,
        value_verbatim="$222.8 million",
        unit_scale="units",
        page=10,
    )

    normalized, trace = normalize_needle_answer(answer, "capex")

    assert normalized.value == 222_800_000
    assert normalized.unit_scale == "millions"
    assert trace["scale_source"] == "value_verbatim"


def test_evidence_validation_rejects_a_derived_number_not_printed_on_page() -> None:
    answer = NeedleAnswer(
        found=True,
        value=491_761_000,
        value_verbatim="491,761",
        unit_scale="thousands",
        page=46,
    )
    page_text = "Depreciation 284,997 | Amortization 206,764"

    validated, trace = validate_needle_evidence(answer, page_text)

    assert validated.found is False
    assert validated.value is None
    assert trace == {"status": "fail", "reason": "verbatim_number_not_on_cited_page"}


def test_evidence_validation_accepts_formatting_variants_of_printed_number() -> None:
    answer = NeedleAnswer(
        found=True,
        value=204_213_000,
        value_verbatim="$ 204,213",
        unit_scale="thousands",
        page=46,
    )

    validated, trace = validate_needle_evidence(
        answer, "Cash and cash equivalents | $204,213 | $889,786"
    )

    assert validated == answer
    assert trace == {"status": "pass"}
