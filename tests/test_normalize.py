import pytest

from finground.normalize import detect_scale, normalize_value, parse_financial_number


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
