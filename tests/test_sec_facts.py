from finground.sec_facts import (
    _ordered_cik_candidates,
    _registrant_name,
    _report_cik,
    extract_sec_kpis,
)


def _fact(value: float, *, start: str | None = None) -> dict:
    fact = {
        "val": value,
        "end": "2017-12-31",
        "filed": "2018-02-20",
        "form": "10-K",
        "fp": "FY",
    }
    if start is not None:
        fact["start"] = start
    return fact


def test_sec_waterfall_prefers_first_concept_and_derives_liabilities() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(90, start="2017-01-01")]}},
                "SalesRevenueNet": {"units": {"USD": [_fact(80, start="2017-01-01")]}},
                "Assets": {"units": {"USD": [_fact(100)]}},
                "StockholdersEquity": {"units": {"USD": [_fact(40)]}},
            }
        }
    }

    values = extract_sec_kpis(facts, 2017)

    assert values["revenue"] == {"value": 90.0, "concept": "Revenues"}
    assert values["total_liabilities"] == {
        "value": 60.0,
        "concept": "sum:Assets-StockholdersEquity",
    }


def test_sec_flow_rejects_quarter_duration() -> None:
    facts = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [_fact(12, start="2017-10-01")]}}}}
    }

    assert "net_income" not in extract_sec_kpis(facts, 2017)


def test_sec_waterfall_does_not_use_noncanonical_concept_fallbacks() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization": {
                    "units": {"USD": [_fact(12, start="2017-01-01")]}
                },
                "InterestExpenseNonoperating": {"units": {"USD": [_fact(3, start="2017-01-01")]}},
                "IncomeTaxExpenseBenefitContinuingOperations": {
                    "units": {"USD": [_fact(2, start="2017-01-01")]}
                },
                "InterestReceivable": {"units": {"USD": [_fact(4)]}},
                "CashEquivalentsAtCarryingValue": {"units": {"USD": [_fact(5)]}},
                "LiabilitiesAndStockholdersEquity": {"units": {"USD": [_fact(20)]}},
            }
        }
    }

    values = extract_sec_kpis(facts, 2017)

    assert "cost_of_revenue" not in values
    assert "interest_expense" not in values
    assert "income_tax_expense" not in values
    assert values["accounts_receivable"] == {
        "value": 4.0,
        "concept": "InterestReceivable",
    }
    assert "cash_and_equivalents" not in values
    assert "total_assets" not in values


def test_report_identity_parses_explicit_cik_and_historical_registrant() -> None:
    text = """\
Corporate Issuer CIK: 738214

# APACHE CORPORATION
(Exact name of Registrant as specified in its charter)
"""

    assert _report_cik(text) == "0000738214"
    assert _registrant_name(text) == "APACHE CORPORATION"


def test_cik_resolution_prefers_exact_ticker_before_fuzzy_historical_hits() -> None:
    assert _ordered_cik_candidates(
        explicit_cik=None,
        current_cik="0000123456",
        historical_ciks=["0000999999", "0000123456"],
    ) == ["0000123456", "0000999999"]


def test_ifrs_annual_facts_are_used_when_us_gaap_is_absent() -> None:
    facts = {
        "facts": {
            "ifrs-full": {
                "Assets": {
                    "units": {
                        "CAD": [
                            {
                                **_fact(125),
                                "form": "6-K",
                            }
                        ]
                    }
                },
                "CashFlowsFromUsedInOperatingActivities": {
                    "units": {
                        "CAD": [
                            {
                                **_fact(30, start="2017-01-01"),
                                "form": "6-K",
                            }
                        ]
                    }
                },
            }
        }
    }

    values = extract_sec_kpis(facts, 2017)

    assert values["total_assets"]["value"] == 125
    assert values["total_assets"]["concept"] == "ifrs-full:Assets"
    assert values["operating_cash_flow"]["value"] == 30
