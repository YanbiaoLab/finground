from finground.sec_facts import (
    _company_identity_score,
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


def test_sec_flow_prefers_fact_from_target_year_filing_over_later_comparative() -> None:
    original = {
        **_fact(50_153_000, start="2017-01-01"),
        "fy": 2017,
        "filed": "2018-03-29",
    }
    later_comparative = {
        **_fact(48_899_000, start="2017-01-01"),
        "fy": 2018,
        "filed": "2019-03-28",
    }
    facts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [original, later_comparative]}
                }
            }
        }
    }

    assert extract_sec_kpis(facts, 2017)["operating_cash_flow"]["value"] == 50_153_000


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


def test_report_identity_prefers_company_in_report_title_over_later_exhibit_name() -> None:
    text = """\
## Manhattan Bridge Capital Year 2017 Achievements

Exhibit for Webster Business Credit Corporation
(Exact name of Registrant as specified in its charter)
"""

    assert _registrant_name(text) == "Manhattan Bridge Capital"


def test_report_identity_reads_plain_company_line_before_annual_report_title() -> None:
    text = """\
Barnwell Industries, Inc.

2017 Annual Report

## FINANCIAL AND OPERATING HIGHLIGHTS
"""

    assert _registrant_name(text) == "Barnwell Industries, Inc."


def test_report_identity_reads_company_before_annual_report_and_accounts() -> None:
    text = """\
![](cover.jpg)

CORE LABORATORIES N.V.

ANNUAL REPORT AND ACCOUNTS
December 31, 2019
"""

    assert _registrant_name(text) == "CORE LABORATORIES N.V."


def test_report_identity_uses_early_company_heading_before_later_legal_names() -> None:
    text = """\
Working Together
2020 ANNUAL REPORT
## Stepan

Later discussion of PQ Corporation.
"""

    assert _registrant_name(text) == "Stepan"


def test_report_identity_prefers_exact_registrant_over_glossy_all_caps_heading() -> None:
    text = """\
## STRONG FINANCIAL RESULTS

Net Sales $3.3B

Central Garden & Pet Company

(Exact name of registrant as specified in its charter)
"""

    assert _registrant_name(text) == "Central Garden & Pet Company"


def test_report_identity_prefers_exact_registrant_over_marketing_annual_title() -> None:
    text = """\
FUTURE

2017 ANNUAL REPORT

# HELIX ENERGY SOLUTIONS GROUP, INC.
(Exact name of registrant as specified in its charter)
"""

    assert _registrant_name(text) == "HELIX ENERGY SOLUTIONS GROUP, INC."


def test_report_identity_prefers_mda_company_over_cover_slogan() -> None:
    text = """\
## BUILDING
OUR FUTURE

## MANAGEMENT'S DISCUSSION AND ANALYSIS

The following is management's discussion and analysis of the operating and financial results
of Baytex Energy Corp. for the years ended December 31, 2017 and 2016.
"""

    assert _registrant_name(text) == "Baytex Energy Corp."


def test_report_identity_combines_multiline_company_name_before_ticker() -> None:
    text = """\
# ANNUAL
REPORT
DECEMBER 31
2022

MANHATTAN
BRIDGE CAPITAL
NASDAQ: LOAN
"""

    assert _registrant_name(text) == "MANHATTAN BRIDGE CAPITAL"


def test_cik_resolution_prefers_current_ticker_before_fuzzy_historical_hits() -> None:
    assert _ordered_cik_candidates(
        explicit_cik=None,
        current_cik="0000123456",
        historical_ciks=["0000999999", "0000123456"],
    ) == ["0000123456", "0000999999"]


def test_company_identity_prefers_registrant_name_over_reused_ticker() -> None:
    assert _company_identity_score(
        ticker="APA",
        report_entity="APACHE CORPORATION",
        facts_entity="APACHE CORPORATION",
    ) > _company_identity_score(
        ticker="APA",
        report_entity="APACHE CORPORATION",
        facts_entity="APA CORPORATION",
    )


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
