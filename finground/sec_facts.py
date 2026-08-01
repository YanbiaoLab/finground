"""SEC Company Facts retrieval aligned with LEDGER's XBRL tag waterfall."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SEC_FACTS_STATE_KEY = "sec_company_facts"
SEC_FACTS_ENABLED_STATE_KEY = "sec_company_facts_enabled"
_CACHE_ROOT = Path(os.environ.get("FINGROUND_SEC_CACHE", ".finground-cache/sec"))
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0

# kind, unit, ordered tags, ordered all-required summation fallbacks.
_DEFS: dict[str, tuple[str, str, tuple[str, ...], tuple[tuple[str, ...], ...]]] = {
    "revenue": (
        "flow",
        "USD",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ),
        (),
    ),
    "cost_of_revenue": (
        "flow",
        "USD",
        (
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfServices",
        ),
        (("CostOfGoodsSold", "CostOfServices"),),
    ),
    "gross_profit": ("flow", "USD", ("GrossProfit",), ()),
    "rd_expense": ("flow", "USD", ("ResearchAndDevelopmentExpense",), ()),
    "sga_expense": (
        "flow",
        "USD",
        ("SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"),
        (),
    ),
    "operating_income": ("flow", "USD", ("OperatingIncomeLoss",), ()),
    "interest_expense": (
        "flow",
        "USD",
        (
            "InterestExpense",
            "InterestExpenseDebt",
        ),
        (),
    ),
    "income_tax_expense": (
        "flow",
        "USD",
        ("IncomeTaxExpenseBenefit",),
        (),
    ),
    "net_income": (
        "flow",
        "USD",
        ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        (),
    ),
    "eps_basic": ("flow", "USD/shares", ("EarningsPerShareBasic",), ()),
    "eps_diluted": ("flow", "USD/shares", ("EarningsPerShareDiluted",), ()),
    "total_assets": ("stock", "USD", ("Assets",), ()),
    "total_liabilities": (
        "stock",
        "USD",
        ("Liabilities",),
        (
            ("LiabilitiesCurrent", "LiabilitiesNoncurrent"),
            ("Assets", "-StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ("Assets", "-StockholdersEquity"),
        ),
    ),
    "stockholders_equity": ("stock", "USD", ("StockholdersEquity",), ()),
    "stockholders_equity_incl_nci": (
        "stock",
        "USD",
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),
        (),
    ),
    "cash_and_equivalents": (
        "stock",
        "USD",
        ("CashAndCashEquivalentsAtCarryingValue",),
        (),
    ),
    "cash_incl_restricted": (
        "stock",
        "USD",
        (
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
        ),
        (),
    ),
    "long_term_debt_total": (
        "stock",
        "USD",
        ("LongTermDebt",),
        (),
    ),
    "long_term_debt_noncurrent": ("stock", "USD", ("LongTermDebtNoncurrent",), ()),
    "long_term_debt_current": (
        "stock",
        "USD",
        ("LongTermDebtCurrent",),
        (),
    ),
    "short_term_borrowings": (
        "stock",
        "USD",
        ("ShortTermBorrowings",),
        (),
    ),
    "inventory": ("stock", "USD", ("InventoryNet",), ()),
    "accounts_receivable": (
        "stock",
        "USD",
        ("AccountsReceivableNetCurrent", "InterestReceivable"),
        (),
    ),
    "accounts_payable": (
        "stock",
        "USD",
        ("AccountsPayableCurrent",),
        (),
    ),
    "shares_outstanding": (
        "stock",
        "shares",
        (
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
        ),
        (),
    ),
    "operating_cash_flow": (
        "flow",
        "USD",
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        (),
    ),
    "investing_cash_flow": ("flow", "USD", ("NetCashProvidedByUsedInInvestingActivities",), ()),
    "financing_cash_flow": ("flow", "USD", ("NetCashProvidedByUsedInFinancingActivities",), ()),
    "capex": (
        "flow",
        "USD",
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        (),
    ),
    "depreciation_amortization": (
        "flow",
        "USD",
        ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"),
        (),
    ),
    "dividends_paid": (
        "flow",
        "USD",
        ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
        (),
    ),
}

_IFRS_DEFS: dict[str, tuple[str, tuple[str, ...]]] = {
    "gross_profit": ("flow", ("GrossProfit",)),
    "rd_expense": ("flow", ("ResearchAndDevelopmentExpense",)),
    "operating_income": ("flow", ("ProfitLossFromOperatingActivities",)),
    "total_assets": ("stock", ("Assets",)),
    "total_liabilities": ("stock", ("Liabilities",)),
    "stockholders_equity": ("stock", ("Equity",)),
    "cash_and_equivalents": ("stock", ("CashAndCashEquivalents",)),
    "long_term_debt_noncurrent": (
        "stock",
        ("NoncurrentBorrowings", "BorrowingsNoncurrent"),
    ),
    "long_term_debt_current": ("stock", ("CurrentBorrowings", "BorrowingsCurrent")),
    "inventory": ("stock", ("Inventories",)),
    "accounts_receivable": (
        "stock",
        ("CurrentTradeReceivables", "TradeAndOtherCurrentReceivables"),
    ),
    "accounts_payable": ("stock", ("TradeAndOtherPayables", "CurrentTradePayables")),
    "operating_cash_flow": ("flow", ("CashFlowsFromUsedInOperatingActivities",)),
    "investing_cash_flow": ("flow", ("CashFlowsFromUsedInInvestingActivities",)),
    "financing_cash_flow": ("flow", ("CashFlowsFromUsedInFinancingActivities",)),
    "depreciation_amortization": (
        "flow",
        (
            "AdjustmentsForDepreciationExpense",
            "DepreciationDepletionAndAmortisationExpense",
            "DepreciationExpense",
        ),
    ),
    "dividends_paid": ("flow", ("DividendsPaid",)),
}


def _get_json(url: str) -> dict[str, Any]:
    global _LAST_REQUEST
    last_error: IncompleteRead | None = None
    for _attempt in range(3):
        with _LOCK:
            delay = 0.12 - (time.monotonic() - _LAST_REQUEST)
            if delay > 0:
                time.sleep(delay)
            request = Request(
                url,
                headers={
                    "User-Agent": os.environ.get(
                        "SEC_USER_AGENT", "FinGround research finground@example.invalid"
                    ),
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=30) as response:
                    data = json.load(response)
            except IncompleteRead as error:
                last_error = error
                continue
            _LAST_REQUEST = time.monotonic()
            return data
    assert last_error is not None
    raise last_error


def _cached_json(path: Path, url: str) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)
    return data


def _report_cik(text: str) -> str | None:
    match = re.search(
        r"(?i)(?:corporate\s+issuer\s+)?CIK\s*[:#]?\s*0*(\d{4,10})",
        text[:40_000],
    )
    return f"{int(match.group(1)):010d}" if match else None


def _registrant_name(text: str) -> str | None:
    cover = text[:40_000]
    plain_report_title = re.search(
        r"(?im)^\s*([A-Z][^\n]{2,100}?)\s*(?:\r?\n){1,3}\s*"
        r"(?:\d{4}\s+Annual Report|Annual Report(?: and Accounts)?)\s*$",
        cover[:2_000],
    )
    if plain_report_title:
        candidate = plain_report_title.group(1).strip(" #:-")
        if candidate.casefold() not in {"annual", "financial", "working together"}:
            return candidate
    ticker_cover_name = re.search(
        r"(?m)^([A-Z][A-Z &'.-]{2,60})\s*\n"
        r"([A-Z][A-Z &'.-]{2,60})\s*\n(?:NASDAQ|NYSE|AMEX)\s*:",
        cover[:2_000],
    )
    if ticker_cover_name:
        return " ".join(ticker_cover_name.groups())
    titled_company = re.search(
        r"(?im)^#{1,3}\s+(.{3,100}?)"
        r"(?:\s+Year\s+\d{4}\b|\s+\d{4}\s+Annual Report\b|\s+Annual Report\b)",
        cover[:4_000],
    )
    if titled_company:
        candidate = titled_company.group(1).strip(" #:-")
        if candidate.casefold() not in {"annual", "fiscal", "financial"}:
            return candidate
    for match in re.finditer(r"(?im)^#{1,3}\s+(.{2,100}?)\s*$", cover[:2_000]):
        candidate = re.sub(
            r"\s+\d{4}\s+(?:achievements|annual report)\s*$",
            "",
            match.group(1),
            flags=re.IGNORECASE,
        ).strip(" #:-")
        lowered = candidate.casefold()
        if any(
            phrase in lowered
            for phrase in (
                "annual",
                "annual report",
                "table of contents",
                "financial highlights",
                "dear ",
                "working together",
                "letter to",
                "securities and exchange commission",
                "edgar filing",
            )
        ):
            continue
        words = candidate.split()
        looks_like_company = (
            len(words) == 1
            or candidate.isupper()
            or bool(
                re.search(
                    r"(?i)\b(?:inc\.?|corp\.?|corporation|company|limited|ltd\.?|plc)\b",
                    candidate,
                )
            )
        )
        if 1 <= len(words) <= 8 and looks_like_company:
            return candidate
    exact_name = re.search(
        r"(?im)^(?:#+\s*)?([A-Z][A-Z0-9&'.,()/ -]{3,100})\s*$"
        r"(?:(?:\r?\n){1,3})(?:\(?Exact name of (?:the )?[Rr]egistrant)",
        cover,
    )
    if exact_name:
        return exact_name.group(1).strip(" #")
    heading_name = re.search(
        r"(?im)^#{1,3}\s+([A-Z][A-Z0-9&'., /-]{2,100}"
        r"(?:INC\.?|CORP\.?|CORPORATION|INCORPORATED|LIMITED|LTD\.?|N\.V\.))\s*$",
        cover,
    )
    if heading_name:
        return heading_name.group(1).strip()
    names = re.findall(
        r"\b([A-Z][A-Za-z0-9&'., -]{2,80}?"
        r"(?:Corporation|Incorporated|Inc\.?|Corp\.?|Limited|Ltd\.?|N\.V\.))\b",
        cover,
    )
    if not names:
        return None
    name = names[0].strip()
    if name.casefold().startswith(("based in ", "headquartered in ")):
        name = name.rsplit(",", 1)[-1].strip()
    return name


_ENTITY_STOP_WORDS = {
    "the",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "limited",
    "ltd",
    "lp",
    "llc",
    "nv",
    "plc",
    "holdings",
}


def _entity_tokens(name: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", name.casefold())
        if token not in _ENTITY_STOP_WORDS and len(token) > 1
    }


def _company_identity_score(
    *,
    ticker: str,
    report_entity: str | None,
    facts_entity: str,
) -> int:
    """Rank Company Facts candidates by report-name identity before ticker reuse."""
    facts_tokens = _entity_tokens(facts_entity)
    if not facts_tokens:
        return 0
    overlap = len(_entity_tokens(report_entity) & facts_tokens) if report_entity is not None else 0
    ticker_match = ticker.casefold() in facts_tokens
    return overlap * 100 + int(ticker_match)


def _historical_ciks(ticker: str, year: int, report_text: str) -> list[str]:
    entity = _registrant_name(report_text)
    if entity is None:
        return []
    query = urlencode(
        {
            "q": f'"{entity}"',
            "dateRange": "custom",
            "startdt": f"{year}-01-01",
            "enddt": f"{year + 1}-12-31",
            "from": 0,
            "size": 40,
        }
    )
    entity_digest = hashlib.sha256(entity.casefold().encode()).hexdigest()[:10]
    cache_path = _CACHE_ROOT / "efts" / f"{ticker.upper()}_{year}_{entity_digest}_v2.json"
    result = _cached_json(cache_path, f"{_EFTS_URL}?{query}")
    entity_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", entity.casefold())
        if token
        not in {
            "the",
            "inc",
            "incorporated",
            "corp",
            "corporation",
            "limited",
            "ltd",
            "nv",
        }
    }
    ignored_name_tokens = {
        "the",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "limited",
        "ltd",
        "nv",
        "cik",
    }
    scored: list[tuple[int, str]] = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        display = " ".join(source.get("display_names", []))
        display_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", display.casefold())
            if token not in ignored_name_tokens
            and token != ticker.casefold()
            and not token.isdigit()
        }
        overlap = len(entity_tokens & display_tokens)
        for cik in source.get("ciks", []):
            if overlap:
                identity_score = overlap * 100 - len(display_tokens - entity_tokens)
                scored.append((identity_score, f"{int(cik):010d}"))
    return [cik for _score, cik in sorted(set(scored), reverse=True)]


def _ordered_cik_candidates(
    *,
    explicit_cik: str | None,
    current_cik: str | None,
    historical_ciks: list[str],
) -> list[str]:
    """Prefer explicit identity, then current mapping, while retaining historical candidates."""
    candidates = [
        *([explicit_cik] if explicit_cik else []),
        *([current_cik] if current_cik else []),
        *historical_ciks,
    ]
    return list(dict.fromkeys(candidates))


def _fiscal_year(entry: dict[str, Any]) -> int | None:
    try:
        end = datetime.strptime(str(entry.get("end")), "%Y-%m-%d").date()
    except ValueError:
        return None
    return end.year - 1 if end.month == 1 else end.year


def _pick(entries: list[dict[str, Any]], year: int, kind: str) -> dict[str, Any] | None:
    candidates = []
    for entry in entries:
        if entry.get("fp") != "FY" or not str(entry.get("form", "")).startswith(("10-K", "20-F")):
            continue
        if _fiscal_year(entry) != year:
            continue
        if kind == "flow":
            try:
                span = (
                    datetime.strptime(entry["end"], "%Y-%m-%d")
                    - datetime.strptime(entry["start"], "%Y-%m-%d")
                ).days
            except (KeyError, ValueError):
                continue
            if not 340 <= span <= 400:
                continue
        candidates.append(entry)
    if not candidates:
        return None
    key = (
        (lambda item: item.get("filed", ""))
        if kind == "flow"
        else (lambda item: (item.get("end", ""), item.get("filed", "")))
    )
    return max(candidates, key=key)


def _tag_value(facts: dict[str, Any], tag: str, unit: str, year: int, kind: str) -> float | None:
    entries = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, [])
    hit = _pick(entries, year, kind)
    return float(hit["val"]) if hit is not None else None


def _ifrs_tag_value(facts: dict[str, Any], tag: str, year: int, kind: str) -> float | None:
    units = facts.get("facts", {}).get("ifrs-full", {}).get(tag, {}).get("units", {})
    candidates: list[dict[str, Any]] = []
    for unit, entries in units.items():
        if unit in {"shares", "pure", "USD/shares"}:
            continue
        for entry in entries:
            if entry.get("fp") != "FY" or _fiscal_year(entry) != year:
                continue
            if str(entry.get("form", "")) not in {"20-F", "40-F", "6-K"}:
                continue
            if kind == "flow":
                try:
                    span = (
                        datetime.strptime(entry["end"], "%Y-%m-%d")
                        - datetime.strptime(entry["start"], "%Y-%m-%d")
                    ).days
                except (KeyError, ValueError):
                    continue
                if not 340 <= span <= 400:
                    continue
            candidates.append(entry)
    if not candidates:
        return None
    hit = max(
        candidates,
        key=lambda entry: (
            entry.get("end", ""),
            entry.get("filed", ""),
        ),
    )
    return float(hit["val"])


def extract_sec_kpis(facts: dict[str, Any], year: int) -> dict[str, dict[str, Any]]:
    """Apply the LEDGER-aligned tag waterfall to one Company Facts payload."""
    values: dict[str, dict[str, Any]] = {}
    for kpi, (kind, unit, tags, fallbacks) in _DEFS.items():
        for tag in tags:
            value = _tag_value(facts, tag, unit, year, kind)
            if value is not None:
                values[kpi] = {"value": value, "concept": tag}
                break
        if kpi in values:
            continue
        for components in fallbacks:
            parts = []
            for component in components:
                sign, tag = (-1, component[1:]) if component.startswith("-") else (1, component)
                value = _tag_value(facts, tag, unit, year, kind)
                if value is None:
                    parts = []
                    break
                parts.append((sign, tag, value))
            if parts:
                values[kpi] = {
                    "value": sum(sign * value for sign, _tag, value in parts),
                    "concept": "sum:"
                    + "".join(
                        ("-" if sign < 0 else ("+" if index else "")) + tag
                        for index, (sign, tag, _value) in enumerate(parts)
                    ),
                }
                break
    for kpi, (kind, tags) in _IFRS_DEFS.items():
        if kpi in values:
            continue
        for tag in tags:
            value = _ifrs_tag_value(facts, tag, year, kind)
            if value is not None:
                values[kpi] = {
                    "value": value,
                    "concept": f"ifrs-full:{tag}",
                }
                break
    return values


def resolve_sec_kpis(ticker: str, year: int, report_text: str = "") -> dict[str, Any]:
    """Return authoritative XBRL values, or a compact unavailable result."""
    try:
        ticker_data = _cached_json(_CACHE_ROOT / "tickers.json", _TICKERS_URL)
        mapping = {
            str(item["ticker"]).upper(): f"{int(item['cik_str']):010d}"
            for item in ticker_data.values()
        }
        current_cik = mapping.get(ticker.upper())
        try:
            historical_ciks = _historical_ciks(ticker, year, report_text)
        except (HTTPError, URLError, IncompleteRead, OSError, ValueError):
            historical_ciks = []
        cik_candidates = _ordered_cik_candidates(
            explicit_cik=_report_cik(report_text),
            current_cik=current_cik,
            historical_ciks=historical_ciks,
        )
        if not cik_candidates:
            return {"status": "unavailable", "reason": "ticker_not_in_sec"}
        report_entity = _registrant_name(report_text)
        best: tuple[int, str, dict[str, Any], dict[str, dict[str, Any]]] | None = None
        for cik in cik_candidates:
            try:
                facts = _cached_json(
                    _CACHE_ROOT / "companyfacts" / f"CIK{cik}.json",
                    _FACTS_URL.format(cik=cik),
                )
            except (HTTPError, URLError, IncompleteRead, OSError, ValueError):
                continue
            values = extract_sec_kpis(facts, year)
            score = _company_identity_score(
                ticker=ticker,
                report_entity=report_entity,
                facts_entity=str(facts.get("entityName", "")),
            )
            if values and (best is None or score > best[0]):
                best = (score, cik, facts, values)
            if values and score >= 100:
                break
        if best is None:
            return {"status": "unavailable", "reason": "companyfacts_unavailable"}
        identity_score, cik, facts, values = best
    except (
        HTTPError,
        URLError,
        IncompleteRead,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {"status": "unavailable", "reason": type(error).__name__}

    return {
        "status": "success",
        "source": _FACTS_URL.format(cik=cik),
        "cik": cik,
        "identity_score": identity_score,
        "fiscal_year": year,
        "values": values,
    }
