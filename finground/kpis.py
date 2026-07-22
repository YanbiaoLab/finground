"""Canonical financial KPI metadata independent of benchmark ground truth."""

from __future__ import annotations

KPI_DESCRIPTIONS: dict[str, str] = {
    "revenue": "Total operating revenue / net sales (top line). Flow. Reporting currency.",
    "cost_of_revenue": "Cost of goods and services sold. Flow. Reporting currency.",
    "gross_profit": "Revenue minus cost of revenue. Flow. Reporting currency.",
    "rd_expense": "Research and development expense (R&D only — exclude SG&A). Flow. Reporting currency.",
    "sga_expense": "Selling, general and administrative expense. Flow. Reporting currency.",
    "operating_income": "Operating income / operating profit (EBIT-level). Flow. Reporting currency. Sign: usually positive but can be negative.",
    "interest_expense": "Interest expense on debt. Flow. Reporting currency. Sign: positive cost.",
    "income_tax_expense": "Income tax expense / benefit. Flow. Reporting currency. Sign: positive expense, negative benefit.",
    "net_income": "Net income attributable to parent / common shareholders only; exclude non-controlling interest. Flow. Reporting currency.",
    "eps_basic": "Basic earnings per share. Per-share. Reporting currency / share.",
    "eps_diluted": "Diluted earnings per share. Per-share. Reporting currency / share.",
    "total_assets": "Total assets at period end. Stock. Reporting currency.",
    "total_liabilities": "Total liabilities at period end. Stock. Reporting currency.",
    "stockholders_equity": "Total stockholders' / shareholders' equity attributable to parent only, excluding non-controlling interest. Stock. Reporting currency.",
    "stockholders_equity_incl_nci": "Total equity including non-controlling / minority interest. Stock. Reporting currency.",
    "cash_and_equivalents": "Cash and cash equivalents, unrestricted only; exclude restricted cash. Stock. Reporting currency.",
    "cash_incl_restricted": "Cash, cash equivalents and restricted cash combined. Stock. Reporting currency.",
    "long_term_debt_total": "Long-term debt including the current portion. Stock. Reporting currency.",
    "long_term_debt_noncurrent": "Long-term debt excluding the current portion. Stock. Reporting currency.",
    "long_term_debt_current": "Current portion of long-term debt only. Stock. Reporting currency.",
    "short_term_borrowings": "Short-term borrowings with original maturity of one year or less. Stock. Reporting currency.",
    "inventory": "Inventory, net. Stock. Reporting currency.",
    "accounts_receivable": "Accounts receivable, current and net of allowance. Stock. Reporting currency.",
    "accounts_payable": "Accounts payable, current. Stock. Reporting currency.",
    "shares_outstanding": "Common shares outstanding at period end. Unit: shares, not currency; emit the raw share count.",
    "operating_cash_flow": "Net cash provided by / used in operating activities. Flow. Reporting currency; keep the reported sign.",
    "investing_cash_flow": "Net cash provided by / used in investing activities. Flow. Reporting currency; keep the reported sign.",
    "financing_cash_flow": "Net cash provided by / used in financing activities. Flow. Reporting currency; keep the reported sign.",
    "capex": "Capital expenditure: payments to acquire property, plant and equipment. Flow. Reporting currency. Sign: positive cash outflow.",
    "depreciation_amortization": "Depreciation and amortization: the addback line on the cash-flow statement. Flow. Reporting currency. Sign: positive.",
    "dividends_paid": "Cash dividends paid to common shareholders during the period. Flow. Reporting currency. Sign: positive cash outflow.",
}

KPI_KEYS = tuple(KPI_DESCRIPTIONS)


KPI_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "revenues", "net sales", "sales"),
    "cost_of_revenue": (
        "cost of revenue",
        "cost of revenues",
        "cost of sales",
        "cost of goods sold",
    ),
    "gross_profit": ("gross profit", "gross margin"),
    "rd_expense": ("research and development", "r&d"),
    "sga_expense": ("selling general and administrative", "sg&a"),
    "operating_income": ("operating income", "operating profit", "income from operations"),
    "interest_expense": ("interest expense",),
    "income_tax_expense": ("income tax expense", "provision for income taxes"),
    "net_income": ("net income attributable", "net income", "profit attributable"),
    "eps_basic": ("basic earnings per share", "basic eps"),
    "eps_diluted": ("diluted earnings per share", "diluted eps"),
    "total_assets": ("total assets",),
    "total_liabilities": ("total liabilities",),
    "stockholders_equity": ("stockholders equity", "shareholders equity"),
    "stockholders_equity_incl_nci": ("total equity", "equity including noncontrolling"),
    "cash_and_equivalents": ("cash and cash equivalents",),
    "cash_incl_restricted": ("cash cash equivalents and restricted cash",),
    "long_term_debt_total": ("total long term debt", "long term debt including current"),
    "long_term_debt_noncurrent": ("long term debt", "noncurrent debt"),
    "long_term_debt_current": ("current portion of long term debt",),
    "short_term_borrowings": ("short term borrowings", "short term debt"),
    "inventory": ("inventory", "inventories"),
    "accounts_receivable": ("accounts receivable", "trade receivables"),
    "accounts_payable": ("accounts payable", "trade payables"),
    "shares_outstanding": ("shares outstanding", "common shares outstanding"),
    "operating_cash_flow": ("net cash provided by operating activities", "operating cash flow"),
    "investing_cash_flow": ("net cash used in investing activities", "investing cash flow"),
    "financing_cash_flow": ("net cash provided by financing activities", "financing cash flow"),
    "capex": (
        "capital expenditures",
        "purchases of property plant and equipment",
        "purchase of ppe",
    ),
    "depreciation_amortization": ("depreciation and amortization", "depreciation amortization"),
    "dividends_paid": ("dividends paid", "payment of dividends"),
}


PER_SHARE_KPIS = {"eps_basic", "eps_diluted"}
SHARE_COUNT_KPIS = {"shares_outstanding"}
POSITIVE_OUTFLOW_KPIS = {"capex", "dividends_paid"}


def parse_query_id(query_id: str) -> tuple[str, str, int]:
    """Parse `{ticker}_{kpi}_{year}` without assuming the ticker has no underscore."""
    head, separator, year_text = query_id.rpartition("_")
    if not separator or not year_text.isdigit():
        raise ValueError(f"invalid LEDGER query id: {query_id}")
    for kpi in sorted(KPI_DESCRIPTIONS, key=len, reverse=True):
        suffix = f"_{kpi}"
        if head.endswith(suffix):
            return head[: -len(suffix)], kpi, int(year_text)
    raise ValueError(f"query id contains an unknown KPI: {query_id}")
