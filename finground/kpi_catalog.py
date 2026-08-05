"""Read-only knowledge used by the KPI worker."""

from __future__ import annotations

from google.adk.tools import ToolContext
from pydantic import BaseModel, ConfigDict

from finground.named_function_tool import NamedFunctionTool

KNOWLEDGE_STATE_KEY = "temp:kpi_knowledge_key"


class KpiKnowledge(BaseModel):
    """Rules for finding and interpreting one canonical KPI."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    definition: str
    accepted_labels: tuple[str, ...]
    rejected_labels: tuple[str, ...]
    preferred_statements: tuple[str, ...]
    retrieval_hints: tuple[str, ...]
    normalization_rule: str
    cautions: tuple[str, ...]


_INCOME = ("Consolidated Statements of Operations", "Consolidated Income Statement")
_BALANCE = ("Consolidated Balance Sheets", "Consolidated Statement of Financial Position")
_CASH_FLOW = ("Consolidated Statements of Cash Flows",)
_COMMON_REJECTIONS = ("segment value", "quarterly value", "adjusted non-GAAP value")
_COMMON_CAUTIONS = (
    "Use the consolidated target-year value.",
    "Reject prior-year, quarterly, segment, and non-GAAP values.",
)


def _knowledge(
    key: str,
    name: str,
    definition: str,
    labels: tuple[str, ...],
    statements: tuple[str, ...],
    normalization: str,
    *,
    rejected: tuple[str, ...] = (),
    cautions: tuple[str, ...] = (),
) -> KpiKnowledge:
    return KpiKnowledge(
        key=key,
        name=name,
        definition=definition,
        accepted_labels=labels,
        rejected_labels=(*_COMMON_REJECTIONS, *rejected),
        preferred_statements=statements,
        retrieval_hints=(
            f"Search {', '.join(statements)} first.",
            f"Try the exact labels: {', '.join(labels)}.",
            "Read the target-year column and the unit governing the selected row.",
        ),
        normalization_rule=normalization,
        cautions=(*_COMMON_CAUTIONS, *cautions),
    )


_MONEY = "Apply only the monetary scale governing the selected row; preserve the reported sign."
_POSITIVE_COST = "Apply the governing monetary scale and return the expense magnitude as positive."
_POSITIVE_OUTFLOW = "Apply the governing monetary scale and return the cash outflow as positive."

KPI_CATALOG: dict[str, KpiKnowledge] = {
    item.key: item
    for item in (
        _knowledge(
            "revenue",
            "Revenue",
            "Consolidated operating revenue or net sales.",
            ("Revenue", "Revenues", "Net sales"),
            _INCOME,
            _MONEY,
            rejected=("interest income alone",),
            cautions=(
                "Do not substitute a component or segment total for the consolidated top line.",
            ),
        ),
        _knowledge(
            "cost_of_revenue",
            "Cost of revenue",
            "Cost of goods and services sold.",
            ("Cost of revenue", "Cost of sales", "Cost of goods sold"),
            _INCOME,
            _POSITIVE_COST,
        ),
        _knowledge(
            "gross_profit",
            "Gross profit",
            "Revenue less cost of revenue as printed by the company.",
            ("Gross profit", "Gross margin"),
            _INCOME,
            _MONEY,
            cautions=("Do not calculate gross profit when no printed total is available.",),
        ),
        _knowledge(
            "rd_expense",
            "Research and development expense",
            "Research and development expense only.",
            ("Research and development", "R&D expense"),
            _INCOME,
            _POSITIVE_COST,
            rejected=("research and development plus SG&A",),
        ),
        _knowledge(
            "sga_expense",
            "Selling, general and administrative expense",
            "Combined selling, general and administrative expense.",
            ("Selling, general and administrative", "SG&A", "General and administrative"),
            _INCOME,
            _POSITIVE_COST,
        ),
        _knowledge(
            "operating_income",
            "Operating income",
            "Operating income or operating profit.",
            ("Operating income", "Income from operations", "Operating profit"),
            _INCOME,
            _MONEY,
        ),
        _knowledge(
            "interest_expense",
            "Interest expense",
            "Interest expense on debt.",
            ("Interest expense", "Finance costs", "Finance expense"),
            _INCOME,
            _POSITIVE_COST,
            rejected=("interest income", "net interest income"),
        ),
        _knowledge(
            "income_tax_expense",
            "Income tax expense",
            "Income tax expense or benefit.",
            ("Income tax expense", "Provision for income taxes", "Income tax benefit"),
            _INCOME,
            _MONEY,
        ),
        _knowledge(
            "net_income",
            "Net income",
            "Net income attributable to the parent or common shareholders.",
            (
                "Net income attributable to common shareholders",
                "Net income attributable to parent",
                "Net income",
            ),
            _INCOME,
            _MONEY,
            rejected=("net income including non-controlling interests",),
            cautions=("Prefer the amount attributable to the parent and exclude NCI.",),
        ),
        _knowledge(
            "eps_basic",
            "Basic EPS",
            "Basic earnings per common share.",
            ("Basic earnings per share", "Basic EPS"),
            _INCOME,
            "Return the printed per-share amount without applying monetary statement scaling.",
            rejected=("diluted EPS",),
        ),
        _knowledge(
            "eps_diluted",
            "Diluted EPS",
            "Diluted earnings per common share.",
            ("Diluted earnings per share", "Diluted EPS"),
            _INCOME,
            "Return the printed per-share amount without applying monetary statement scaling.",
            rejected=("basic EPS",),
        ),
        _knowledge(
            "total_assets",
            "Total assets",
            "Total assets at fiscal year end.",
            ("Total assets",),
            _BALANCE,
            _MONEY,
        ),
        _knowledge(
            "total_liabilities",
            "Total liabilities",
            "Total liabilities at fiscal year end.",
            ("Total liabilities",),
            _BALANCE,
            _MONEY,
            rejected=("total liabilities and equity",),
        ),
        _knowledge(
            "stockholders_equity",
            "Stockholders' equity",
            "Equity attributable to the parent, excluding NCI.",
            ("Total stockholders' equity", "Shareholders' equity attributable to owners"),
            _BALANCE,
            _MONEY,
            rejected=("total equity including NCI",),
            cautions=("Exclude non-controlling interest.",),
        ),
        _knowledge(
            "stockholders_equity_incl_nci",
            "Total equity including NCI",
            "Total equity including non-controlling interests.",
            ("Total equity", "Stockholders' equity and non-controlling interests"),
            _BALANCE,
            _MONEY,
            cautions=("Require a total that includes NCI.",),
        ),
        _knowledge(
            "cash_and_equivalents",
            "Cash and cash equivalents",
            "Unrestricted cash and cash equivalents at period end.",
            ("Cash and cash equivalents",),
            _BALANCE,
            _MONEY,
            rejected=("cash including restricted cash",),
            cautions=("Exclude restricted cash.",),
        ),
        _knowledge(
            "cash_incl_restricted",
            "Cash including restricted cash",
            "Cash, cash equivalents, and restricted cash combined.",
            ("Cash, cash equivalents and restricted cash",),
            _CASH_FLOW,
            _MONEY,
            rejected=("cash and cash equivalents alone",),
        ),
        _knowledge(
            "long_term_debt_total",
            "Total long-term debt",
            "Printed total long-term debt excluding a separately reported current portion.",
            ("Total long-term debt", "Long-term debt"),
            _BALANCE,
            _MONEY,
            cautions=("Do not double count a separately printed current portion.",),
        ),
        _knowledge(
            "long_term_debt_noncurrent",
            "Noncurrent long-term debt",
            "Long-term debt classified as noncurrent.",
            ("Long-term debt, noncurrent", "Long-term debt less current portion"),
            _BALANCE,
            _MONEY,
            rejected=("current portion of long-term debt",),
        ),
        _knowledge(
            "long_term_debt_current",
            "Current portion of long-term debt",
            "Current maturities or current portion of long-term debt.",
            ("Current portion of long-term debt", "Current maturities of long-term debt"),
            _BALANCE,
            _MONEY,
            rejected=("all short-term borrowings",),
        ),
        _knowledge(
            "short_term_borrowings",
            "Short-term borrowings",
            "Borrowings with original maturity of one year or less.",
            ("Short-term borrowings", "Short-term debt", "Commercial paper"),
            _BALANCE,
            _MONEY,
            rejected=("current portion of long-term debt alone",),
        ),
        _knowledge(
            "inventory",
            "Inventory",
            "Inventory net of allowances at period end.",
            ("Inventories", "Inventory, net"),
            _BALANCE,
            _MONEY,
        ),
        _knowledge(
            "accounts_receivable",
            "Accounts receivable",
            "Current trade accounts receivable net of allowance.",
            ("Accounts receivable, net", "Trade receivables"),
            _BALANCE,
            _MONEY,
            rejected=("noncurrent receivables",),
        ),
        _knowledge(
            "accounts_payable",
            "Accounts payable",
            "Current trade accounts payable.",
            ("Accounts payable", "Trade payables"),
            _BALANCE,
            _MONEY,
            rejected=("accounts payable and unrelated accrued liabilities",),
        ),
        _knowledge(
            "shares_outstanding",
            "Shares outstanding",
            "Common shares outstanding at fiscal year end.",
            ("Common shares outstanding", "Shares outstanding"),
            _BALANCE,
            "Apply only a clearly governing share-count scale and return the raw share count.",
            rejected=("weighted-average shares",),
        ),
        _knowledge(
            "operating_cash_flow",
            "Operating cash flow",
            "Net cash provided by or used in operating activities.",
            ("Net cash provided by operating activities", "Net cash used in operating activities"),
            _CASH_FLOW,
            _MONEY,
        ),
        _knowledge(
            "investing_cash_flow",
            "Investing cash flow",
            "Net cash provided by or used in investing activities.",
            ("Net cash provided by investing activities", "Net cash used in investing activities"),
            _CASH_FLOW,
            _MONEY,
        ),
        _knowledge(
            "financing_cash_flow",
            "Financing cash flow",
            "Net cash provided by or used in financing activities.",
            ("Net cash provided by financing activities", "Net cash used in financing activities"),
            _CASH_FLOW,
            _MONEY,
        ),
        _knowledge(
            "capex",
            "Capital expenditure",
            "Cash paid to acquire property, plant, and equipment.",
            ("Purchases of property, plant and equipment", "Capital expenditures"),
            _CASH_FLOW,
            _POSITIVE_OUTFLOW,
            rejected=("total investing cash flow",),
        ),
        _knowledge(
            "depreciation_amortization",
            "Depreciation and amortization",
            "Depreciation, depletion, and amortization addback.",
            (
                "Depreciation and amortization",
                "Depreciation, depletion and amortization",
                "Depreciation",
            ),
            _CASH_FLOW,
            _POSITIVE_COST,
        ),
        _knowledge(
            "dividends_paid",
            "Dividends paid",
            "Cash dividends paid to common shareholders.",
            ("Dividends paid", "Payments of dividends"),
            _CASH_FLOW,
            _POSITIVE_OUTFLOW,
            rejected=("dividends declared",),
        ),
    )
}


def get_kpi_knowledge(kpi_key: str, tool_context: ToolContext) -> dict:
    """Return the canonical extraction rules for one exact KPI key."""
    knowledge = KPI_CATALOG.get(kpi_key)
    if knowledge is None:
        return {
            "status": "error",
            "error": f"unknown KPI key: {kpi_key}",
            "supported_kpis": list(KPI_CATALOG),
        }
    tool_context.state[KNOWLEDGE_STATE_KEY] = kpi_key
    return {"status": "success", "knowledge": knowledge.model_dump(mode="json")}


get_kpi_knowledge_tool = NamedFunctionTool(get_kpi_knowledge, name="GetKpiKnowledge")
