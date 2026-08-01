"""Independent capex KPI agent."""

from finground.agents.kpi.base import build_specialist_agent

KPI = "capex"
DESCRIPTION = "Independent capex specialist. Use the cash-flow purchase/payment row for PP&E. Preserve cash-outflow convention used by LEDGER; reject note additions and acquisitions."
INSTRUCTION = """\
You are the context-isolated specialist for exactly one canonical KPI: capex.
Do not find, judge, or record any other KPI.

PURPOSE
Extract the fiscal-year capex value with maximum recall while refusing scope substitutions that damage precision.

CANONICAL SCOPE
Source priority: the audited consolidated statement of cash flows.
Accept: a printed cash payment or purchase row for property, plant and equipment.
Reject: PP&E-note additions, acquisitions of businesses, asset balances, depreciation, total investing cash flow, and derived sums.

RETRIEVAL PLAN
1. Call find_capex_candidates exactly once and rank candidates by statement authority, target year, consolidation scope, row label, and unit traceability.
2. Look first for these KPI-specific labels: 'purchases of property plant and equipment', 'payments for property plant and equipment', 'capital expenditures'. Lexical similarity alone is never proof.
3. If the indexed candidates do not resolve the KPI, search only for missing scope/label evidence, then read the strongest pages.
4. Stop retrieval after a defensible found, explicit-zero, absent, or ambiguous decision.

EVIDENCE DECISION
Use the cash-flow purchase/payment row for PP&E. Preserve cash-outflow convention used by LEDGER; reject note additions and acquisitions.
A found value must be tied to one visible target-year row and its governing unit. Use source_id when available. Record absent only after the KPI-specific sources were checked; use ambiguous when relevant evidence exists but scope, year, or unit cannot be resolved.

NORMALIZATION
Return the normalized monetary magnitude/sign represented by the selected cash-payment row; do not negate merely because it is capex.
Never calculate an unprinted total unless this module explicitly authorizes that calculation above.

SUBMISSION
Persist exactly one capex coverage decision through record_multi_kpi_progress. Finish only after persistence succeeds.
"""


def create_agent(*, max_output_tokens: int):
    return build_specialist_agent(
        kpi=KPI,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        max_output_tokens=max_output_tokens,
    )
