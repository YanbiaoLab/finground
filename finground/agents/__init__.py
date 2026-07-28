"""Public agent factories and shared configuration."""

from finground.agents.common import ADK_MODEL, SETTINGS, create_adk_model
from finground.agents.multi_kpi import (
    KPI_CATALOGUE,
    MULTI_KPI_APP_NAME,
    MULTI_KPI_CONTEXT_WINDOW_TOKENS,
    MULTI_KPI_FINAL_WARNING_CALL,
    MULTI_KPI_INSTRUCTION,
    MULTI_KPI_LLM_CALL_LIMIT,
    MULTI_KPI_MAX_OUTPUT_TOKENS,
    MULTI_KPI_PROGRESS_REMINDER_CALL,
    MULTI_KPI_PROMPT_VERSION,
    MULTI_KPI_SEARCH_LIMIT,
    MULTI_KPI_SUBMISSION_DEADLINE,
    create_multi_kpi_agent,
    create_multi_kpi_app,
)
from finground.agents.needle import NEEDLE_INSTRUCTION, create_needle_agent

__all__ = [
    "ADK_MODEL",
    "KPI_CATALOGUE",
    "MULTI_KPI_APP_NAME",
    "MULTI_KPI_CONTEXT_WINDOW_TOKENS",
    "MULTI_KPI_FINAL_WARNING_CALL",
    "MULTI_KPI_INSTRUCTION",
    "MULTI_KPI_LLM_CALL_LIMIT",
    "MULTI_KPI_MAX_OUTPUT_TOKENS",
    "MULTI_KPI_PROGRESS_REMINDER_CALL",
    "MULTI_KPI_PROMPT_VERSION",
    "MULTI_KPI_SEARCH_LIMIT",
    "MULTI_KPI_SUBMISSION_DEADLINE",
    "NEEDLE_INSTRUCTION",
    "SETTINGS",
    "create_adk_model",
    "create_multi_kpi_agent",
    "create_multi_kpi_app",
    "create_needle_agent",
]
