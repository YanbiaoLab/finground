"""Ingest annual-report attachments from ADK chat into session artifacts."""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from pathlib import PurePath
from typing import Any, Protocol

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from finground.ocr_client import (
    OcrConfig,
    PdfExtractionError,
    VllmPdfOcrClient,
)
from finground.report_tools import REPORT_STATE_KEY

_PENDING_ARTIFACTS_KEY = "temp:report_upload_artifacts"
_PENDING_MANIFEST_KEY = "temp:report_upload_manifest"
_PAGE_BREAK = re.compile(r"<---\s*Page Split\s*--->", re.IGNORECASE)
_OCR_CACHE_ROOT = ".finground/ocr"
_OCR_CACHE_GLOBAL_USER_ID = "finground-ocr-cache"
_OCR_CACHE_GLOBAL_SESSION_ID = "global"

logger = logging.getLogger(__name__)


class PdfOcrClient(Protocol):
    """The OCR behavior required by the report upload plugin."""

    def cache_fingerprint(self) -> str: ...

    async def pdf_to_markdown(self, pdf_data: bytes) -> tuple[str, int]: ...


def _is_markdown(blob: types.Blob) -> bool:
    name = blob.display_name or ""
    return blob.mime_type in {"text/markdown", "text/x-markdown"} or name.lower().endswith(
        (".md", ".markdown")
    )


def _is_pdf(blob: types.Blob) -> bool:
    name = blob.display_name or ""
    return blob.mime_type == "application/pdf" or name.lower().endswith(".pdf")


def _safe_filename(blob: types.Blob, invocation_id: str, index: int) -> str:
    display_name = PurePath(blob.display_name or "").name
    return display_name or f"report_{invocation_id}_{index}.md"


def _report_ref(filename: str) -> str:
    stem = PurePath(filename).stem
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return normalized or "report"


def _ocr_cache_filename(pdf_data: bytes, fingerprint: str) -> str:
    pdf_digest = hashlib.sha256(pdf_data).hexdigest()
    settings_digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return f"{_OCR_CACHE_ROOT}/{settings_digest}/{pdf_digest}.md"


def _cached_markdown(artifact: types.Part | None) -> str | None:
    if artifact is None:
        return None
    if artifact.text:
        return artifact.text
    blob = artifact.inline_data
    if blob is None or blob.mime_type not in {"text/markdown", "text/x-markdown"}:
        return None
    try:
        text = blob.data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text or None


class ReportUploadPlugin(BasePlugin):
    """Save uploaded Markdown reports and make the latest one current."""

    def __init__(self, ocr_client: PdfOcrClient | None = None) -> None:
        super().__init__(name="report_upload")
        if ocr_client is not None:
            self._ocr_client = ocr_client
        else:
            config = OcrConfig.from_env_if_configured()
            self._ocr_client = VllmPdfOcrClient(config) if config is not None else None

    async def _load_pdf_cache(
        self,
        invocation_context: InvocationContext,
        filename: str,
    ) -> str | None:
        try:
            artifact = await invocation_context.artifact_service.load_artifact(
                app_name=invocation_context.app_name,
                user_id=_OCR_CACHE_GLOBAL_USER_ID,
                session_id=_OCR_CACHE_GLOBAL_SESSION_ID,
                filename=filename,
            )
        except Exception:  # noqa: BLE001 - cache failure must not block OCR
            logger.warning("Unable to read PDF OCR cache", exc_info=True)
            return None
        return _cached_markdown(artifact)

    async def _save_pdf_cache(
        self,
        invocation_context: InvocationContext,
        filename: str,
        text: str,
    ) -> None:
        artifact = types.Part(
            inline_data=types.Blob(
                data=text.encode("utf-8"),
                mime_type="text/markdown",
            )
        )
        try:
            await invocation_context.artifact_service.save_artifact(
                app_name=invocation_context.app_name,
                user_id=_OCR_CACHE_GLOBAL_USER_ID,
                session_id=_OCR_CACHE_GLOBAL_SESSION_ID,
                filename=filename,
                artifact=artifact,
            )
        except Exception:  # noqa: BLE001 - cache failure must not discard OCR output
            logger.warning("Unable to write PDF OCR cache", exc_info=True)

    async def _pdf_to_markdown(
        self,
        invocation_context: InvocationContext,
        pdf_data: bytes,
    ) -> tuple[str, int]:
        artifact_service = invocation_context.artifact_service
        if artifact_service is None:
            raise RuntimeError("an artifact service is required for report uploads")
        if self._ocr_client is None:
            raise PdfExtractionError(
                "PDF OCR is not configured; set FINGROUND_OCR_BASE_URL and "
                "FINGROUND_OCR_MODEL"
            )
        ocr_cache_filename = _ocr_cache_filename(
            pdf_data,
            self._ocr_client.cache_fingerprint(),
        )
        cached_text = await self._load_pdf_cache(
            invocation_context,
            ocr_cache_filename,
        )
        if cached_text is not None:
            return cached_text, len(_PAGE_BREAK.split(cached_text))
        text, total_pages = await self._ocr_client.pdf_to_markdown(pdf_data)
        await self._save_pdf_cache(
            invocation_context,
            ocr_cache_filename,
            text,
        )
        return text, total_pages

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        if not user_message.parts:
            return None
        report_parts = [
            part
            for part in user_message.parts
            if part.inline_data is not None
            and (_is_markdown(part.inline_data) or _is_pdf(part.inline_data))
        ]
        if not report_parts:
            return None
        if invocation_context.artifact_service is None:
            raise RuntimeError("an artifact service is required for report uploads")

        new_parts: list[types.Part] = []
        pending_artifacts: dict[str, int] = {}
        latest_manifest: dict[str, Any] | None = None
        for index, part in enumerate(user_message.parts):
            blob = part.inline_data
            if blob is None or not (_is_markdown(blob) or _is_pdf(blob)):
                new_parts.append(part)
                continue
            filename = _safe_filename(blob, invocation_context.invocation_id, index)
            if _is_pdf(blob):
                text, total_pages = await self._pdf_to_markdown(
                    invocation_context,
                    blob.data,
                )
                filename = f"{PurePath(filename).stem}.ocr.md"
                artifact_part = types.Part(
                    inline_data=types.Blob(
                        data=text.encode("utf-8"),
                        mime_type="text/markdown",
                        display_name=filename,
                    )
                )
            else:
                try:
                    text = blob.data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("uploaded Markdown report must be UTF-8") from exc
                total_pages = len(_PAGE_BREAK.split(text))
                artifact_part = copy.copy(part)
            report_ref = _report_ref(filename)
            version = await invocation_context.artifact_service.save_artifact(
                app_name=invocation_context.app_name,
                user_id=invocation_context.user_id,
                session_id=invocation_context.session.id,
                filename=filename,
                artifact=artifact_part,
            )
            pending_artifacts[filename] = version
            latest_manifest = {
                "report_ref": report_ref,
                "artifact_name": filename,
                "mime_type": "text/markdown",
                "total_pages": total_pages,
                "total_chars": len(text),
            }
            if _is_pdf(blob):
                latest_manifest["source_mime_type"] = "application/pdf"
                latest_manifest["extraction_method"] = "ocr"
            new_parts.append(
                types.Part.from_text(
                    text=(
                        f'[Uploaded annual report artifact: "{filename}"; '
                        f'use report_ref "{report_ref}" for KPI tasks.]'
                    )
                )
            )

        state = invocation_context.session.state
        state[_PENDING_ARTIFACTS_KEY] = pending_artifacts
        state[_PENDING_MANIFEST_KEY] = latest_manifest
        state[REPORT_STATE_KEY] = latest_manifest
        return types.Content(role=user_message.role, parts=new_parts)

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        del agent
        pending_artifacts = callback_context.state.get(_PENDING_ARTIFACTS_KEY, {})
        if pending_artifacts:
            callback_context.actions.artifact_delta.update(pending_artifacts)
            callback_context.state[_PENDING_ARTIFACTS_KEY] = {}
        manifest = callback_context.state.get(_PENDING_MANIFEST_KEY)
        if manifest:
            callback_context.state[REPORT_STATE_KEY] = manifest
            callback_context.state[_PENDING_MANIFEST_KEY] = None
