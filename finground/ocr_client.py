"""Render PDFs and extract their content through a vLLM OCR endpoint."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from dataclasses import dataclass, field

import fitz
import httpx

from finground.config import required_env, required_service_url

# Unlimited-OCR's chat template requires this task token; free-form transcription
# prompts can produce an immediate empty response.
DEFAULT_OCR_PROMPT = "<image>document parsing."
MULTI_PAGE_OCR_PROMPT = "<image>Multi page parsing."
OCR_PAGE_SEPARATOR = "<PAGE>"
MARKDOWN_PAGE_SEPARATOR = "\n\n<--- Page Split --->\n\n"


class PdfExtractionError(RuntimeError):
    """Raised when PDF extraction cannot produce usable text."""


class OcrError(PdfExtractionError):
    """Raised when a PDF cannot be rendered or OCR fails."""


class _OcrOutputTruncatedError(OcrError):
    """Raised when the OCR service exhausts its output token budget."""

    def __init__(self, message: str, partial_text: str) -> None:
        super().__init__(message)
        self.partial_text = partial_text


class _OcrBatchStructureError(OcrError):
    """Raised when multi-page OCR cannot be mapped safely back to its pages."""


class _OcrTransientRequestError(OcrError):
    """Base class for bounded retries of transient transport failures."""


class _OcrRequestTimeoutError(_OcrTransientRequestError):
    """Raised when the OCR endpoint does not respond within the configured timeout."""


class _OcrTransportError(_OcrTransientRequestError):
    """Raised when the OCR connection is interrupted in transit."""


@dataclass(frozen=True)
class OcrConfig:
    base_url: str = field(default_factory=lambda: required_env("FINGROUND_OCR_BASE_URL"))
    model: str = field(default_factory=lambda: required_env("FINGROUND_OCR_MODEL"))
    api_key: str = field(default_factory=lambda: os.getenv("FINGROUND_OCR_API_KEY", ""))
    timeout_seconds: float = 180.0
    max_pages: int = 500
    batch_pages: int = 20
    max_parallel_pages: int = 8
    render_dpi: int = 144
    cache_version: str = "v1"
    trust_env: bool = False

    @classmethod
    def from_env(cls) -> OcrConfig:
        return cls(
            base_url=required_service_url("FINGROUND_OCR_BASE_URL"),
            model=required_env("FINGROUND_OCR_MODEL"),
            api_key=os.getenv("FINGROUND_OCR_API_KEY", ""),
            timeout_seconds=float(os.getenv("FINGROUND_OCR_TIMEOUT_SECONDS", "180")),
            max_pages=int(os.getenv("FINGROUND_OCR_MAX_PAGES", "500")),
            batch_pages=int(os.getenv("FINGROUND_OCR_BATCH_PAGES", "20")),
            max_parallel_pages=int(os.getenv("FINGROUND_OCR_MAX_PARALLEL_PAGES", "8")),
            render_dpi=int(os.getenv("FINGROUND_OCR_RENDER_DPI", "144")),
            cache_version=os.getenv("FINGROUND_OCR_CACHE_VERSION", "v1"),
            trust_env=os.getenv("FINGROUND_OCR_TRUST_ENV", "false").casefold()
            in {"1", "true", "yes", "on"},
        )

    @classmethod
    def from_env_if_configured(cls) -> OcrConfig | None:
        base_url = os.getenv("FINGROUND_OCR_BASE_URL", "").strip()
        model = os.getenv("FINGROUND_OCR_MODEL", "").strip()
        if not base_url and not model:
            return None
        return cls.from_env()


class VllmPdfOcrClient:
    """Render PDF pages locally and OCR them concurrently through vLLM."""

    def __init__(self, config: OcrConfig | None = None) -> None:
        self.config = config or OcrConfig.from_env()
        if self.config.max_parallel_pages < 1:
            raise ValueError("max_parallel_pages must be at least 1")
        if self.config.max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if not 10 <= self.config.batch_pages <= 30:
            raise ValueError("batch_pages must be between 10 and 30")
        if self.config.render_dpi < 72:
            raise ValueError("render_dpi must be at least 72")
        if not self.config.cache_version.strip():
            raise ValueError("cache_version must not be empty")

    def cache_fingerprint(self) -> str:
        """Identify OCR settings that can change cached Markdown output."""
        settings = {
            "base_url": self.config.base_url,
            "batch_pages": self.config.batch_pages,
            "cache_version": self.config.cache_version,
            "max_pages": self.config.max_pages,
            "model": self.config.model,
            "multi_page_prompt": MULTI_PAGE_OCR_PROMPT,
            "page_separator": OCR_PAGE_SEPARATOR,
            "render_dpi": self.config.render_dpi,
            "single_page_prompt": DEFAULT_OCR_PROMPT,
        }
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def pdf_to_markdown(self, pdf_data: bytes) -> tuple[str, int]:
        try:
            document = fitz.open(stream=pdf_data, filetype="pdf")
        except Exception as exc:
            raise OcrError(f"uploaded file is not a readable PDF: {exc}") from exc
        tasks: list[asyncio.Task[list[str]]] = []
        try:
            page_count = document.page_count
            if page_count == 0:
                raise OcrError("uploaded PDF has no pages")
            if page_count > self.config.max_pages:
                raise OcrError(
                    f"PDF has {page_count} pages; configured limit is {self.config.max_pages}"
                )
            scale = self.config.render_dpi / 72
            matrix = fitz.Matrix(scale, scale)
            semaphore = asyncio.Semaphore(self.config.max_parallel_pages)
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            timeout = httpx.Timeout(self.config.timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                trust_env=self.config.trust_env,
            ) as client:
                for start in range(0, page_count, self.config.batch_pages):
                    end = min(start + self.config.batch_pages, page_count)
                    images = self._render_pages(document, start, end, matrix)
                    task = asyncio.create_task(
                        self._ocr_batch_with_fallback(
                            client,
                            images,
                            start + 1,
                            semaphore,
                        )
                    )
                    tasks.append(task)
                    # Rendering is synchronous. Yield between batches so OCR can
                    # start before the rest of a large document is rendered.
                    await asyncio.sleep(0)
                batches = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            document.close()
        pages = [page for batch in batches for page in batch]
        return MARKDOWN_PAGE_SEPARATOR.join(pages), page_count

    @staticmethod
    def _render_pages(
        document: fitz.Document,
        start: int,
        end: int,
        matrix: fitz.Matrix,
    ) -> list[bytes]:
        return [
            document.load_page(index).get_pixmap(matrix=matrix, alpha=False).tobytes("png")
            for index in range(start, end)
        ]

    async def _ocr_batch_with_fallback(
        self,
        client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
        *,
        transport_splits_remaining: int = 1,
    ) -> list[str]:
        if len(images) == 1:
            return [await self._ocr_page(client, images[0], first_page_number, semaphore)]
        try:
            return await self._ocr_batch(client, images, first_page_number, semaphore)
        except (_OcrBatchStructureError, _OcrRequestTimeoutError):
            return await self._split_batch(
                client,
                images,
                first_page_number,
                semaphore,
                transport_splits_remaining=transport_splits_remaining,
            )
        except _OcrTransportError:
            if transport_splits_remaining == 0:
                raise
            return await self._split_batch(
                client,
                images,
                first_page_number,
                semaphore,
                transport_splits_remaining=transport_splits_remaining - 1,
            )

    async def _split_batch(
        self,
        client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
        *,
        transport_splits_remaining: int,
    ) -> list[str]:
        midpoint = len(images) // 2
        left, right = await asyncio.gather(
            self._ocr_batch_with_fallback(
                client,
                images[:midpoint],
                first_page_number,
                semaphore,
                transport_splits_remaining=transport_splits_remaining,
            ),
            self._ocr_batch_with_fallback(
                client,
                images[midpoint:],
                first_page_number + midpoint,
                semaphore,
                transport_splits_remaining=transport_splits_remaining,
            ),
        )
        return [*left, *right]

    async def _ocr_batch(
        self,
        client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> list[str]:
        content: list[dict] = [{"type": "text", "text": MULTI_PAGE_OCR_PROMPT}]
        content.extend(self._image_part(image) for image in images)
        try:
            text = await self._request_batch(
                client,
                content,
                first_page_number,
                semaphore,
                window_size=1024,
                max_tokens=24_576,
                image_mode="base",
            )
        except _OcrOutputTruncatedError as exc:
            sections = self._split_pages(exc.partial_text)
            complete_count = min(max(len(sections) - 1, 0), len(images) - 1)
            if complete_count == 0:
                raise _OcrBatchStructureError(str(exc)) from exc
            tail = await self._ocr_batch_with_fallback(
                client,
                images[complete_count:],
                first_page_number + complete_count,
                semaphore,
            )
            return [*sections[:complete_count], *tail]
        pages = self._split_pages(text)
        if len(pages) != len(images):
            last_page = first_page_number + len(images) - 1
            raise _OcrBatchStructureError(
                f"OCR batch pages {first_page_number}-{last_page} returned "
                f"{len(pages)} page sections"
            )
        return pages

    async def _request_batch(
        self,
        client: httpx.AsyncClient,
        content: list[dict],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
        *,
        window_size: int,
        max_tokens: int,
        image_mode: str,
    ) -> str:
        try:
            return await self._request_ocr(
                client,
                content,
                first_page_number,
                semaphore,
                window_size=window_size,
                max_tokens=max_tokens,
                image_mode=image_mode,
            )
        except _OcrTransportError:
            await asyncio.sleep(0.25)
            return await self._request_ocr(
                client,
                content,
                first_page_number,
                semaphore,
                window_size=window_size,
                max_tokens=max_tokens,
                image_mode=image_mode,
            )

    @staticmethod
    def _split_pages(text: str) -> list[str]:
        return [page.strip() for page in text.split(OCR_PAGE_SEPARATOR) if page.strip()]

    async def _ocr_page(
        self,
        client: httpx.AsyncClient,
        image: bytes,
        page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        content = [{"type": "text", "text": DEFAULT_OCR_PROMPT}, self._image_part(image)]
        try:
            return await self._request_single_page(
                client,
                content,
                page_number,
                semaphore,
                window_size=128,
                max_tokens=8_192,
                image_mode="gundam",
            )
        except _OcrOutputTruncatedError:
            return await self._request_single_page(
                client,
                content,
                page_number,
                semaphore,
                window_size=128,
                max_tokens=24_576,
                image_mode="gundam",
            )

    async def _request_single_page(
        self,
        client: httpx.AsyncClient,
        content: list[dict],
        page_number: int,
        semaphore: asyncio.Semaphore,
        *,
        window_size: int,
        max_tokens: int,
        image_mode: str,
    ) -> str:
        try:
            return await self._request_ocr(
                client,
                content,
                page_number,
                semaphore,
                window_size=window_size,
                max_tokens=max_tokens,
                image_mode=image_mode,
            )
        except _OcrTransientRequestError:
            await asyncio.sleep(0.25)
            return await self._request_ocr(
                client,
                content,
                page_number,
                semaphore,
                window_size=window_size,
                max_tokens=max_tokens,
                image_mode=image_mode,
            )

    @staticmethod
    def _image_part(image: bytes) -> dict:
        encoded = base64.b64encode(image).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    async def _request_ocr(
        self,
        client: httpx.AsyncClient,
        content: list[dict],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
        *,
        window_size: int,
        max_tokens: int,
        image_mode: str,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "skip_special_tokens": False,
            "images_config": {"image_mode": image_mode},
            "vllm_xargs": {"ngram_size": 35, "window_size": window_size},
        }
        try:
            async with semaphore:
                response = await client.post(
                    f"{self.config.base_url}/v1/chat/completions", json=payload
                )
                response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            finish_reason = body["choices"][0].get("finish_reason")
        except httpx.TimeoutException as exc:
            raise _OcrRequestTimeoutError(
                f"OCR {type(exc).__name__} after {self.config.timeout_seconds:g}s "
                f"starting at PDF page {first_page_number}"
            ) from exc
        except httpx.TransportError as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise _OcrTransportError(
                f"OCR transport failure ({detail}) starting at PDF page {first_page_number}"
            ) from exc
        except httpx.HTTPError as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OcrError(
                f"OCR failed starting at PDF page {first_page_number}: {detail}"
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OcrError(
                f"OCR returned an invalid response starting at PDF page "
                f"{first_page_number}: {detail}"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise OcrError(f"OCR returned empty content for PDF page {first_page_number}")
        if finish_reason == "length":
            raise _OcrOutputTruncatedError(
                f"OCR output was truncated starting at PDF page {first_page_number}",
                text,
            )
        return text.strip()
