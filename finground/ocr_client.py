"""PDF rendering and OCR through an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass

import fitz
import httpx

DEFAULT_OCR_BASE_URL = "http://60.171.65.125:30691"
DEFAULT_OCR_MODEL = "Unlimited-OCR"
# Unlimited-OCR's chat template requires this task token; free-form transcription
# prompts can produce an immediate empty response.
DEFAULT_OCR_PROMPT = "<image>document parsing."
MULTI_PAGE_OCR_PROMPT = "<image>Multi page parsing."
OCR_PAGE_SEPARATOR = "<PAGE>"


class OcrError(RuntimeError):
    """Raised when a PDF cannot be rendered or OCR fails."""


@dataclass(frozen=True)
class OcrConfig:
    base_url: str = DEFAULT_OCR_BASE_URL
    model: str = DEFAULT_OCR_MODEL
    api_key: str = ""
    timeout_seconds: float = 180.0
    max_pages: int = 500
    batch_pages: int = 20
    max_parallel_pages: int = 4
    render_dpi: int = 144

    @classmethod
    def from_env(cls) -> OcrConfig:
        return cls(
            base_url=os.getenv("FINGROUND_OCR_BASE_URL", DEFAULT_OCR_BASE_URL).rstrip("/"),
            model=os.getenv("FINGROUND_OCR_MODEL", DEFAULT_OCR_MODEL),
            api_key=os.getenv("FINGROUND_OCR_API_KEY", ""),
            timeout_seconds=float(os.getenv("FINGROUND_OCR_TIMEOUT_SECONDS", "180")),
            max_pages=int(os.getenv("FINGROUND_OCR_MAX_PAGES", "500")),
            batch_pages=int(os.getenv("FINGROUND_OCR_BATCH_PAGES", "20")),
            max_parallel_pages=int(os.getenv("FINGROUND_OCR_MAX_PARALLEL_PAGES", "4")),
            render_dpi=int(os.getenv("FINGROUND_OCR_RENDER_DPI", "144")),
        )


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

    async def pdf_to_markdown(self, pdf_data: bytes) -> tuple[str, int]:
        try:
            document = fitz.open(stream=pdf_data, filetype="pdf")
        except Exception as exc:
            raise OcrError(f"uploaded file is not a readable PDF: {exc}") from exc
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
            images = [
                document.load_page(index).get_pixmap(matrix=matrix, alpha=False).tobytes("png")
                for index in range(page_count)
            ]
        finally:
            document.close()

        semaphore = asyncio.Semaphore(self.config.max_parallel_pages)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            batches = await asyncio.gather(
                *(
                    self._ocr_batch_with_fallback(
                        client,
                        images[start : start + self.config.batch_pages],
                        start + 1,
                        semaphore,
                    )
                    for start in range(0, page_count, self.config.batch_pages)
                )
            )
        pages = [page for batch in batches for page in batch]
        return "\n\n<--- Page Split --->\n\n".join(pages), page_count

    async def _ocr_batch_with_fallback(
        self,
        client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> list[str]:
        if len(images) == 1:
            return [await self._ocr_page(client, images[0], first_page_number, semaphore)]
        try:
            return await self._ocr_batch(client, images, first_page_number, semaphore)
        except OcrError:
            return list(
                await asyncio.gather(
                    *(
                        self._ocr_page(client, image, first_page_number + index, semaphore)
                        for index, image in enumerate(images)
                    )
                )
            )

    async def _ocr_batch(
        self,
        client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> list[str]:
        content: list[dict] = [{"type": "text", "text": MULTI_PAGE_OCR_PROMPT}]
        content.extend(self._image_part(image) for image in images)
        text = await self._request_ocr(
            client,
            content,
            first_page_number,
            semaphore,
            window_size=1024,
        )
        pages = [page.strip() for page in text.split(OCR_PAGE_SEPARATOR) if page.strip()]
        if len(pages) != len(images):
            last_page = first_page_number + len(images) - 1
            raise OcrError(
                f"OCR batch pages {first_page_number}-{last_page} returned "
                f"{len(pages)} page sections"
            )
        return pages

    async def _ocr_page(
        self,
        client: httpx.AsyncClient,
        image: bytes,
        page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        return await self._request_ocr(
            client,
            [{"type": "text", "text": DEFAULT_OCR_PROMPT}, self._image_part(image)],
            page_number,
            semaphore,
            window_size=128,
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
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 24_576 if window_size == 1024 else 8_192,
            "skip_special_tokens": False,
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
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise OcrError(f"OCR failed starting at PDF page {first_page_number}: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise OcrError(f"OCR returned empty content for PDF page {first_page_number}")
        if finish_reason == "length":
            raise OcrError(f"OCR output was truncated starting at PDF page {first_page_number}")
        return text.strip()
