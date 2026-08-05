import asyncio
import json

import fitz
import httpx
import pytest

from finground.ocr_client import (
    OcrConfig,
    OcrError,
    VllmPdfOcrClient,
)


def _pdf_with_text_pages(*page_texts: str) -> bytes:
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def test_invalid_pdf_is_rejected_before_network_call() -> None:
    client = VllmPdfOcrClient(OcrConfig())

    with pytest.raises(OcrError, match="not a readable PDF"):
        asyncio.run(client.pdf_to_markdown(b"not a pdf"))


def test_ocr_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGROUND_OCR_BASE_URL", "http://ocr.example/")
    monkeypatch.setenv("FINGROUND_OCR_MODEL", "ocr-model")
    monkeypatch.setenv("FINGROUND_OCR_BATCH_PAGES", "15")
    monkeypatch.setenv("FINGROUND_OCR_MAX_PARALLEL_PAGES", "2")
    monkeypatch.setenv("FINGROUND_OCR_CACHE_VERSION", "annual-report-v2")
    monkeypatch.setenv("FINGROUND_OCR_TRUST_ENV", "true")

    config = OcrConfig.from_env()

    assert config.base_url == "http://ocr.example"
    assert config.model == "ocr-model"
    assert config.batch_pages == 15
    assert config.max_parallel_pages == 2
    assert config.cache_version == "annual-report-v2"
    assert config.trust_env is True


def test_ocr_config_defaults_to_eight_parallel_requests() -> None:
    assert OcrConfig().max_parallel_pages == 8


def test_ocr_cache_fingerprint_changes_with_output_configuration() -> None:
    baseline = VllmPdfOcrClient(OcrConfig()).cache_fingerprint()

    assert VllmPdfOcrClient(OcrConfig(render_dpi=200)).cache_fingerprint() != baseline
    assert VllmPdfOcrClient(OcrConfig(cache_version="v2")).cache_fingerprint() != baseline


def test_batch_size_must_stay_in_supported_range() -> None:
    with pytest.raises(ValueError, match="between 10 and 30"):
        VllmPdfOcrClient(OcrConfig(batch_pages=9))


def test_single_page_retries_truncated_output_with_larger_budget() -> None:
    requested_budgets: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requested_budgets.append(payload["max_tokens"])
        if payload["max_tokens"] == 8_192:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "partial markdown"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "complete page markdown"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> str:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_page(
                http_client,
                b"png bytes",
                page_number=66,
                semaphore=asyncio.Semaphore(1),
            )

    assert asyncio.run(run()) == "complete page markdown"
    assert requested_budgets == [8_192, 24_576]


def test_single_page_rejects_output_truncated_after_retry() -> None:
    requested_budgets: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requested_budgets.append(payload["max_tokens"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "partial markdown"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    async def run() -> str:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_page(
                http_client,
                b"png bytes",
                page_number=66,
                semaphore=asyncio.Semaphore(1),
            )

    with pytest.raises(OcrError, match="truncated starting at PDF page 66"):
        asyncio.run(run())
    assert requested_budgets == [8_192, 24_576]


def test_truncated_batch_preserves_complete_pages_and_retries_only_tail() -> None:
    request_sizes: list[int] = []
    image_modes: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_count = sum(part["type"] == "image_url" for part in payload["messages"][0]["content"])
        request_sizes.append(image_count)
        image_modes.append(payload["images_config"]["image_mode"])
        if image_count == 4:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "page one<PAGE>partial page two"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "page two<PAGE>page three<PAGE>page four"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> list[str]:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_batch_with_fallback(
                http_client,
                [b"1", b"2", b"3", b"4"],
                first_page_number=1,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == ["page one", "page two", "page three", "page four"]
    assert request_sizes == [4, 3]
    assert image_modes == ["base", "base"]


def test_malformed_batch_is_bisected_before_single_page_fallback() -> None:
    request_sizes: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_count = sum(part["type"] == "image_url" for part in payload["messages"][0]["content"])
        request_sizes.append(image_count)
        content = "only one section" if image_count == 4 else "left<PAGE>right"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
        )

    async def run() -> list[str]:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_batch_with_fallback(
                http_client,
                [b"1", b"2", b"3", b"4"],
                first_page_number=1,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == ["left", "right", "left", "right"]
    assert request_sizes == [4, 2, 2]


def test_timed_out_batch_is_bisected_into_smaller_requests() -> None:
    request_sizes: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_count = sum(part["type"] == "image_url" for part in payload["messages"][0]["content"])
        request_sizes.append(image_count)
        if image_count == 4:
            raise httpx.ReadTimeout("", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "left<PAGE>right"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> list[str]:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_batch_with_fallback(
                http_client,
                [b"1", b"2", b"3", b"4"],
                first_page_number=161,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == ["left", "right", "left", "right"]
    assert request_sizes == [4, 2, 2]


def test_single_page_timeout_retries_once_and_keeps_error_details() -> None:
    requests = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ReadTimeout("", request=request)

    async def run() -> str:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig(timeout_seconds=180))
            return await client._ocr_page(
                http_client,
                b"png bytes",
                page_number=161,
                semaphore=asyncio.Semaphore(1),
            )

    with pytest.raises(OcrError, match=r"ReadTimeout.*180.*page 161"):
        asyncio.run(run())
    assert requests == 2


def test_batch_read_error_retries_same_batch_before_fallback() -> None:
    request_sizes: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_count = sum(part["type"] == "image_url" for part in payload["messages"][0]["content"])
        request_sizes.append(image_count)
        if len(request_sizes) == 1:
            raise httpx.ReadError("", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "one<PAGE>two<PAGE>three<PAGE>four"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> list[str]:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_batch_with_fallback(
                http_client,
                [b"1", b"2", b"3", b"4"],
                first_page_number=121,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == ["one", "two", "three", "four"]
    assert request_sizes == [4, 4]


def test_repeated_batch_read_error_splits_only_once() -> None:
    request_sizes: list[int] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_count = sum(part["type"] == "image_url" for part in payload["messages"][0]["content"])
        request_sizes.append(image_count)
        if image_count == 4:
            raise httpx.ReadError("", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "left<PAGE>right"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> list[str]:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_batch_with_fallback(
                http_client,
                [b"1", b"2", b"3", b"4"],
                first_page_number=121,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == ["left", "right", "left", "right"]
    assert request_sizes == [4, 4, 2, 2]


def test_single_page_read_error_retries_once() -> None:
    requests = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ReadError("", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "page"}, "finish_reason": "stop"}]},
        )

    async def run() -> str:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_page(
                http_client,
                b"png bytes",
                page_number=121,
                semaphore=asyncio.Semaphore(1),
            )

    assert asyncio.run(run()) == "page"
    assert requests == 2


def test_single_page_uses_gundam_image_mode() -> None:
    image_modes: list[str] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_modes.append(payload["images_config"]["image_mode"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "page"}, "finish_reason": "stop"}]},
        )

    async def run() -> str:
        transport = httpx.MockTransport(handle_request)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VllmPdfOcrClient(OcrConfig())
            return await client._ocr_page(
                http_client,
                b"png bytes",
                page_number=1,
                semaphore=asyncio.Semaphore(8),
            )

    assert asyncio.run(run()) == "page"
    assert image_modes == ["gundam"]


def test_pdf_rendering_is_pipelined_with_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def render_pages(
        _document: fitz.Document,
        start: int,
        end: int,
        _matrix: fitz.Matrix,
    ) -> list[bytes]:
        events.append(f"render:{start + 1}")
        return [str(index).encode() for index in range(start, end)]

    async def ocr_batch(
        _http_client: httpx.AsyncClient,
        images: list[bytes],
        first_page_number: int,
        _semaphore: asyncio.Semaphore,
    ) -> list[str]:
        events.append(f"ocr:{first_page_number}")
        await asyncio.sleep(0)
        return [image.decode() for image in images]

    document = fitz.open()
    for _ in range(20):
        document.new_page()
    pdf_data = document.tobytes()
    document.close()

    client = VllmPdfOcrClient(OcrConfig(batch_pages=10))
    monkeypatch.setattr(client, "_render_pages", render_pages)
    monkeypatch.setattr(client, "_ocr_batch_with_fallback", ocr_batch)

    markdown, page_count = asyncio.run(client.pdf_to_markdown(pdf_data))

    assert page_count == 20
    assert markdown.startswith("0\n\n<--- Page Split --->\n\n1")
    assert events.index("ocr:1") < events.index("render:11")
