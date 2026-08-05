import pytest

from finground.ocr_client import OcrConfig, OcrError, VllmPdfOcrClient


def test_invalid_pdf_is_rejected_before_network_call() -> None:
    client = VllmPdfOcrClient(OcrConfig())

    with pytest.raises(OcrError, match="not a readable PDF"):
        import asyncio

        asyncio.run(client.pdf_to_markdown(b"not a pdf"))


def test_ocr_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGROUND_OCR_BASE_URL", "http://ocr.example/")
    monkeypatch.setenv("FINGROUND_OCR_MODEL", "ocr-model")
    monkeypatch.setenv("FINGROUND_OCR_BATCH_PAGES", "15")
    monkeypatch.setenv("FINGROUND_OCR_MAX_PARALLEL_PAGES", "2")

    config = OcrConfig.from_env()

    assert config.base_url == "http://ocr.example"
    assert config.model == "ocr-model"
    assert config.batch_pages == 15
    assert config.max_parallel_pages == 2


def test_batch_size_must_stay_in_supported_range() -> None:
    with pytest.raises(ValueError, match="between 10 and 30"):
        VllmPdfOcrClient(OcrConfig(batch_pages=9))
