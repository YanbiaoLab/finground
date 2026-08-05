"""Non-secret runtime configuration used while collecting the test suite."""

import os

os.environ.setdefault("FINGROUND_MODEL", "openai/test-model")
os.environ.setdefault("FINGROUND_MODEL_BASE_URL", "https://model.test/v1")
os.environ.setdefault("FINGROUND_MODEL_API_KEY", "test-key")
os.environ.setdefault("FINGROUND_OCR_BASE_URL", "https://ocr.test")
os.environ.setdefault("FINGROUND_OCR_MODEL", "test-ocr-model")
