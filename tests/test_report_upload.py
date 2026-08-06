import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from finground.ocr_client import OcrError, PdfExtractionError
from finground.report_plugin import ReportUploadPlugin


class StubOcrClient:
    def __init__(self, cache_version: str = "stub-v1") -> None:
        self.cache_version = cache_version
        self.calls = 0

    def cache_fingerprint(self) -> str:
        return self.cache_version

    async def pdf_to_markdown(self, pdf_data: bytes) -> tuple[str, int]:
        assert pdf_data == b"fake pdf"
        self.calls += 1
        return "# Page one\n\nRevenue 123\n\n<--- Page Split --->\n\n# Page two", 2


class FailingOcrClient(StubOcrClient):
    async def pdf_to_markdown(self, pdf_data: bytes) -> tuple[str, int]:
        del pdf_data
        self.calls += 1
        raise OcrError("service unavailable")

class FinalResponseLlm(BaseLlm):
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Report ready")],
            )
        )


def test_markdown_attachment_becomes_current_report_artifact() -> None:
    async def run() -> tuple[dict, types.Part, list]:
        root = Agent(
            name="root",
            model=FinalResponseLlm(model="scripted"),
            instruction="Acknowledge the uploaded report.",
        )
        app = App(
            name="report_upload_test",
            root_agent=root,
            plugins=[ReportUploadPlugin()],
        )
        sessions = InMemorySessionService()
        artifacts = InMemoryArtifactService()
        await sessions.create_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        runner = Runner(app=app, session_service=sessions, artifact_service=artifacts)
        _ = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="revenue 是多少"),
                        types.Part(
                            inline_data=types.Blob(
                                data=b"# 2017 Annual Report\n\nRevenue 123",
                                mime_type="text/markdown",
                                display_name="AAP_2017.md",
                            )
                        ),
                    ],
                ),
            )
        ]
        session = await sessions.get_session(
            app_name=app.name,
            user_id="user",
            session_id="session",
        )
        artifact = await artifacts.load_artifact(
            app_name=app.name,
            user_id="user",
            session_id="session",
            filename="AAP_2017.md",
        )
        return session.state, artifact, session.events

    state, artifact, events = asyncio.run(run())

    assert state["report"] == {
        "report_ref": "AAP_2017",
        "artifact_name": "AAP_2017.md",
        "mime_type": "text/markdown",
        "total_pages": 1,
        "total_chars": 33,
    }
    assert artifact.inline_data.data == b"# 2017 Annual Report\n\nRevenue 123"
    user_event = next(event for event in events if event.author == "user")
    assert all(part.inline_data is None for part in user_event.content.parts)
    assert 'report_ref "AAP_2017"' in user_event.content.parts[1].text


def test_pdf_attachment_is_ocrd_into_markdown_artifact() -> None:
    async def run() -> tuple[dict, types.Part, list]:
        root = Agent(
            name="root",
            model=FinalResponseLlm(model="scripted"),
            instruction="Acknowledge the uploaded report.",
        )
        app = App(
            name="pdf_upload_test",
            root_agent=root,
            plugins=[ReportUploadPlugin(ocr_client=StubOcrClient())],
        )
        sessions = InMemorySessionService()
        artifacts = InMemoryArtifactService()
        await sessions.create_session(app_name=app.name, user_id="user", session_id="session")
        runner = Runner(app=app, session_service=sessions, artifact_service=artifacts)
        _ = [
            event
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="revenue 是多少"),
                        types.Part(
                            inline_data=types.Blob(
                                data=b"fake pdf",
                                mime_type="application/pdf",
                                display_name="AAP_2017.pdf",
                            )
                        ),
                    ],
                ),
            )
        ]
        session = await sessions.get_session(
            app_name=app.name, user_id="user", session_id="session"
        )
        artifact = await artifacts.load_artifact(
            app_name=app.name,
            user_id="user",
            session_id="session",
            filename="AAP_2017.ocr.md",
        )
        return session.state, artifact, session.events

    state, artifact, events = asyncio.run(run())

    assert state["report"] == {
        "report_ref": "AAP_2017_ocr",
        "artifact_name": "AAP_2017.ocr.md",
        "mime_type": "text/markdown",
        "source_mime_type": "application/pdf",
        "extraction_method": "ocr",
        "total_pages": 2,
        "total_chars": 57,
    }
    assert artifact.inline_data.mime_type == "text/markdown"
    assert b"Revenue 123" in artifact.inline_data.data
    user_event = next(event for event in events if event.author == "user")
    assert all(part.inline_data is None for part in user_event.content.parts)
    assert 'report_ref "AAP_2017_ocr"' in user_event.content.parts[1].text


def test_pdf_without_configured_ocr_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINGROUND_OCR_BASE_URL")
    monkeypatch.delenv("FINGROUND_OCR_MODEL")
    plugin = ReportUploadPlugin()
    context = SimpleNamespace(
        artifact_service=InMemoryArtifactService(),
        app_name="pdf_without_ocr_test",
        user_id="user",
        session=SimpleNamespace(id="session"),
    )

    with pytest.raises(PdfExtractionError, match="PDF OCR is not configured"):
        asyncio.run(plugin._pdf_to_markdown(context, b"fake pdf"))


def test_pdf_ocr_failure_is_propagated() -> None:
    client = FailingOcrClient()
    plugin = ReportUploadPlugin(ocr_client=client)
    context = SimpleNamespace(
        artifact_service=InMemoryArtifactService(),
        app_name="pdf_ocr_failure_test",
        user_id="user",
        session=SimpleNamespace(id="session"),
    )

    with pytest.raises(OcrError, match="service unavailable"):
        asyncio.run(plugin._pdf_to_markdown(context, b"fake pdf"))

    assert client.calls == 1


def test_pdf_ocr_cache_is_reused_across_sessions_and_users() -> None:
    async def run() -> tuple[int, types.Part, list[str]]:
        ocr_client = StubOcrClient()
        root = Agent(
            name="root",
            model=FinalResponseLlm(model="scripted"),
            instruction="Acknowledge the uploaded report.",
        )
        app = App(
            name="pdf_cache_test",
            root_agent=root,
            plugins=[ReportUploadPlugin(ocr_client=ocr_client)],
        )
        sessions = InMemorySessionService()
        artifacts = InMemoryArtifactService()
        runner = Runner(app=app, session_service=sessions, artifact_service=artifacts)

        uploads = (
            ("user", "one", "first.pdf"),
            ("user", "two", "renamed.pdf"),
            ("other-user", "three", "same.pdf"),
        )
        for user_id, session_id, filename in uploads:
            await sessions.create_session(
                app_name=app.name,
                user_id=user_id,
                session_id=session_id,
            )
            _ = [
                event
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    data=b"fake pdf",
                                    mime_type="application/pdf",
                                    display_name=filename,
                                )
                            )
                        ],
                    ),
                )
            ]

        second_artifact = await artifacts.load_artifact(
            app_name=app.name,
            user_id="user",
            session_id="two",
            filename="renamed.ocr.md",
        )
        cache_keys = await artifacts.list_artifact_keys(
            app_name=app.name,
            user_id="finground-ocr-cache",
            session_id="global",
        )
        return ocr_client.calls, second_artifact, cache_keys

    calls, artifact, cache_keys = asyncio.run(run())

    assert calls == 1
    assert b"Revenue 123" in artifact.inline_data.data
    assert any(key.startswith(".finground/ocr/") for key in cache_keys)
