"""ADK discovery entry point."""

from google.adk.apps import App
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

from finground.context_compaction import ScopedContextCompactionPlugin
from finground.report_plugin import ReportUploadPlugin
from finground.root_agent import create_root_agent
from finground.task_plugin import TaskProgressPlugin

root_agent = create_root_agent()
app = App(
    name="finground",
    root_agent=root_agent,
    plugins=[
        ReportUploadPlugin(),
        TaskProgressPlugin(),
        ScopedContextCompactionPlugin(
            summarizer=LlmEventSummarizer(llm=root_agent.canonical_model)
        ),
    ],
)
