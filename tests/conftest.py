from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from docqa.agent import DocQAAgent
from docqa.retriever import Retriever
from docqa.tracing import configure_tracing

DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"

_exporter = InMemorySpanExporter()
configure_tracing(exporter=_exporter, simple=True)


@pytest.fixture
def spans():
    _exporter.clear()
    return _exporter


@pytest.fixture(scope="session")
def agent():
    return DocQAAgent(Retriever.from_directory(DOCS))
