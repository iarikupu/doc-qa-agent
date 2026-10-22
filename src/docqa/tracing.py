"""OpenTelemetry setup. Spans are exported to the console by default; tests
swap in an in-memory exporter to assert on the spans an agent call produces."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

SERVICE_NAME = "docqa"
_provider: TracerProvider | None = None


def configure_tracing(exporter: SpanExporter | None = None, simple: bool = False) -> TracerProvider:
    """Install a tracer provider once. Pass `simple=True` for synchronous export (tests)."""
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        trace.set_tracer_provider(_provider)
    processor_cls = SimpleSpanProcessor if simple else BatchSpanProcessor
    _provider.add_span_processor(processor_cls(exporter or ConsoleSpanExporter()))
    return _provider


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE_NAME)
