"""The agent: retrieve, decide whether the evidence is good enough, then answer.

Every step is a span, so a slow or wrong answer can be traced back to the
retrieval scores and the LLM call that produced it."""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry.trace import format_trace_id

from .llm import LLM, get_llm
from .retriever import Retriever
from .tracing import get_tracer

REFUSAL = "I don't know based on the provided documents."


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    top_score: float = 0.0
    refused: bool = False
    trace_id: str = ""


class DocQAAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None = None,
        k: int = 3,
        min_score: float = 0.15,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.k = k
        self.min_score = min_score
        self._tracer = get_tracer()

    def ask(self, question: str) -> Answer:
        with self._tracer.start_as_current_span("docqa.ask") as root:
            root.set_attribute("docqa.question_length", len(question))
            trace_id = format_trace_id(root.get_span_context().trace_id)

            with self._tracer.start_as_current_span("docqa.retrieve") as span:
                hits = self.retriever.search(question, k=self.k)
                top_score = hits[0][1] if hits else 0.0
                span.set_attribute("docqa.hits", len(hits))
                span.set_attribute("docqa.top_score", top_score)

            if top_score < self.min_score:
                root.set_attribute("docqa.refused", True)
                return Answer(REFUSAL, [], top_score, True, trace_id)

            with self._tracer.start_as_current_span("docqa.generate") as span:
                span.set_attribute("docqa.llm", self.llm.name)
                text = self.llm.generate(question, [chunk.text for chunk, _ in hits])
                span.set_attribute("docqa.answer_length", len(text))

            root.set_attribute("docqa.refused", False)
            sources = sorted({chunk.source for chunk, _ in hits})
            return Answer(text, sources, top_score, False, trace_id)
