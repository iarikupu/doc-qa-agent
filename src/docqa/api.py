"""FastAPI wrapper: POST /ask {"question": "..."} -> answer, sources, trace id."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import DocQAAgent
from .retriever import Retriever
from .tracing import configure_tracing

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "data" / "sample_docs"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    top_score: float
    trace_id: str


def create_app(docs_dir: str | Path | None = None, trace: bool = True) -> FastAPI:
    if trace:
        configure_tracing()
    directory = docs_dir or os.environ.get("DOCQA_DOCS", DEFAULT_DOCS)
    agent = DocQAAgent(Retriever.from_directory(directory))
    app = FastAPI(title="docqa")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        result = agent.ask(body.question)
        return AskResponse(
            answer=result.text,
            sources=result.sources,
            refused=result.refused,
            top_score=result.top_score,
            trace_id=result.trace_id,
        )

    return app

