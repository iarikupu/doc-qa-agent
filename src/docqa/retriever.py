"""TF-IDF retrieval over paragraph-sized chunks of markdown documents.

This is deliberately classical retrieval (no embeddings, no vector database):
it has no network or GPU dependency, and it is easy to test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def split_into_chunks(source: str, text: str, min_chars: int = 40) -> list[Chunk]:
    """Split a document into paragraph chunks, dropping tiny fragments."""
    chunks: list[Chunk] = []
    for para in text.split("\n\n"):
        cleaned = " ".join(para.split())
        if len(cleaned) >= min_chars:
            chunks.append(Chunk(source=source, index=len(chunks), text=cleaned))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(c.text for c in chunks)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Retriever":
        chunks: list[Chunk] = []
        for path in sorted(Path(directory).glob("*.md")):
            chunks.extend(split_into_chunks(path.name, path.read_text(encoding="utf-8")))
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, cosine score) pairs, best first, score > 0 only."""
        query_vec = self._vectorizer.transform([query])
        scores = linear_kernel(query_vec, self._matrix).ravel()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
