"""LLM backends. The default backend is local and deterministic so the whole
project (tests, evals, CI) runs without network access or an API key."""

from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    name: str

    def generate(self, question: str, context: list[str]) -> str: ...


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


class _Words:
    """Tokenizer with a naive plural strip so 'engineers' matches 'engineer'."""

    @staticmethod
    def findall(text: str) -> list[str]:
        return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD_RE.findall(text)]


_WORD = _Words()


_STOP = frozenset(
    "a an and are as at be by can do does for from how in is it its many much of on or the to what when where which who why with".split()
)


class ExtractiveLLM:
    """Return the two context sentences that share the most content words with
    the question, kept in their original order. Deterministic and offline."""

    name = "extractive"

    def generate(self, question: str, context: list[str]) -> str:
        q_words = set(_WORD.findall(question.lower())) - _STOP
        scored: list[tuple[int, int, str]] = []
        position = 0
        for passage in context:
            for sentence in _SENTENCE.split(passage):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(q_words & (set(_WORD.findall(sentence.lower())) - _STOP))
                scored.append((overlap, position, sentence))
                position += 1
        best = sorted(scored, key=lambda t: (-t[0], t[1]))[:2]
        return " ".join(sentence for _, _, sentence in sorted(best, key=lambda t: t[1]))


class OpenAILLM:
    """Optional backend using the OpenAI chat completions API.

    Requires `pip install openai` and OPENAI_API_KEY. Not used by tests or CI.
    """

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # imported lazily so the package stays optional

        self._client = OpenAI()
        self._model = model or os.environ.get("DOCQA_OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, question: str, context: list[str]) -> str:
        joined = "\n\n".join(context)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the provided context. If the context "
                    "does not contain the answer, say you do not know.",
                },
                {"role": "user", "content": f"Context:\n{joined}\n\nQuestion: {question}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def get_llm(kind: str | None = None) -> LLM:
    kind = (kind or os.environ.get("DOCQA_LLM", "extractive")).lower()
    if kind == "openai":
        return OpenAILLM()
    return ExtractiveLLM()
