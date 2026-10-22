"""Evaluation harness: score the agent against a labelled question set.

Three metrics, because one number hides different failures:
- retrieval_hit_rate: did the expected source file appear in the sources?
- answer_accuracy:    does the answer contain the expected keyword?
- refusal_accuracy:   does it refuse the out-of-scope questions (and only those)?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .agent import DocQAAgent
from .retriever import Retriever
from .tracing import configure_tracing

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalReport:
    retrieval_hit_rate: float
    answer_accuracy: float
    refusal_accuracy: float
    failures: list[str]

    def passed(self, threshold: float) -> bool:
        return min(self.retrieval_hit_rate, self.answer_accuracy, self.refusal_accuracy) >= threshold


def run_eval(agent: DocQAAgent, cases: list[dict]) -> EvalReport:
    answerable = [c for c in cases if c.get("expected_source")]
    unanswerable = [c for c in cases if not c.get("expected_source")]
    failures: list[str] = []
    hits = correct = refused_ok = 0

    for case in answerable:
        result = agent.ask(case["question"])
        if case["expected_source"] in result.sources:
            hits += 1
        else:
            failures.append(f"retrieval: {case['question']!r} -> {result.sources}")
        if case["expected_keyword"].lower() in result.text.lower() and not result.refused:
            correct += 1
        else:
            failures.append(f"answer: {case['question']!r} -> {result.text!r}")

    for case in unanswerable:
        result = agent.ask(case["question"])
        if result.refused:
            refused_ok += 1
        else:
            failures.append(f"should refuse: {case['question']!r} -> {result.text!r}")

    def rate(n: int, d: int) -> float:
        return n / d if d else 1.0

    return EvalReport(
        rate(hits, len(answerable)),
        rate(correct, len(answerable)),
        rate(refused_ok, len(unanswerable)),
        failures,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the docqa agent")
    parser.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    parser.add_argument("--cases", default=str(ROOT / "evals" / "eval_set.json"))
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args(argv)

    configure_tracing(simple=True, exporter=_NullExporter())
    agent = DocQAAgent(Retriever.from_directory(args.docs))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = run_eval(agent, cases)

    print(f"retrieval_hit_rate: {report.retrieval_hit_rate:.2f}")
    print(f"answer_accuracy:    {report.answer_accuracy:.2f}")
    print(f"refusal_accuracy:   {report.refusal_accuracy:.2f}")
    for failure in report.failures:
        print(f"  FAIL {failure}")
    return 0 if report.passed(args.threshold) else 1


class _NullExporter:
    """Discard spans during evals so the report output stays readable."""

    def export(self, spans):  # noqa: ANN001
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


if __name__ == "__main__":
    sys.exit(main())
