from docqa.evals import run_eval
import json
from pathlib import Path

CASES = json.loads((Path(__file__).resolve().parents[1] / "evals" / "eval_set.json").read_text())


def test_eval_set_meets_the_quality_bar(agent, spans):
    report = run_eval(agent, CASES)
    assert report.retrieval_hit_rate >= 0.9, report.failures
    assert report.answer_accuracy >= 0.8, report.failures
    assert report.refusal_accuracy >= 0.9, report.failures


def test_eval_reports_failures_for_a_wrong_expectation(agent, spans):
    bad = [{"question": "How long must passwords be?", "expected_source": "expenses.md", "expected_keyword": "zzz"}]
    report = run_eval(agent, bad)
    assert report.retrieval_hit_rate == 0.0
    assert report.answer_accuracy == 0.0
    assert report.failures
