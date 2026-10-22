from docqa.agent import REFUSAL


def test_answers_from_the_right_document(agent, spans):
    result = agent.ask("How many days per week can employees work remotely?")
    assert not result.refused
    assert "three days" in result.text
    assert result.sources == ["remote_work.md"] or "remote_work.md" in result.sources


def test_refuses_out_of_scope_questions(agent, spans):
    result = agent.ask("What is the capital of Australia?")
    assert result.refused
    assert result.text == REFUSAL
    assert result.sources == []


def test_each_call_emits_root_retrieve_and_generate_spans(agent, spans):
    agent.ask("How long must passwords be?")
    names = [s.name for s in spans.get_finished_spans()]
    assert "docqa.ask" in names
    assert "docqa.retrieve" in names
    assert "docqa.generate" in names


def test_refusal_skips_the_generate_span(agent, spans):
    agent.ask("Recommend a good pasta recipe")
    names = [s.name for s in spans.get_finished_spans()]
    assert "docqa.retrieve" in names
    assert "docqa.generate" not in names


def test_trace_id_matches_the_root_span(agent, spans):
    result = agent.ask("What is the meal limit for travel?")
    root = next(s for s in spans.get_finished_spans() if s.name == "docqa.ask")
    assert result.trace_id == format(root.context.trace_id, "032x")
    assert root.attributes["docqa.refused"] is False


def test_retrieval_span_records_top_score(agent, spans):
    agent.ask("How long must passwords be?")
    span = next(s for s in spans.get_finished_spans() if s.name == "docqa.retrieve")
    assert span.attributes["docqa.top_score"] > 0
    assert span.attributes["docqa.hits"] >= 1
