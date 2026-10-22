from docqa.llm import ExtractiveLLM, get_llm


def test_extractive_llm_picks_the_most_relevant_sentences():
    context = ["Cats sleep a lot. Passwords must be at least 14 characters long. The cafeteria opens at nine."]
    answer = ExtractiveLLM().generate("How long must passwords be?", context)
    assert "14 characters" in answer
    assert "cafeteria" not in answer


def test_get_llm_defaults_to_the_offline_backend(monkeypatch):
    monkeypatch.delenv("DOCQA_LLM", raising=False)
    assert get_llm().name == "extractive"
