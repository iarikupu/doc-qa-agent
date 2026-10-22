from fastapi.testclient import TestClient

from docqa.api import create_app

from .conftest import DOCS

client = TestClient(create_app(DOCS, trace=False))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_sources_and_trace_id():
    response = client.post("/ask", json={"question": "How long must passwords be?"})
    body = response.json()
    assert response.status_code == 200
    assert "14 characters" in body["answer"]
    assert "security.md" in body["sources"]
    assert len(body["trace_id"]) == 32
    assert body["refused"] is False


def test_out_of_scope_question_is_refused():
    body = client.post("/ask", json={"question": "Who won the football world cup in 2010?"}).json()
    assert body["refused"] is True


def test_empty_question_is_rejected_with_422():
    assert client.post("/ask", json={"question": ""}).status_code == 422
