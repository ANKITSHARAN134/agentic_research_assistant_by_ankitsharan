import io

import pytest

from agent.agent import ResearchAgent
from agent.rag import VectorStore
from app import create_app


class StubAgent(ResearchAgent):
    def chat(self, session_id, message):
        return {"answer": f"echo: {message}", "steps": [], "sources": []}


@pytest.fixture
def client(tmp_path):
    app = create_app(StubAgent(VectorStore(tmp_path)))
    return app.test_client()


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["retrieval"] == "tfidf"


def test_chat_roundtrip_and_validation(client):
    assert client.post("/chat", json={}).status_code == 400
    res = client.post("/chat", json={"message": "hi", "session_id": "abc"})
    assert res.status_code == 200
    assert res.get_json()["answer"] == "echo: hi"
    assert res.get_json()["session_id"] == "abc"


def test_ingest_and_list(client):
    data = {"file": (io.BytesIO(b"Spring Boot builds REST APIs in Java."), "notes.md")}
    res = client.post("/ingest", data=data, content_type="multipart/form-data")
    assert res.status_code == 200 and res.get_json()["chunks"] == 1
    assert client.get("/documents").get_json()["documents"][0]["source"] == "notes.md"


def test_ingest_rejects_bad_type(client):
    data = {"file": (io.BytesIO(b"MZ"), "virus.exe")}
    assert client.post("/ingest", data=data, content_type="multipart/form-data").status_code == 400


def test_missing_api_key_returns_503(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(ResearchAgent(VectorStore(tmp_path)))
    res = app.test_client().post("/chat", json={"message": "hi"})
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.get_json()["error"]
