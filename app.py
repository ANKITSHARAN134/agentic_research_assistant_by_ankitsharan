"""Flask API for the research agent."""
from __future__ import annotations

import io
import logging
import uuid

from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from agent import config
from agent.agent import ResearchAgent
from agent.rag import VectorStore

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

log = logging.getLogger(__name__)


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def create_app(agent: ResearchAgent | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

    if agent is None:
        agent = ResearchAgent(VectorStore(config.DATA_DIR / "index"))
    store = agent.store

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", retrieval=store.backend, model=agent.model)

    @app.post("/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify(error="'message' is required."), 400
        session_id = body.get("session_id") or str(uuid.uuid4())
        try:
            result = agent.chat(session_id, message)
        except RuntimeError as exc:  # e.g. missing API key
            return jsonify(error=str(exc)), 503
        except Exception as exc:
            log.exception("Agent error")
            err_msg = str(exc) if "anthropic" in type(exc).__module__.lower() else "The agent failed to complete the request."
            return jsonify(error=err_msg), 500
        return jsonify(session_id=session_id, **result)

    @app.post("/reset")
    def reset():
        body = request.get_json(silent=True) or {}
        if body.get("session_id"):
            agent.reset(body["session_id"])
        return jsonify(status="reset")

    @app.post("/ingest")
    def ingest():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="Attach a file in the 'file' form field."), 400
        name = secure_filename(file.filename)
        if not name or not any(name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            return jsonify(error="Supported types: .txt, .md, .pdf"), 400
        try:
            text = extract_text(name, file.read())
        except Exception:
            return jsonify(error="Could not read that file."), 400
        chunks = store.add_document(name, text)
        if chunks == 0:
            return jsonify(error="No text found in the file."), 400
        return jsonify(source=name, chunks=chunks)

    @app.get("/documents")
    def documents():
        return jsonify(documents=store.list_sources())

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="0.0.0.0", port=8000, debug=False)
