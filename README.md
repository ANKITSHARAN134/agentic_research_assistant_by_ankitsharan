# AI Research Agent

An LLM-powered research agent that decides for itself which tools to use, gathers evidence over multiple steps, and answers with sources.

**Tools the agent can call**

| Tool | What it does |
|------|--------------|
| `search_documents` | Semantic search (RAG) over PDFs, TXT and MD files you upload |
| `web_search` | Web search via DuckDuckGo (no API key needed) |
| `read_webpage` | Fetches and reads a public web page (private/internal addresses are blocked) |

It keeps per-session conversation memory, so follow-up questions work. The response includes the answer, the tools it used, and its sources.

## Architecture

```
Browser / API client
        |
     Flask API (app.py)  --  /chat  /ingest  /documents  /reset  /health
        |
  ResearchAgent (agent/agent.py)  -- tool-calling loop on the Anthropic Messages API
        |
   Toolbox (agent/tools.py) -----> web_search | read_webpage | search_documents
                                                                   |
                                              VectorStore (agent/rag.py)
                                     chunking + embeddings + cosine similarity
```

Retrieval uses local embeddings (`fastembed`, model `BAAI/bge-small-en-v1.5`, downloaded on first run). If embeddings are unavailable, or `EMBEDDINGS=tfidf` is set, it falls back to TF-IDF. `/health` shows which backend is active.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then put your ANTHROPIC_API_KEY in .env
python app.py                 # http://localhost:8000
```

Open http://localhost:8000 for the chat UI, or use the API:

```bash
# index a document
curl -F file=@docs/sample_notes.md http://localhost:8000/ingest

# ask a question
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "How does the JVM manage memory?"}'
```

## Tests

```bash
pip install -r requirements-dev.txt
EMBEDDINGS=tfidf pytest -q
```

Tests use a scripted fake LLM client, so they need no API key or network.

## Docker

```bash
docker build -t ai-research-agent .
docker run -p 8000:8000 --env-file .env ai-research-agent
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI research agent"
git branch -M main
git remote add origin https://github.com/ANKITSHARAN134/agentic_research_assistant_by_ankitsharan.git
git push -u origin main
```

GitHub Actions (`.github/workflows/ci.yml`) runs the tests and a Docker build on every push.

GitHub itself only hosts the code. To run the app online, deploy the Docker image to a host such as Render, Railway or Fly.io, and set `ANTHROPIC_API_KEY` as an environment variable there. Never commit `.env`.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | required | API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the agent |
| `EMBEDDINGS` | `auto` | Set to `tfidf` to skip the embedding model |
| `MAX_STEPS` | `6` | Max tool-use rounds per question |
| `HISTORY_TURNS` | `10` | Conversation turns remembered per session |
| `DATA_DIR` | `data` | Where the document index is stored |

## Limitations

- Conversation memory is in-process, so it resets on restart. The Docker image runs one worker for this reason.
- No authentication on the API. Add auth before exposing it publicly, since each request spends API credits.
- Web search quality depends on DuckDuckGo results.
