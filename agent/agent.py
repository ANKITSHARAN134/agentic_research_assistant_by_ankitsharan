"""Tool-calling research agent built on the Anthropic Messages API."""
from __future__ import annotations

import os
import threading

from . import config
from .rag import VectorStore
from .tools import TOOL_SCHEMAS, Toolbox

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question by gathering evidence with your tools, then writing a clear, well-organised answer.

- Use search_documents first when the question may relate to the user's uploaded files.
- Use web_search for current or external information; use read_webpage to read a promising result in full.
- You may call several tools across multiple steps. Stop searching once you have enough evidence.
- Base claims on tool results, mention which document or URL supports them, and say so when evidence is missing or conflicting.
- Be concise."""


class ResearchAgent:
    def __init__(self, store: VectorStore, client=None, model: str | None = None):
        self.store = store
        self._client = client
        self.model = model or config.MODEL
        self.sessions: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def reset(self, session_id: str) -> None:
        with self._lock:
            self.sessions.pop(session_id, None)

    def chat(self, session_id: str, message: str) -> dict:
        with self._lock:
            history = list(self.sessions.get(session_id, []))
        messages = history + [{"role": "user", "content": message}]
        toolbox = Toolbox(self.store)
        steps: list[dict] = []
        answer = ""

        for _ in range(config.MAX_STEPS):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    steps.append({"tool": block.name, "input": block.input})
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": toolbox.run(block.name, block.input),
                        }
                    )
            if not results:
                answer = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
                break
            messages.append({"role": "user", "content": results})
        else:
            answer = (
                "I reached the step limit before finishing. Try a narrower question "
                "or raise MAX_STEPS."
            )

        # Memory keeps only the text of each turn (not intermediate tool traffic).
        with self._lock:
            turns = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer or "(no answer)"},
            ]
            self.sessions[session_id] = turns[-2 * config.HISTORY_TURNS :]

        return {"answer": answer, "steps": steps, "sources": toolbox.sources}
