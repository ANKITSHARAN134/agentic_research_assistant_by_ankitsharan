"""Tools the agent can call: web search, page reading, document search."""
from __future__ import annotations

import ipaddress
import json
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .rag import VectorStore

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the user's uploaded documents (PDF/TXT/MD). Use this first for "
            "questions about the user's own files or notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for"}},
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Returns titles, URLs and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_webpage",
        "description": "Fetch a public web page and return its readable text (truncated).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL"}},
            "required": ["url"],
        },
    },
]

MAX_PAGE_CHARS = 6000


def is_safe_url(url: str) -> bool:
    """Reject non-http(s) URLs and hosts resolving to private/loopback addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = list(DDGS().text(query, max_results=max_results) or [])
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]


def read_webpage(url: str) -> dict:
    if not is_safe_url(url):
        return {"error": "URL not allowed (must be a public http/https address)."}
    resp = requests.get(url, timeout=10, headers={"User-Agent": "ai-research-agent/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    title = soup.title.get_text().strip() if soup.title and soup.title.get_text() else url
    return {"title": title, "url": url, "text": text[:MAX_PAGE_CHARS]}


class Toolbox:
    """Executes tool calls for one chat request and records the sources used."""

    def __init__(self, store: VectorStore):
        self.store = store
        self.sources: list[dict] = []

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "search_documents":
                hits = self.store.search(args["query"])
                for h in hits:
                    self._add_source("document", h["source"])
                result = hits or {"message": "No relevant passages found in uploaded documents."}
            elif name == "web_search":
                hits = web_search(args["query"])
                for h in hits:
                    self._add_source("web", h["url"], h["title"])
                result = hits or {"message": "No web results."}
            elif name == "read_webpage":
                result = read_webpage(args["url"])
                if "error" not in result:
                    self._add_source("web", result["url"], result["title"])
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:  # tool failures are reported to the model, not raised
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)

    def _add_source(self, kind: str, ref: str, title: str | None = None) -> None:
        entry = {"type": kind, "ref": ref}
        if title:
            entry["title"] = title
        if entry not in self.sources:
            self.sources.append(entry)
