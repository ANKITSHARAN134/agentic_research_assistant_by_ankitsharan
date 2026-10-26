"""Document store with semantic retrieval (RAG).

Uses local embeddings (fastembed, ONNX) with cosine similarity over a numpy
matrix persisted to disk. If embeddings are unavailable (or EMBEDDINGS=tfidf),
it falls back to TF-IDF so the app still works offline.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def _load_embedder():
    if os.getenv("EMBEDDINGS", "auto").lower() == "tfidf":
        return None
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(EMBED_MODEL)
    except Exception as exc:  # missing package or model download failure
        log.warning("Embeddings unavailable (%s); falling back to TF-IDF.", exc)
        return None


class VectorStore:
    def __init__(self, store_dir: str | Path):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.dir / "chunks.json"
        self.vectors_path = self.dir / "vectors.npy"
        self.embedder = _load_embedder()
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._lock = threading.Lock()
        with self._lock:
            self._load()

    @property
    def backend(self) -> str:
        return "embeddings" if self.embedder is not None else "tfidf"

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.embedder.embed(texts)), dtype=np.float32)

    def _load(self) -> None:
        if self.chunks_path.exists():
            self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if self.embedder is not None and self.chunks:
            if self.vectors_path.exists():
                saved = np.load(self.vectors_path)
                if len(saved) == len(self.chunks):
                    self.vectors = saved
            if self.vectors is None:
                self.vectors = self._embed([c["text"] for c in self.chunks])
                self._save()

    def _save(self) -> None:
        self.chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)
        elif self.vectors_path.exists():
            self.vectors_path.unlink()

    def add_document(self, source: str, text: str) -> int:
        """Index a document. Re-adding the same source replaces it."""
        pieces = chunk_text(text)
        if not pieces:
            return 0
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
            self.chunks = [self.chunks[i] for i in keep]
            if self.vectors is not None:
                self.vectors = self.vectors[keep] if keep else None
            self.chunks += [{"source": source, "text": p} for p in pieces]
            if self.embedder is not None:
                new = self._embed(pieces)
                self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
            self._save()
        return len(pieces)

    def list_sources(self) -> list[dict]:
        with self._lock:
            counts: dict[str, int] = {}
            for c in self.chunks:
                counts[c["source"]] = counts.get(c["source"], 0) + 1
            return [{"source": s, "chunks": n} for s, n in counts.items()]

    def search(self, query: str, k: int = 4) -> list[dict]:
        with self._lock:
            if not self.chunks:
                return []
            if self.vectors is not None:
                scores = cosine_similarity(self._embed([query]), self.vectors)[0]
            else:
                try:
                    vec = TfidfVectorizer(stop_words="english")
                    matrix = vec.fit_transform([c["text"] for c in self.chunks] + [query])
                except ValueError:  # empty vocabulary
                    return []
                scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
            order = np.argsort(scores)[::-1][:k]
            return [
                {
                    "source": self.chunks[i]["source"],
                    "text": self.chunks[i]["text"],
                    "score": round(float(scores[i]), 3),
                }
                for i in order
                if scores[i] > 0
            ]
