from agent.rag import VectorStore, chunk_text


def test_chunk_text_overlaps_and_covers_everything():
    text = " ".join(f"w{i}" for i in range(450))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) == 3
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    assert chunks[-1].split()[-1] == "w449"


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_search_finds_relevant_document(tmp_path):
    store = VectorStore(tmp_path)
    store.add_document("java.txt", "Java uses the JVM and garbage collection to manage memory automatically.")
    store.add_document("cooking.txt", "Simmer the tomato sauce with garlic and basil for twenty minutes.")
    hits = store.search("how does the JVM handle garbage collection")
    assert hits and hits[0]["source"] == "java.txt"


def test_persistence_and_replace(tmp_path):
    store = VectorStore(tmp_path)
    store.add_document("a.txt", "alpha beta gamma delta")
    store.add_document("a.txt", "completely different words about oranges")
    assert store.list_sources() == [{"source": "a.txt", "chunks": 1}]
    reloaded = VectorStore(tmp_path)
    assert reloaded.search("oranges")[0]["source"] == "a.txt"


def test_search_empty_store(tmp_path):
    assert VectorStore(tmp_path).search("anything") == []
