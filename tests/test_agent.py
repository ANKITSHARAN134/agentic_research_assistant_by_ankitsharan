from types import SimpleNamespace as NS

from agent.agent import ResearchAgent
from agent.rag import VectorStore
from agent.tools import Toolbox, is_safe_url


class FakeClient:
    """Scripted stand-in for the Anthropic client."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def text(t):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=t)])


def tool_call(name, args, id="tu_1"):
    return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=id, name=name, input=args)])


def make_agent(tmp_path, responses):
    store = VectorStore(tmp_path)
    store.add_document("notes.txt", "The CDAC course covers Java, data structures and databases.")
    return ResearchAgent(store, client=FakeClient(responses))


def test_agent_uses_document_tool_then_answers(tmp_path):
    agent = make_agent(
        tmp_path,
        [tool_call("search_documents", {"query": "CDAC course"}), text("It covers Java.")],
    )
    result = agent.chat("s1", "What does the CDAC course cover?")
    assert result["answer"] == "It covers Java."
    assert result["steps"] == [{"tool": "search_documents", "input": {"query": "CDAC course"}}]
    assert result["sources"] == [{"type": "document", "ref": "notes.txt"}]
    # tool result was fed back to the model
    tool_result = agent.client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "notes.txt" in tool_result["content"]


def test_memory_carries_across_turns(tmp_path):
    agent = make_agent(tmp_path, [text("First."), text("Second.")])
    agent.chat("s1", "hello")
    agent.chat("s1", "and again")
    roles = [m["role"] for m in agent.client.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    agent.reset("s1")
    assert "s1" not in agent.sessions


def test_step_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.config.MAX_STEPS", 2)
    agent = make_agent(tmp_path, [tool_call("search_documents", {"query": "x"})] * 2)
    assert "step limit" in agent.chat("s1", "loop")["answer"]


def test_unknown_tool_and_tool_errors_are_reported_not_raised(tmp_path):
    box = Toolbox(VectorStore(tmp_path))
    assert "Unknown tool" in box.run("nope", {})
    assert "error" in box.run("read_webpage", {"url": "http://127.0.0.1/admin"})


def test_ssrf_guard():
    assert not is_safe_url("http://127.0.0.1:8000")
    assert not is_safe_url("http://localhost/")
    assert not is_safe_url("file:///etc/passwd")
    assert not is_safe_url("ftp://example.com")


def test_read_webpage_html_title(monkeypatch):
    from agent.tools import read_webpage
    monkeypatch.setattr("agent.tools.is_safe_url", lambda url: True)

    class FakeResponse:
        text = "<html><head><title>Page <b>Title</b></title></head><body><p>Content text</p></body></html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("agent.tools.requests.get", lambda *args, **kwargs: FakeResponse())
    result = read_webpage("https://example.com/test")
    assert result["title"] == "Page Title"
    assert "Content text" in result["text"]
