import importlib
import sys
import types
from pathlib import Path
import math
import pytest


class _Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def _make_fake_faiss(results=None, raise_on_with_scores=False):
    """Return a fake FAISS class with a load_local that returns an object implementing
    similarity_search_with_relevance_scores and similarity_search."""

    class FakeIndex:
        def __init__(self, results, raise_with_scores=False):
            self._results = results or []
            self._raise = raise_with_scores

        def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
            if self._raise:
                raise RuntimeError("no scores available")
            return self._results[:k]

        def similarity_search(self, query, k=4, **kwargs):
            # return docs only
            return [r[0] if isinstance(r, tuple) else r for r in self._results[:k]]

    class FAISS:
        @classmethod
        def load_local(cls, *args, **kwargs):
            return FakeIndex(results, raise_on_with_scores)

    return FAISS


def _inject_stubs(monkeypatch, faiss_cls):
    # stub the modules imported by retrival.py
    langchain_comm = types.ModuleType("langchain_community")
    langchain_comm_vectorstores = types.ModuleType("langchain_community.vectorstores")
    langchain_comm_vectorstores.FAISS = faiss_cls

    langchain_comm_emb = types.ModuleType("langchain_community.embeddings")
    langchain_comm_emb.OllamaEmbeddings = lambda *args, **kwargs: None

    langchain_llms = types.ModuleType("langchain.llms")
    # default stub LLM class (we'll replace retrival.llm explicitly in tests)
    class Ollama:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, prompt):
            return "stub"

    langchain_llms.Ollama = Ollama

    # inject
    monkeypatch.setitem(sys.modules, "langchain_community", langchain_comm)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", langchain_comm_vectorstores)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", langchain_comm_emb)
    monkeypatch.setitem(sys.modules, "langchain.llms", langchain_llms)


def _reload_retrival():
    # Load `retrival.py` directly from file to avoid import path issues in tests
    import importlib.util
    retrival_path = Path(__file__).resolve().parent.parent / "retrival.py"
    spec = importlib.util.spec_from_file_location("retrival", str(retrival_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["retrival"] = module
    spec.loader.exec_module(module)
    return module


def test_retrieve_similar_chunks_normalizes_scores(monkeypatch):
    # Prepare fake results: two docs with distance-like scores (negative or large distances)
    docs = [(_Doc("Doc A"), -10.0), (_Doc("Doc B"), -20.0)]
    FAISS = _make_fake_faiss(results=docs, raise_on_with_scores=False)
    _inject_stubs(monkeypatch, FAISS)

    retrival = _reload_retrival()

    chunks = retrival.retrieve_similar_chunks("query")
    assert len(chunks) == 2
    # scores should be normalized in (0,1)
    assert 0 < chunks[0]["score"] < 1
    # With distance-like raw scores [-10, -20], scale = max_abs = 20 -> normalized = exp(-abs(s)/20)
    import math
    expected = math.exp(-abs(-10.0) / 20.0)
    assert pytest.approx(chunks[0]["score"], rel=1e-3) == expected
    assert chunks[0]["content"] == "Doc A"


def test_retrieve_fallback_to_similarity_search(monkeypatch):
    # Make FAISS that raises on with_relevance_scores, but returns docs on similarity_search
    docs_only = [_Doc("Only A"), _Doc("Only B")]
    FAISS = _make_fake_faiss(results=[(docs_only[0], None), (docs_only[1], None)], raise_on_with_scores=True)
    _inject_stubs(monkeypatch, FAISS)

    retrival = _reload_retrival()

    chunks = retrival.retrieve_similar_chunks("q")
    assert len(chunks) == 2
    # Since fallback provides docs with no scores, retrival should set score to None
    assert all(c["score"] is None for c in chunks)


def test_generate_answer_builds_prompt_and_confidence_high_and_low(monkeypatch, capsys):
    # Simple import with default FAISS returning nothing (we won't use retrieve in this test)
    FAISS = _make_fake_faiss(results=[])
    _inject_stubs(monkeypatch, FAISS)
    retrival = _reload_retrival()

    # Capture prompt passed to LLM
    captured = {}

    def llm_stub(prompt):
        captured["prompt"] = prompt
        return "LLM_ANSWER"

    retrival.llm = llm_stub

    # High confidence: single strong-scoring chunk
    chunks_high = [{"content": "Strong evidence", "score": 100.0}]
    out = retrival.generate_answer(retrival.getFinalPrompt("q"), chunks_high)
    captured_print = capsys.readouterr().out
    assert "Confidence Level: High" in captured_print
    assert out == "LLM_ANSWER"
    # ensure chunk label is present in the prompt
    assert "[chunk-1]" in captured["prompt"]
    assert "Strong evidence" in captured["prompt"]

    # Low confidence: three low equal scores -> softmax yields low top probability
    captured.clear()
    chunks_low = [{"content": "A", "score": 0.1}, {"content": "B", "score": 0.1}, {"content": "C", "score": 0.1}]
    out2 = retrival.generate_answer(retrival.getFinalPrompt("q"), chunks_low)
    printed = capsys.readouterr().out
    # With three equal low scores, dominance can push this into Medium
    assert "Confidence Level: Medium" in printed
    assert "Disclaimer: The confidence level of this answer is low" not in out2
    # prompt still contains chunk labels
    assert "[chunk-1]" in captured["prompt"] and "[chunk-2]" in captured["prompt"] and "[chunk-3]" in captured["prompt"]
