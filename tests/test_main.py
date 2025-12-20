import runpy
import sys
import types
import os
from pathlib import Path


def test_main_runs_and_saves_vector_store(tmp_path, monkeypatch, capsys):
    # Create a temporary Book.txt in the test working directory
    book = tmp_path / "Book.txt"
    book.write_text("This is a test book. " * 200, encoding="utf-8")

    # --- Stub implementations for langchain modules ---
    class TextLoader:
        def __init__(self, file_path, encoding=None, autodetect_encoding=False):
            self.file_path = file_path
            self.encoding = encoding
            self.autodetect_encoding = autodetect_encoding

        def load(self):
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()

            class Doc:
                def __init__(self, text):
                    self.page_content = text

            return [Doc(text)]

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            full_text = "".join(getattr(d, "page_content", str(d)) for d in documents)
            chunks = [full_text[i : i + self.chunk_size] for i in range(0, len(full_text), self.chunk_size)]
            return chunks

    class OllamaEmbeddings:
        def __init__(self, model=None):
            self.model = model

    class FAISS:
        @classmethod
        def from_documents(cls, docs, embeddings):
            inst = cls()
            inst.docs = docs
            inst.embeddings = embeddings
            return inst

        def save_local(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "saved.txt"), "w", encoding="utf-8") as f:
                f.write(f"saved {len(self.docs)} docs")

    # --- Inject stub modules into sys.modules ---
    # Parent packages
    langchain_community_mod = types.ModuleType("langchain_community")
    langchain_mod = types.ModuleType("langchain")

    # Submodules
    doc_loaders_mod = types.ModuleType("langchain_community.document_loaders")
    doc_loaders_mod.TextLoader = TextLoader

    splitters_mod = types.ModuleType("langchain_text_splitters")
    splitters_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

    emb_mod = types.ModuleType("langchain.embeddings")
    emb_mod.OllamaEmbeddings = OllamaEmbeddings

    vec_mod = types.ModuleType("langchain.vectorstores")
    vec_mod.FAISS = FAISS

    # Use monkeypatch to ensure modules are restored after the test
    monkeypatch.setitem(sys.modules, "langchain_community", langchain_community_mod)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", doc_loaders_mod)

    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitters_mod)

    monkeypatch.setitem(sys.modules, "langchain", langchain_mod)
    monkeypatch.setitem(sys.modules, "langchain.embeddings", emb_mod)
    monkeypatch.setitem(sys.modules, "langchain.vectorstores", vec_mod)

    # Run `main.py` with cwd set to tmp_path so it picks up the temporary Book.txt
    monkeypatch.chdir(tmp_path)

    # Ensure `main` module will be executed fresh
    sys.modules.pop("main", None)

    main_path = Path(__file__).resolve().parent.parent / "main.py"
    assert main_path.exists(), f"main.py not found at {main_path}"

    # Execute the script
    runpy.run_path(str(main_path), run_name="__main__")

    # Capture stdout and assert expected output
    out = capsys.readouterr().out
    assert "Number of chunks:" in out
    assert "Vector store saved" in out

    # Check that the vector store marker file was created
    vector_dir = tmp_path / "vector_db"
    assert (vector_dir / "saved.txt").exists(), "Vector store was not saved to the expected path"
