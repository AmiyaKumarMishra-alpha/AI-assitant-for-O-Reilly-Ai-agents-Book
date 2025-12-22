# RAG Practice – Agentic Retrieval and Vector Store Demo

A demo application that loads a book text file, splits it into chunks, creates embeddings, builds a FAISS vector store, and implements a **multi-agent Retrieval Augmented Generation (RAG)** pipeline using **local LLMs (Ollama)**.

The project evolves from a basic retrieval + LLM flow into an **agentic system** where query analysis, retrieval, answer generation, and evaluation are coordinated by a master agent.

No external APIs are used. The system runs fully offline.

---

## Project Structure

```
rag PRACTICE/
├── Agentic_RAG/
│   ├── __init__.py
│   ├── main.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_analyzer.py
│   │   ├── retrieval_agent.py
│   │   ├── master_agent.py
│   │   └── test/
│   │       ├── test_master.py
│   │       └── test_retrieval.py
│   ├── tools/
│   │   └── retrieval.py
│   └── experimental/
│       └── mcp/
│           ├── mcp_client.py
│           ├── mcp_server.py
│           └── mcp_test.py
├── tests/
│   ├── test_main.py
│   └── test_retrieval.py
├── build_vector_db.py
├── Book.txt
├── vectordb/
├── requirements.txt
├── .gitignore
├── .gitlab-ci.yml
└── README.md
```

---

## Architecture Overview

High-level execution flow:

```
User Query
   ↓
Query Analyzer Agent
   ↓
Retrieval Agent
   ├── FAISS semantic search
   ├── Context aggregation
   └── Local LLM (Ollama)
   ↓
Master Agent
   ├── Latency evaluation
   ├── Retrieval quality evaluation
   └── ACCEPT / REVIEW decision
```

The **Retrieval Agent** performs both retrieval and answer generation.
The **Master Agent** coordinates execution and evaluates the output.

---

## Requirements

* Python 3.10 or newer
* Use a virtual environment for local development
* Ollama installed locally

Recommended Ollama models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## Quick Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

---

## Step 1: Build the Vector Store

The vector store is built from `Book.txt`.

```bash
python build_vector_db.py
```

This script:

* Loads the book text
* Splits it into chunks
* Generates embeddings
* Stores a FAISS index in `vectordb/`

---

## Step 2: Run Retrieval and Agentic RAG

Run the agentic system using module execution:

```bash
python -m Agentic_RAG.main
```

The output includes:

* Generated answer
* Retrieval scores
* Latency breakdown
* Final decision (ACCEPT or REVIEW)

---

## Testing

Run unit tests:

```bash
pytest -q
```

Run tests with coverage and generate CI reports:

```bash
pytest -q \
  --junitxml=tests/junit.xml \
  --cov=. \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml
```

---

## Continuous Integration

A GitLab CI pipeline is provided in `.gitlab-ci.yml`.

The pipeline:

* Installs dependencies from `requirements.txt`
* Runs unit tests
* Generates coverage reports
* Stores JUnit and coverage artifacts

---

## Notes

* LangChain deprecation warnings may appear for certain classes

  * Prefer `langchain_community` and `langchain-ollama` imports where applicable
* The vector database directory (`vectordb/`) is ignored via `.gitignore`
* MCP (Model Context Protocol) experiments are archived under `Agentic_RAG/experimental/`

  * MCP is intentionally not part of the active execution path due to SDK instability
  * The codebase is structured so MCP can be reintroduced later without refactoring

---

## Design Rationale

This project is intended to demonstrate:

* Retrieval-Augmented Generation using FAISS
* Local LLM integration without external APIs
* Multi-agent system design
* Tool-based retrieval abstraction
* Evaluation-driven acceptance of LLM output
* Production-aligned project structure

The focus is on clarity, extensibility, and correctness rather than minimal examples.

---

## Contributing

* Open an issue or submit a pull request
* Please include tests for any new functionality
* Keep experimental work isolated under `experimental/`

---

## License

This repository is provided for learning, experimentation, and portfolio demonstration purposes.

---

