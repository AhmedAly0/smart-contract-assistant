# Smart Contract Summary & Q&A Assistant

A **RAG (Retrieval-Augmented Generation)** application for analyzing smart contracts and legal documents, built as a capstone project for the **NVIDIA DLI — Building RAG Agents using LLMs** course.

## Architecture

```
User ─► Ingestion Pipeline ─► Vector Store
                                   │
User ─► Gradio UI ─► FastAPI/LangServe ─► Retriever ─► LLM Pipeline ─► Response
```

### Core Modules

| File | Responsibility | Course Notebook |
|---|---|---|
| `config.py` | Centralised settings & environment | N09 (`%%writefile`) |
| `models.py` | Pydantic data models | N04 (Running State) |
| `utils.py` | `RPrint`, `docs2str`, helpers | N04, N07 |
| `document_utils.py` | PDF/DOCX loading & chunking | N05 (Documents) |
| `embedding_utils.py` | Embedder + `SemanticGuardrail` | N06 (Embeddings) |
| `vectorstore_utils.py` | FAISS build / save / load / merge | N07 (VectorStores) |
| `chains.py` | LCEL chains (chat, RAG, summarizer) | N03, N04, N05, N07 |
| `state_management.py` | `RExtract`, intent routing | N04 (Running State) |
| `server_app.py` | FastAPI + LangServe endpoints | N09 (LangServe) |
| `frontend_app.py` | Gradio chat / upload / summary UI | N09 |
| `evaluation.py` | LLM-as-a-Judge pipeline | N08 (Evaluation) |

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-user>/smart-contract-assistant.git
cd smart-contract-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Start the backend

```bash
python server_app.py
```

The FastAPI server starts at `http://0.0.0.0:9012`.

Available routes:
- `POST /upload` — Upload & ingest a document (PDF / DOCX)
- `POST /ask` — Ask a question with RAG + guardrails
- `POST /summarize` — Progressive document summarization
- `GET  /health` — Health check
- `/basic_chat/` — LangServe basic LLM chat
- `/generator/` — LangServe generator chain

### 4. Start the frontend

```bash
python frontend_app.py
```

Gradio UI launches at `http://0.0.0.0:8090` with four tabs:
1. **Upload Document** — Ingest PDF/DOCX into the vector store
2. **Chat with Document** — RAG-powered Q&A with citations
3. **Document Summary** — Progressive summarization (running summary → main ideas → loose ends)
4. **Direct LLM Chat** — Baseline chat without retrieval

### 5. Run evaluation (optional)

```bash
python evaluation.py
```

Generates synthetic Q&A pairs, runs pairwise comparison (LLM-as-a-Judge), and reports the preference score.

## Key Patterns from the Course

- **LCEL Pipe Operator** — All chains built with `|` composition (Notebook 03)
- **RunnableAssign / Running State** — Parallel context injection (Notebook 04)
- **RExtract** — Pydantic slot-filling extraction (Notebook 04)
- **RecursiveCharacterTextSplitter** — Chunk with separators (Notebook 05)
- **Progressive Summarization** — `RSummarizer` for large documents (Notebook 05)
- **SemanticGuardrail** — Cosine-similarity topic gating (Notebook 06)
- **FAISS + LongContextReorder** — Vector search with relevance reordering (Notebook 07)
- **LLM-as-a-Judge** — Synthetic eval with pairwise comparison (Notebook 08)
- **LangServe + FastAPI** — Production serving (Notebook 09)

## Tech Stack

- **LLM**: `gemini-2.5-flash` via Google Gemini API
- **Embeddings**: `models/text-embedding-004` (Google)
- **Vector Store**: FAISS (`IndexFlatL2` + `InMemoryDocstore`)
- **Framework**: LangChain LCEL, LangServe, FastAPI, Gradio
- **Document Processing**: PyMuPDF, python-docx

## License

This project is part of the NVIDIA DLI course curriculum and is for educational purposes.
