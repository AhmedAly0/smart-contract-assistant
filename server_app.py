"""
FastAPI + LangServe server application.
Pattern: Notebook 09 — %%writefile server_app.py with FastAPI + add_routes.

Endpoints:
    /basic_chat  — Simple LLM chat (no RAG)
    /retriever   — Document retrieval (returns documents)
    /generator   — Generation from context (returns string)
    /upload      — File upload + ingestion (REST, not LangServe)
    /summarize   — Document summarization (REST)

Run: python server_app.py
"""

import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from config import SERVER_HOST, SERVER_PORT
from chains import (
    build_llm,
    build_basic_chat_chain,
    build_retrieval_chain,
    build_generator_chain,
    build_rag_chain,
    build_summarizer,
)
from embedding_utils import get_embedder, SemanticGuardrail
from vectorstore_utils import (
    get_or_create_vectorstore,
    save_vectorstore,
    add_documents_to_store,
)
from document_utils import ingest_document, save_uploaded_file


# ─── Global State ────────────────────────────────────────────────────────────
# These are initialized at startup and shared across requests.

llm = None
embedder = None
docstore = None
convstore = None
guardrail = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on server startup.
    Pattern: Notebook 09 — set up chains and stores before serving.
    """
    global llm, embedder, docstore, convstore, guardrail

    print("[Server] Initializing components...")

    # 1. Build LLM
    llm = build_llm()
    print(f"[Server] LLM ready: {llm.model}")

    # 2. Build embedder
    embedder = get_embedder()
    print(f"[Server] Embedder ready")

    # 3. Load or create vector stores
    docstore = get_or_create_vectorstore(embedder, name="docstore")
    convstore = get_or_create_vectorstore(embedder, name="convstore")
    print("[Server] Vector stores ready")

    # 4. Initialize guardrail
    guardrail = SemanticGuardrail(embedder)
    print("[Server] Semantic guardrail ready")

    # ─── Register LangServe routes ─────────────────────────────────────
    # Pattern: Notebook 09 — add_routes(app, chain, path="/endpoint")

    # /basic_chat — raw LLM, no retrieval
    basic_chain = build_basic_chat_chain(llm)
    add_routes(app, basic_chain, path="/basic_chat")

    # /generator — takes {input, context}, returns answer string
    generator_chain = build_generator_chain(llm)
    add_routes(app, generator_chain, path="/generator")

    print("[Server] LangServe routes registered: /basic_chat, /generator")
    print(f"[Server] Ready at http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"[Server] API docs at http://{SERVER_HOST}:{SERVER_PORT}/docs")

    yield  # Server runs

    # Shutdown: save vector stores
    print("[Server] Saving vector stores...")
    save_vectorstore(docstore, "docstore")
    save_vectorstore(convstore, "convstore")
    print("[Server] Shutdown complete.")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Contract Assistant",
    version="1.0",
    description=(
        "RAG-powered contract Q&A assistant. "
        "Upload documents and ask questions with source citations. "
        "Built as a capstone project for NVIDIA DLI 'Building RAG Agents using LLMs'."
    ),
    lifespan=lifespan,
)

# Allow Gradio frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints (not LangServe) ─────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF/DOCX/TXT file, ingest it, and add to the vector store.

    This endpoint:
    1. Saves the uploaded file to data/
    2. Extracts text and creates chunks (document_utils.ingest_document)
    3. Adds chunks to the FAISS docstore
    4. Persists the updated store to disk
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Save file
        contents = await file.read()
        filepath = save_uploaded_file(contents, file.filename)

        # Ingest: load → chunk → metadata
        chunks = ingest_document(filepath, doc_name=file.filename)

        # Add to vector store
        global docstore
        docstore = add_documents_to_store(docstore, chunks)

        # Persist
        save_vectorstore(docstore, "docstore")

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": f"Document '{file.filename}' ingested successfully with {len(chunks)} chunks.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask")
async def ask_question(query: str, use_guardrail: bool = True):
    """Ask a question about uploaded documents.

    This endpoint composes retrieval + generation:
    1. Optionally checks the semantic guardrail
    2. Retrieves relevant chunks from docstore + conversation history from convstore
    3. Generates an answer with source citations
    4. Saves the exchange to conversation memory
    """
    global docstore, convstore, guardrail

    # Guardrail check
    if use_guardrail and guardrail:
        guard_result = guardrail.check(query)
        if not guard_result["allowed"]:
            return {
                "answer": (
                    "I'm sorry, but that question doesn't appear to be related to "
                    "the uploaded document(s). Please ask about contract terms, "
                    "clauses, obligations, or other document content."
                ),
                "guardrail_blocked": True,
                "similarity": guard_result["max_similarity"],
            }

    try:
        # Build and run the full RAG chain
        rag_chain = build_rag_chain(docstore, convstore, llm)
        result = rag_chain.invoke(query)

        # Save to conversation memory
        from vectorstore_utils import save_conversation_memory
        save_conversation_memory(convstore, query, result["output"])

        return {
            "answer": result["output"],
            "guardrail_blocked": False,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/summarize")
async def summarize_document(doc_name: Optional[str] = None):
    """Generate a progressive summary of the uploaded document.
    Pattern: Notebook 05 — RSummarizer iterating over chunks.
    """
    global docstore

    try:
        # Retrieve all chunks from the docstore
        # Use a broad query to get relevant document chunks
        query = doc_name or "document summary overview"
        retriever = docstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)

        if not docs:
            return {"summary": "No documents found. Please upload a document first."}

        summarizer = build_summarizer(llm)
        summary = summarizer.invoke(docs)

        return {
            "summary": summary.model_dump(),
            "chunks_processed": len(docs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_model": LLM_MODEL if llm else "not initialized",
        "docstore_ready": docstore is not None,
        "convstore_ready": convstore is not None,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import LLM_MODEL
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
