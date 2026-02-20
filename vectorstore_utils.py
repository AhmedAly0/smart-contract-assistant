"""
FAISS vector store management: creation, persistence, and conversation memory.
Pattern: Notebook 07 (default_FAISS, save/load, aggregate, conversation memory)
"""

import os
from typing import Optional

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.schema import Document

from config import VECTORSTORE_DIR
from embedding_utils import get_embedding_dims


# ─── FAISS Factory ───────────────────────────────────────────────────────────

def default_FAISS(embedder: NVIDIAEmbeddings) -> FAISS:
    """Create an empty FAISS vector store.
    Pattern: Notebook 07 — IndexFlatL2 + InMemoryDocstore factory.

    This is used when no pre-existing index exists, or when we need a
    fresh store to merge into.
    """
    embed_dims = get_embedding_dims(embedder)
    return FAISS(
        embedding_function=embedder,
        index=faiss.IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False,
    )


# ─── Build from Documents ───────────────────────────────────────────────────

def build_vectorstore(
    documents: list[Document],
    embedder: NVIDIAEmbeddings
) -> FAISS:
    """Build a FAISS vector store from a list of LangChain Documents.
    Pattern: Notebook 07 — FAISS.from_documents().
    """
    if not documents:
        return default_FAISS(embedder)
    return FAISS.from_documents(documents, embedding=embedder)


def add_documents_to_store(
    vstore: FAISS,
    documents: list[Document]
) -> FAISS:
    """Add new documents to an existing vector store."""
    if documents:
        vstore.add_documents(documents)
    return vstore


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_vectorstore(
    vstore: FAISS,
    name: str = "docstore",
    directory: str = VECTORSTORE_DIR,
) -> str:
    """Save a FAISS vector store to disk.
    Pattern: Notebook 07 — vstore.save_local().

    Returns the full path to the saved index.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    vstore.save_local(path)
    print(f"[VectorStore] Saved to: {path}")
    return path


def load_vectorstore(
    embedder: NVIDIAEmbeddings,
    name: str = "docstore",
    directory: str = VECTORSTORE_DIR,
) -> Optional[FAISS]:
    """Load a FAISS vector store from disk.
    Pattern: Notebook 07 — FAISS.load_local() with allow_dangerous_deserialization.

    Returns None if the index doesn't exist.
    """
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        print(f"[VectorStore] No index found at: {path}")
        return None
    vstore = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    print(f"[VectorStore] Loaded from: {path}")
    return vstore


def get_or_create_vectorstore(
    embedder: NVIDIAEmbeddings,
    name: str = "docstore",
) -> FAISS:
    """Load an existing vector store or create an empty one.
    Convenience wrapper combining load + fallback creation.
    """
    vstore = load_vectorstore(embedder, name)
    if vstore is None:
        vstore = default_FAISS(embedder)
        print(f"[VectorStore] Created new empty store: {name}")
    return vstore


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_vstores(
    vectorstores: list[FAISS],
    embedder: NVIDIAEmbeddings,
) -> FAISS:
    """Merge multiple FAISS vector stores into one.
    Pattern: Notebook 07 — aggregate_vstores using merge_from.
    """
    agg_vstore = default_FAISS(embedder)
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore


# ─── Conversation Memory ────────────────────────────────────────────────────

def save_conversation_memory(
    convstore: FAISS,
    user_input: str,
    agent_output: str,
) -> None:
    """Save a conversation turn to the conversation vector store.
    Pattern: Notebook 07 — save_memory_and_get_output.

    Stores both sides of the exchange for history-aware retrieval.
    """
    convstore.add_texts([
        f"User said: {user_input}",
        f"Assistant said: {agent_output}",
    ])


def save_memory_and_get_output(d: dict, vstore: FAISS) -> str:
    """Save conversation to memory and return the output.
    Pattern: Notebook 07 — used inline in the RAG chain.

    Expects dict with 'input' and 'output' keys.
    """
    vstore.add_texts([
        f"User said: {d['input']}",
        f"Assistant said: {d['output']}",
    ])
    return d["output"]
