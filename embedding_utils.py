"""
Embedding model setup and semantic guardrails.
Pattern: Notebook 06 (NVIDIAEmbeddings with query/document pathways)
Pattern: Notebook 06 (Semantic guardrails — cosine similarity filtering)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    EMBEDDING_MODEL,
    GUARDRAIL_THRESHOLD,
    SAFE_TOPICS,
)


# ─── Embedder Factory ───────────────────────────────────────────────────────

def get_embedder() -> NVIDIAEmbeddings:
    """Create and return an NVIDIAEmbeddings instance.
    Pattern: Notebook 06 — nvidia/llama-3.2-nv-embedqa-1b-v2 with dual pathways.

    The embedder supports two modes:
    - embed_query(): for short-form queries (questions)
    - embed_documents(): for long-form document passages (batch)
    """
    embedder = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        nvidia_api_key=NVIDIA_API_KEY,
        base_url=f"{NVIDIA_BASE_URL}",
    )
    return embedder


def get_embedding_dims(embedder: NVIDIAEmbeddings) -> int:
    """Get the embedding dimensionality for FAISS index initialization.
    Pattern: Notebook 07 — needed for IndexFlatL2 initialization.
    """
    test_embedding = embedder.embed_query("dimension test")
    return len(test_embedding)


# ─── Semantic Guardrails ─────────────────────────────────────────────────────

class SemanticGuardrail:
    """Embedding-based guardrail that filters queries by semantic similarity.
    Pattern: Notebook 06 — using cosine similarity to block off-topic/harmful inputs.

    The guardrail computes cosine similarity between the incoming query and
    a set of pre-defined safe topics. If the maximum similarity is below
    the threshold, the query is flagged as off-topic.
    """

    def __init__(
        self,
        embedder: NVIDIAEmbeddings,
        safe_topics: list[str] = SAFE_TOPICS,
        threshold: float = GUARDRAIL_THRESHOLD,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.safe_topics = safe_topics
        # Pre-compute safe topic embeddings
        self._safe_embeddings = None

    def _ensure_safe_embeddings(self):
        """Lazy-load safe topic embeddings on first use."""
        if self._safe_embeddings is None:
            self._safe_embeddings = np.array(
                self.embedder.embed_documents(self.safe_topics)
            )

    def check(self, query: str) -> dict:
        """Check if a query is semantically on-topic.

        Returns:
            dict with keys:
                - 'allowed' (bool): True if query passes the guardrail
                - 'max_similarity' (float): highest similarity to any safe topic
                - 'closest_topic' (str): the most similar safe topic
        """
        self._ensure_safe_embeddings()

        query_embedding = np.array(self.embedder.embed_query(query)).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self._safe_embeddings)[0]

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        return {
            "allowed": max_sim >= self.threshold,
            "max_similarity": max_sim,
            "closest_topic": self.safe_topics[max_idx],
        }

    def __call__(self, query: str) -> str:
        """Callable interface for use in LCEL chains.
        Returns the query if allowed, raises ValueError if blocked.
        """
        result = self.check(query)
        if not result["allowed"]:
            return (
                "I'm sorry, but that question doesn't appear to be related to "
                "the uploaded document or contract analysis. Please ask a question "
                "about the document's content, terms, clauses, or obligations."
            )
        return query
