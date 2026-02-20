"""
Utility helpers.
"""


def docs2str(docs, title="Document"):
    """Convert a list of LangChain Document objects to a formatted string with citations.
    Pattern: Notebook 07 — docs2str used in RAG chain for context injection.

    Each chunk is prefixed with [Quote from <source>] for citation tracking.
    """
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, "metadata", {}).get("Title", "")
        if not doc_name:
            doc_name = getattr(doc, "metadata", {}).get("source", title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, "page_content", str(doc)) + "\n"
    return out_str



