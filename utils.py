"""
Reusable utility runnables and helper functions.
Pattern: Notebook 03 (RPrint, RInput — debug/utility runnables)
Pattern: Notebook 07 (docs2str — document-to-string formatter with citations)
"""

from functools import partial
from langchain.schema.runnable import RunnableLambda


# ─── Debug Utility Runnables (Notebook 03 pattern) ───────────────────────────

def _print_and_return(x, preface=""):
    """Print the value and return it unchanged — useful for chain debugging."""
    if preface:
        print(f"{preface}: {x}")
    else:
        print(x)
    return x


def RPrint(preface=""):
    """Runnable that prints the current chain value then passes it through.
    Pattern: Notebook 03 — RPrint for debugging chain intermediate values.

    Usage in chain:
        chain = step1 | RPrint("After step1") | step2
    """
    return RunnableLambda(partial(_print_and_return, preface=preface))


def _make_dictionary(x, key="input"):
    """Wrap a value into a dictionary with the given key."""
    return {key: x}


def RInput(key="input"):
    """Runnable that wraps its input into a dictionary.
    Pattern: Notebook 03 — RInput for creating initial state dictionaries.

    Usage: RInput("query") converts "hello" → {"query": "hello"}
    """
    return RunnableLambda(partial(_make_dictionary, key=key))


# ─── Document Formatting (Notebook 07 pattern) ──────────────────────────────

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


# ─── General Helpers ─────────────────────────────────────────────────────────

def format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format Gradio chat history into a string for context injection."""
    formatted = ""
    for user_msg, bot_msg in history:
        formatted += f"User: {user_msg}\n"
        if bot_msg:
            formatted += f"Assistant: {bot_msg}\n"
    return formatted
