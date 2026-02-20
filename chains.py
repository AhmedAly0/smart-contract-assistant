"""
LCEL chain construction for the RAG pipeline.
"""

from operator import itemgetter

from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS

from config import GOOGLE_API_KEY, LLM_MODEL
from utils import docs2str


# ─── LLM Factory ─────────────────────────────────────────────────────────────

def build_llm(model: str = LLM_MODEL, **kwargs) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI LLM instance.
    Pattern: Notebook 02 — ChatGoogleGenerativeAI as highest-abstraction LLM connector.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        **kwargs,
    )


# ─── Prompt Templates ───────────────────────────────────────────────────────

RAG_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Smart Contract Q&A Assistant. "
     "Use ONLY the provided context to answer questions. "
     "Always cite the source document when providing information. "
     "If the context does not contain the answer, say: "
     "'I cannot find this information in the uploaded document(s).'\n\n"
     "Conversation History:\n{history}\n\n"
     "Document Context:\n{context}"),
    ("user", "{input}"),
])

SUMMARIZE_CHUNK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document summarization expert. "
     "You are progressively summarizing a document chunk by chunk. "
     "Update the running summary with new information from the latest chunk. "
     "Maintain the most important points and track open questions.\n\n"
     "Current Summary State:\n{info_base}\n\n"
     "{format_instructions}"),
    ("user", "New chunk to incorporate:\n{input}"),
])


# ─── Retrieval Chain ─────────────────────────────────────────────────────────

def build_retrieval_chain(
    docstore: FAISS,
    convstore: FAISS,
):
    """Build the retrieval chain that fetches context and history.
    Pattern: Notebook 07 — RunnableAssign with retriever + LongContextReorder + docs2str.

    Returns a chain that takes a string input and outputs a dict with:
    - input: original query
    - history: relevant conversation history from convstore
    - context: relevant document chunks from docstore
    """
    long_reorder = LongContextReorder()

    retrieval_chain = (
        {"input": RunnableLambda(lambda x: x)}
        | RunnableAssign({
            "history": (
                itemgetter("input")
                | convstore.as_retriever()
                | long_reorder.transform_documents
                | docs2str
            )
        })
        | RunnableAssign({
            "context": (
                itemgetter("input")
                | docstore.as_retriever()
                | long_reorder.transform_documents
                | docs2str
            )
        })
    )
    return retrieval_chain


# ─── Full RAG Chain ──────────────────────────────────────────────────────────

def build_rag_chain(
    docstore: FAISS,
    convstore: FAISS,
    llm=None,
):
    """Build the complete RAG chain: retrieve → generate.
    Pattern: Notebook 07/09 — composition of retrieval_chain | generator_chain.

    Takes a string input and returns a dict with 'output' added.
    """
    if llm is None:
        llm = build_llm()

    retrieval_chain = build_retrieval_chain(docstore, convstore)
    generator_chain = RAG_CONTEXT_PROMPT | llm | StrOutputParser()

    rag_chain = (
        retrieval_chain
        | RunnableAssign({"output": generator_chain})
    )
    return rag_chain


# ─── Summarization Chain ────────────────────────────────────────────────────

def build_summarizer(llm=None):
    """Build a progressive document summarizer.
    Pattern: Notebook 05 — RSummarizer iterating over chunks with RExtract.

    This returns a function that takes a list of Documents and returns
    a DocumentSummaryBase with the progressive summary.
    """
    if llm is None:
        llm = build_llm()

    # Import here to avoid circular dependency
    from state_management import RExtract
    from models import DocumentSummaryBase

    def summarize_docs(docs):
        """Iterate over document chunks, progressively building a summary."""
        parse_chain = RunnableAssign({
            "info_base": RExtract(DocumentSummaryBase, llm, SUMMARIZE_CHUNK_PROMPT)
        })

        state = {"info_base": DocumentSummaryBase(), "input": ""}

        for i, doc in enumerate(docs):
            content = getattr(doc, "page_content", str(doc))
            state["input"] = content
            try:
                state = parse_chain.invoke(state)
                print(f"  [Summarizer] Processed chunk {i+1}/{len(docs)}")
            except Exception as e:
                print(f"  [Summarizer] Skipping chunk {i+1}: {e}")
                continue

        return state["info_base"]

    return RunnableLambda(summarize_docs)
