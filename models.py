"""
Pydantic models for structured data throughout the application.
Pattern: Notebook 04 (KnowledgeBase with Field descriptors)
Pattern: Notebook 05 (DocumentSummaryBase for progressive summarization)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeBase(BaseModel):
    """Running knowledge base for conversation state tracking.
    Pattern: Notebook 04 — slot-filling extraction via RExtract.
    """
    user_name: str = Field("unknown", description="The user's name if provided")
    document_name: str = Field("unknown", description="Name of the uploaded document being discussed")
    discussion_summary: str = Field(
        "",
        description="Running summary of what has been discussed so far"
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics/clauses that have been discussed (max 5)"
    )
    response: str = Field(
        "",
        description="The ideal response to the user's latest message"
    )


class DocumentSummaryBase(BaseModel):
    """Progressive document summarization model.
    Pattern: Notebook 05 — RSummarizer iterates over chunks accumulating state.
    """
    running_summary: str = Field(
        "",
        description="Running description of the document so far, updated with each chunk"
    )
    main_ideas: List[str] = Field(
        default_factory=list,
        description="Most important points from the document (max 5)"
    )
    loose_ends: List[str] = Field(
        default_factory=list,
        description="Open questions or unresolved points (max 3)"
    )
    document_type: str = Field(
        "unknown",
        description="Type of document (contract, insurance policy, report, etc.)"
    )
    parties_involved: List[str] = Field(
        default_factory=list,
        description="Names of parties mentioned in the document"
    )


class ConversationState(BaseModel):
    """Full conversation state dictionary flowing through the chain.
    Pattern: Notebook 04 — RunnableAssign accumulates keys in running state.
    """
    input: str = ""
    context: str = ""
    history: str = ""
    output: str = ""
    know_base: KnowledgeBase = Field(default_factory=KnowledgeBase)


class EvalQAPair(BaseModel):
    """A synthetic Q&A pair for RAG evaluation.
    Pattern: Notebook 08 — synthetic Q&A generation from random chunks.
    """
    question: str = Field(..., description="A question derived from document chunks")
    ground_truth_answer: str = Field(..., description="The ground truth answer from the document")
    rag_answer: Optional[str] = Field(None, description="The RAG pipeline's answer")
    eval_score: Optional[str] = Field(None, description="[1] or [2] pairwise evaluation score")
    justification: Optional[str] = Field(None, description="Evaluator justification")
