"""
Pydantic models for structured data.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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


class EvalQAPair(BaseModel):
    """A synthetic Q&A pair for RAG evaluation.
    Pattern: Notebook 08 — synthetic Q&A generation from random chunks.
    """
    question: str = Field(..., description="A question derived from document chunks")
    ground_truth_answer: str = Field(..., description="The ground truth answer from the document")
    rag_answer: Optional[str] = Field(None, description="The RAG pipeline's answer")
    eval_score: Optional[str] = Field(None, description="[1] or [2] pairwise evaluation score")
    justification: Optional[str] = Field(None, description="Evaluator justification")
