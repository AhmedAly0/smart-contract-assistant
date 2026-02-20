"""
Dialog state management with structured extraction and conditional routing.
Pattern: Notebook 04 (RExtract — PydanticOutputParser + LLM for slot-filling)
Pattern: Notebook 04 (RunnableBranch for conditional routing)
Pattern: Notebook 04 (Running state chain with RunnableAssign)
"""

import json
import re

from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models import KnowledgeBase


# ─── RExtract: Structured Extraction via LLM ────────────────────────────────

def RExtract(pydantic_class, llm, prompt):
    """Create a Runnable that extracts structured data from text using an LLM.
    Pattern: Notebook 04 — PydanticOutputParser + format instructions + LLM + parse.

    This is the core slot-filling mechanism used throughout the course:
    1. Generate format_instructions from the Pydantic model
    2. Inject them into the prompt
    3. Send to LLM
    4. Parse the output back into a Pydantic instance

    Args:
        pydantic_class: The Pydantic BaseModel class to extract.
        llm: The ChatNVIDIA LLM instance.
        prompt: ChatPromptTemplate with {format_instructions} placeholder.

    Returns:
        A Runnable chain that outputs a Pydantic model instance.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_class)

    # Inject format_instructions into the state dictionary
    instruct_merge = RunnableAssign({
        "format_instructions": lambda x: parser.get_format_instructions()
    })

    def preparse(string):
        """Clean up LLM output before JSON parsing.
        Pattern: Notebook 04 — handle markdown code fences and trailing content.
        """
        if isinstance(string, str):
            # Remove markdown code fences
            string = re.sub(r"```json\s*", "", string)
            string = re.sub(r"```\s*", "", string)
            # Try to extract JSON object
            match = re.search(r"\{.*\}", string, re.DOTALL)
            if match:
                string = match.group()
        return string

    return instruct_merge | prompt | llm | preparse | parser


# ─── Knowledge Base Update Chain ────────────────────────────────────────────

KB_UPDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are tracking conversation context for a contract Q&A assistant. "
     "Update the knowledge base with any new information from the latest exchange. "
     "Keep the discussion summary concise but comprehensive. "
     "Only update fields where you have new information; keep existing values "
     "for fields with no updates.\n\n"
     "Current Knowledge Base:\n{know_base}\n\n"
     "{format_instructions}"),
    ("user",
     "Latest message: {input}\n"
     "Latest response: {output}"),
])


def build_kb_update_chain(llm):
    """Build a chain that updates the KnowledgeBase after each exchange.
    Pattern: Notebook 04 — RunnableAssign({'know_base': RExtract(...)}).

    This chain takes a state dict with 'input', 'output', and 'know_base'
    keys and returns the state with an updated 'know_base'.
    """
    extractor = RExtract(KnowledgeBase, llm, KB_UPDATE_PROMPT)
    return RunnableAssign({"know_base": extractor})


# ─── Intent Routing ──────────────────────────────────────────────────────────

def detect_intent(state: dict) -> str:
    """Simple rule-based intent detection for routing.
    Checks the input text for keywords to determine the intent.
    """
    text = state.get("input", "").lower()

    if any(word in text for word in ["summarize", "summary", "overview", "tldr", "tl;dr"]):
        return "summarize"
    elif any(word in text for word in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    else:
        return "question"


def build_routing_chain(
    question_chain,
    summary_chain,
    greeting_chain,
):
    """Build a conditional routing chain based on user intent.
    Pattern: Notebook 04 — RunnableBranch for switch/case on chain state.

    Routes to different chains based on detected intent:
    - "summarize" → summary_chain
    - "greeting" → greeting_chain
    - default → question_chain (RAG Q&A)
    """
    return RunnableBranch(
        (lambda x: detect_intent(x) == "summarize", summary_chain),
        (lambda x: detect_intent(x) == "greeting", greeting_chain),
        question_chain,  # default
    )
