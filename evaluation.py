"""
RAG evaluation pipeline using LLM-as-a-Judge.
Pattern: Notebook 08 (Synthetic Q&A generation, pairwise comparison, preference scoring)
"""

import random
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from config import EVAL_NUM_SAMPLES, EVAL_PREFERENCE_THRESHOLD
from models import EvalQAPair


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYNTH_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a test data generator for a contract Q&A system. "
     "Given two document chunks, generate ONE question-answer pair. "
     "The question should require information from at least one chunk. "
     "The answer should be factual and based only on the provided content.\n\n"
     "Output format:\n"
     "Question: <your question>\n"
     "Answer: <your answer>"),
    ("user",
     "Document Chunk 1:\n{chunk1}\n\n"
     "Document Chunk 2:\n{chunk2}"),
])

PAIRWISE_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert evaluator for a contract Q&A system. "
     "Evaluate the following Question-Answer pair for human preference and consistency.\n\n"
     "Assume the FIRST answer is the ground truth and must be correct.\n"
     "Compare the SECOND answer (from the RAG system) against it.\n\n"
     "Scoring:\n"
     "[1] The second answer lies, does not answer the question, or is clearly inferior.\n"
     "[2] The second answer is accurate, relevant, and does not introduce inconsistencies.\n\n"
     "Output Format: [Score] Justification\n"
     "Example: [2] The RAG answer correctly identifies the payment terms..."),
    ("user",
     "Question: {question}\n\n"
     "Answer 1 (Ground Truth): {ground_truth}\n\n"
     "Answer 2 (RAG System): {rag_answer}"),
])


# ─── Synthetic Q&A Generation ───────────────────────────────────────────────

def generate_synthetic_qa(
    docs: list[Document],
    llm,
    n: int = EVAL_NUM_SAMPLES,
) -> list[EvalQAPair]:
    """Generate synthetic Q&A pairs from random document chunk sampling.
    Pattern: Notebook 08 — random.sample(docs, 2) → generate pair.

    Args:
        docs: List of document chunks.
        llm: The ChatNVIDIA LLM instance.
        n: Number of pairs to generate.

    Returns:
        List of EvalQAPair objects with question and ground_truth_answer.
    """
    qa_chain = SYNTH_QA_PROMPT | llm | StrOutputParser()
    pairs = []

    # Ensure we have enough docs to sample from
    if len(docs) < 2:
        print("[Eval] Not enough document chunks for evaluation (need at least 2)")
        return pairs

    for i in range(n):
        try:
            # Random sample 2 chunks
            doc1, doc2 = random.sample(docs, 2)

            result = qa_chain.invoke({
                "chunk1": doc1.page_content[:500],  # Limit chunk size for prompt
                "chunk2": doc2.page_content[:500],
            })

            # Parse Q&A from output
            question, answer = _parse_qa_output(result)
            if question and answer:
                pairs.append(EvalQAPair(
                    question=question,
                    ground_truth_answer=answer,
                ))
                print(f"  [Eval] Generated pair {i+1}/{n}: {question[:60]}...")

        except Exception as e:
            print(f"  [Eval] Failed to generate pair {i+1}: {e}")
            continue

    print(f"[Eval] Generated {len(pairs)}/{n} synthetic Q&A pairs")
    return pairs


def _parse_qa_output(text: str) -> tuple[str, str]:
    """Parse 'Question: ...\nAnswer: ...' format from LLM output."""
    question = ""
    answer = ""

    lines = text.strip().split("\n")
    current = None

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith("question:"):
            current = "q"
            question = line_stripped[len("question:"):].strip()
        elif line_stripped.lower().startswith("answer:"):
            current = "a"
            answer = line_stripped[len("answer:"):].strip()
        elif current == "q":
            question += " " + line_stripped
        elif current == "a":
            answer += " " + line_stripped

    return question.strip(), answer.strip()


# ─── Pairwise Evaluation ────────────────────────────────────────────────────

def run_pairwise_eval(
    qa_pairs: list[EvalQAPair],
    rag_chain,
    llm,
) -> list[EvalQAPair]:
    """Run pairwise evaluation: ground truth vs RAG answer.
    Pattern: Notebook 08 — pairwise string evaluation with [1]/[2] scores.

    For each pair:
    1. Get RAG answer by running the question through the RAG chain
    2. Send both answers to the evaluator LLM
    3. Parse [1] or [2] score with justification
    """
    eval_chain = PAIRWISE_EVAL_PROMPT | llm | StrOutputParser()

    for i, pair in enumerate(qa_pairs):
        try:
            # Get RAG answer
            rag_result = rag_chain.invoke(pair.question)
            if isinstance(rag_result, dict):
                pair.rag_answer = rag_result.get("output", str(rag_result))
            else:
                pair.rag_answer = str(rag_result)

            # Run pairwise evaluation
            eval_result = eval_chain.invoke({
                "question": pair.question,
                "ground_truth": pair.ground_truth_answer,
                "rag_answer": pair.rag_answer,
            })

            # Parse score
            if "[2]" in eval_result:
                pair.eval_score = "[2]"
            elif "[1]" in eval_result:
                pair.eval_score = "[1]"
            else:
                pair.eval_score = "[?]"

            pair.justification = eval_result
            print(f"  [Eval] Pair {i+1}/{len(qa_pairs)}: {pair.eval_score}")

        except Exception as e:
            pair.eval_score = "[error]"
            pair.justification = str(e)
            print(f"  [Eval] Pair {i+1} failed: {e}")

    return qa_pairs


# ─── Scoring ─────────────────────────────────────────────────────────────────

def compute_preference_score(qa_pairs: list[EvalQAPair]) -> float:
    """Compute the preference score (proportion of [2] ratings).
    Pattern: Notebook 08 — sum("[2]" in score) / len(scores).
    """
    scored = [p for p in qa_pairs if p.eval_score in ("[1]", "[2]")]
    if not scored:
        return 0.0
    return sum(1 for p in scored if p.eval_score == "[2]") / len(scored)


def generate_eval_report(qa_pairs: list[EvalQAPair]) -> str:
    """Generate a formatted evaluation report."""
    pref_score = compute_preference_score(qa_pairs)

    report = "=" * 60 + "\n"
    report += "RAG EVALUATION REPORT\n"
    report += "=" * 60 + "\n\n"
    report += f"Total Q&A Pairs: {len(qa_pairs)}\n"
    report += f"Preference Score: {pref_score:.2%}\n"
    report += f"Threshold: {EVAL_PREFERENCE_THRESHOLD:.2%}\n"
    report += f"Result: {'✅ PASS' if pref_score >= EVAL_PREFERENCE_THRESHOLD else '❌ FAIL'}\n"
    report += "\n" + "-" * 60 + "\n"

    for i, pair in enumerate(qa_pairs, 1):
        report += f"\n--- Pair {i} ---\n"
        report += f"Q: {pair.question}\n"
        report += f"Ground Truth: {pair.ground_truth_answer[:200]}...\n"
        report += f"RAG Answer: {(pair.rag_answer or 'N/A')[:200]}...\n"
        report += f"Score: {pair.eval_score}\n"
        report += f"Justification: {(pair.justification or 'N/A')[:200]}\n"

    report += "\n" + "=" * 60 + "\n"
    return report


# ─── Full Evaluation Pipeline ───────────────────────────────────────────────

def run_full_evaluation(
    docstore,
    rag_chain,
    llm,
    n_samples: int = EVAL_NUM_SAMPLES,
) -> dict:
    """Run the complete evaluation pipeline.
    Pattern: Notebook 08 — generate synthetic pairs → pairwise eval → score.

    Returns:
        dict with 'preference_score', 'report', 'qa_pairs', and 'passed'.
    """
    print("[Eval] Starting full evaluation pipeline...")

    # 1. Get document chunks from the store
    retriever = docstore.as_retriever(search_kwargs={"k": 50})
    docs = retriever.invoke("document content")

    if len(docs) < 2:
        print("[Eval] Not enough documents for evaluation")
        return {
            "preference_score": 0.0,
            "report": "Insufficient documents for evaluation.",
            "qa_pairs": [],
            "passed": False,
        }

    # 2. Generate synthetic Q&A pairs
    print(f"[Eval] Generating {n_samples} synthetic Q&A pairs...")
    qa_pairs = generate_synthetic_qa(docs, llm, n=n_samples)

    if not qa_pairs:
        return {
            "preference_score": 0.0,
            "report": "Failed to generate synthetic Q&A pairs.",
            "qa_pairs": [],
            "passed": False,
        }

    # 3. Run pairwise evaluation
    print("[Eval] Running pairwise evaluation...")
    qa_pairs = run_pairwise_eval(qa_pairs, rag_chain, llm)

    # 4. Compute score and generate report
    pref_score = compute_preference_score(qa_pairs)
    report = generate_eval_report(qa_pairs)

    print(report)

    return {
        "preference_score": pref_score,
        "report": report,
        "qa_pairs": qa_pairs,
        "passed": pref_score >= EVAL_PREFERENCE_THRESHOLD,
    }


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run evaluation from the command line."""
    from chains import build_llm, build_rag_chain
    from embedding_utils import get_embedder
    from vectorstore_utils import get_or_create_vectorstore

    print("[Eval] Initializing components...")
    llm = build_llm()
    embedder = get_embedder()
    docstore = get_or_create_vectorstore(embedder, "docstore")
    convstore = get_or_create_vectorstore(embedder, "convstore")

    rag_chain = build_rag_chain(docstore, convstore, llm)

    results = run_full_evaluation(docstore, rag_chain, llm)

    if results["passed"]:
        print(f"\n✅ Evaluation PASSED with score: {results['preference_score']:.2%}")
    else:
        print(f"\n❌ Evaluation FAILED with score: {results['preference_score']:.2%}")
