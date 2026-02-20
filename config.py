"""
Configuration module for Smart Contract Assistant.
Centralizes environment variables and project settings.
Pattern: Notebook 00 (pydantic.SecretStr for secrets, os.environ for config)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Google Gemini / LLM Settings ────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# LLM model — Gemini 2.5 Flash
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# Embedding model — Google's gemini-embedding-001
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")

# ─── Document Processing Settings ────────────────────────────────────────────

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
MIN_CHUNK_LENGTH = int(os.environ.get("MIN_CHUNK_LENGTH", "200"))

# Separators for RecursiveCharacterTextSplitter (from Notebook 05)
CHUNK_SEPARATORS = ["\n\n", "\n", ".", ";", ",", " ", ""]

# ─── Vector Store Settings ────────────────────────────────────────────────────

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore_index")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Server Settings ─────────────────────────────────────────────────────────

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "9012"))

# ─── Guardrail Settings ──────────────────────────────────────────────────────

GUARDRAIL_THRESHOLD = float(os.environ.get("GUARDRAIL_THRESHOLD", "0.3"))

SAFE_TOPICS = [
    "contract terms and conditions",
    "legal clauses and obligations",
    "payment terms and pricing",
    "insurance policy coverage",
    "liability and indemnification",
    "confidentiality and non-disclosure",
    "termination and renewal",
    "compliance and regulatory requirements",
    "document summary and overview",
    "definitions and interpretations",
]

# ─── Evaluation Settings ─────────────────────────────────────────────────────

EVAL_NUM_SAMPLES = int(os.environ.get("EVAL_NUM_SAMPLES", "10"))
EVAL_PREFERENCE_THRESHOLD = float(os.environ.get("EVAL_PREFERENCE_THRESHOLD", "0.7"))
