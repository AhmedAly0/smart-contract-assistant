"""
Gradio frontend application.
Pattern: Notebook 03 (gr.ChatInterface with streaming generator)
Pattern: Notebook 09 (RemoteRunnable consuming LangServe endpoints)
Pattern: Notebook 03 (frontend_server.py composing retrieval + generation)
"""

import os
import requests
import gradio as gr

from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from operator import itemgetter

from config import SERVER_HOST, SERVER_PORT
from utils import docs2str

# ─── Remote Chain Connections ────────────────────────────────────────────────
# Pattern: Notebook 09 — RemoteRunnable connects to LangServe endpoints

SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

chains_dict = {
    "basic": RemoteRunnable(f"{SERVER_URL}/basic_chat/"),
    "generator": RemoteRunnable(f"{SERVER_URL}/generator/"),
}


# ─── Chat Handler ───────────────────────────────────────────────────────────

def chat_stream(message: str, history: list):
    """Handle chat messages with RAG pipeline.
    Pattern: Notebook 03 — streaming generator for gr.ChatInterface.

    Uses the /ask REST endpoint which handles:
    - Guardrail checking
    - Retrieval from docstore + convstore
    - Generation with citations
    - Conversation memory saving
    """
    try:
        response = requests.post(
            f"{SERVER_URL}/ask",
            params={"query": message, "use_guardrail": True},
            timeout=60,
        )
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No response generated.")
            if result.get("guardrail_blocked"):
                yield f"⚠️ {answer}"
            else:
                # Simulate streaming by yielding progressively
                # (RemoteRunnable streaming could also be used here)
                buffer = ""
                words = answer.split(" ")
                for word in words:
                    buffer += word + " "
                    yield buffer
        else:
            yield f"Error: Server returned {response.status_code}. {response.text}"

    except requests.exceptions.ConnectionError:
        yield (
            "❌ Cannot connect to the backend server. "
            "Please ensure server_app.py is running:\n"
            "```\npython server_app.py\n```"
        )
    except Exception as e:
        yield f"Error: {str(e)}"


def basic_chat_stream(message: str, history: list):
    """Direct LLM chat without RAG — uses /basic_chat endpoint.
    Pattern: Notebook 03 — streaming generator with RemoteRunnable.
    """
    try:
        buffer = ""
        for token in chains_dict["basic"].stream({"input": message}):
            if isinstance(token, str):
                buffer += token
            else:
                buffer += str(token)
            yield buffer
    except Exception as e:
        yield f"Error: {str(e)}"


# ─── Upload Handler ─────────────────────────────────────────────────────────

def upload_file(file):
    """Upload a document to the backend for ingestion."""
    if file is None:
        return "No file selected."

    try:
        # Read the file and send to backend
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f)}
            response = requests.post(f"{SERVER_URL}/upload", files=files, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return (
                f"✅ **{result['filename']}** uploaded successfully!\n\n"
                f"- Chunks created: **{result['chunks_created']}**\n"
                f"- Status: {result['message']}\n\n"
                f"You can now ask questions about this document in the Chat tab."
            )
        else:
            return f"❌ Upload failed: {response.text}"

    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to the backend server. "
            "Please start it with: `python server_app.py`"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─── Summarize Handler ──────────────────────────────────────────────────────

def summarize_document(doc_name: str = ""):
    """Request a progressive summary of the uploaded document.
    Pattern: Notebook 05 — RSummarizer called via REST endpoint.
    """
    try:
        params = {}
        if doc_name.strip():
            params["doc_name"] = doc_name.strip()

        response = requests.post(
            f"{SERVER_URL}/summarize",
            params=params,
            timeout=180,
        )

        if response.status_code == 200:
            result = response.json()
            summary = result.get("summary", {})
            chunks_processed = result.get("chunks_processed", 0)

            # Format the summary nicely
            output = f"## Document Summary\n\n"
            output += f"**Chunks Processed:** {chunks_processed}\n\n"

            if isinstance(summary, dict):
                if summary.get("running_summary"):
                    output += f"### Overview\n{summary['running_summary']}\n\n"
                if summary.get("document_type"):
                    output += f"**Document Type:** {summary['document_type']}\n\n"
                if summary.get("parties_involved"):
                    output += f"**Parties:** {', '.join(summary['parties_involved'])}\n\n"
                if summary.get("main_ideas"):
                    output += "### Key Points\n"
                    for idea in summary["main_ideas"]:
                        output += f"- {idea}\n"
                    output += "\n"
                if summary.get("loose_ends"):
                    output += "### Open Questions\n"
                    for q in summary["loose_ends"]:
                        output += f"- {q}\n"
            else:
                output += str(summary)

            return output
        else:
            return f"❌ Summarization failed: {response.text}"

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to the backend server."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui():
    """Build the Gradio interface with Upload, Chat, and Summary tabs.
    Pattern: Notebook 03 — gr.ChatInterface with queue().launch().
    """
    with gr.Blocks(
        title="Smart Contract Assistant",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# 📄 Smart Contract Summary & Q&A Assistant\n"
            "Upload contracts, insurance policies, or reports and chat with them. "
            "Powered by RAG with source citations and semantic guardrails.\n\n"
            "*DLI Course Capstone Project — Building RAG Agents using LLMs (Gemini 2.5 Flash)*"
        )

        with gr.Tabs():
            # ─── Upload Tab ──────────────────────────────────────────
            with gr.Tab("📤 Upload Document"):
                gr.Markdown(
                    "Upload a PDF, DOCX, or TXT file. "
                    "The system will extract, chunk, and embed the content."
                )
                with gr.Row():
                    file_input = gr.File(
                        label="Select Document",
                        file_types=[".pdf", ".docx", ".txt"],
                    )
                upload_btn = gr.Button("Upload & Process", variant="primary")
                upload_output = gr.Markdown(label="Upload Status")
                upload_btn.click(
                    fn=upload_file,
                    inputs=[file_input],
                    outputs=[upload_output],
                )

            # ─── Chat Tab ────────────────────────────────────────────
            with gr.Tab("💬 Chat with Document"):
                gr.Markdown(
                    "Ask questions about uploaded documents. "
                    "Answers include source citations from the document."
                )
                gr.ChatInterface(
                    fn=chat_stream,
                    examples=[
                        "What are the key terms and conditions?",
                        "Who are the parties involved in this contract?",
                        "What are the payment terms?",
                        "What happens upon termination?",
                        "Summarize the main obligations.",
                    ],
                )

            # ─── Summary Tab ─────────────────────────────────────────
            with gr.Tab("📋 Document Summary"):
                gr.Markdown(
                    "Generate a progressive summary of the uploaded document."
                )
                doc_name_input = gr.Textbox(
                    label="Document Name (optional)",
                    placeholder="Leave empty to summarize the most recent document",
                )
                summarize_btn = gr.Button("Generate Summary", variant="primary")
                summary_output = gr.Markdown(label="Summary")
                summarize_btn.click(
                    fn=summarize_document,
                    inputs=[doc_name_input],
                    outputs=[summary_output],
                )

            # ─── Direct Chat Tab ─────────────────────────────────────
            with gr.Tab("🤖 Direct LLM Chat"):
                gr.Markdown(
                    "Chat directly with the LLM without document retrieval. "
                    "Useful for general questions."
                )
                gr.ChatInterface(
                    fn=basic_chat_stream,
                )

    return demo


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=8090,
        share=False,
        show_error=True,
    )
