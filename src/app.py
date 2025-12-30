"""
Gradio UI - Interactive interface for the RAG system.

WHY GRADIO:
Gradio makes it easy to create web interfaces for ML applications:
- Simple Python API
- Built-in components for chat, sliders, etc.
- Automatic API generation
- Easy to share (public URLs)

INTERVIEW INSIGHT:
"I built a Gradio interface that lets users adjust retrieval parameters
in real-time. They can see the retrieved chunks to understand why the
system gave a particular answer - this transparency builds trust."
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.chain.rag_chain import RAGPipeline, create_retriever, format_context


# Default paths
DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


def format_sources_display(sources: list[dict]) -> str:
    """
    Format sources for display in the UI.

    Creates a readable list of sources with section, pages, and preview.
    """
    if not sources:
        return "No sources retrieved."

    lines = ["**Retrieved Sources:**\n"]

    for i, source in enumerate(sources, start=1):
        pages_str = ", ".join(str(p) for p in source["pages"])
        lines.append(f"**[{i}] {source['section']}** (Pages: {pages_str})")
        lines.append(f"Score: {source['score']:.4f}")
        lines.append(f"Preview: {source['preview']}")
        lines.append("")

    return "\n".join(lines)


class RAGApp:
    """
    Gradio application wrapper for the RAG pipeline.

    Manages state and provides callback functions for the UI.
    """

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.pipeline: RAGPipeline | None = None
        self.current_top_k = 5

    def initialize_pipeline(self, top_k: int = 5, model_name: str = "llama-3.1-8b-instant"):
        """Initialize or reinitialize the RAG pipeline."""
        self.current_top_k = top_k
        self.pipeline = RAGPipeline(
            index_dir=self.index_dir,
            top_k=top_k,
            model_name=model_name
        )

    def chat_response(
        self,
        message: str,
        history: list[list[str]],
        top_k: int,
        show_sources: bool,
        model_name: str
    ) -> tuple[str, str]:
        """
        Process a chat message through the RAG pipeline.

        Args:
            message: User's question
            history: Chat history (not used currently, but required by Gradio)
            top_k: Number of chunks to retrieve
            show_sources: Whether to display retrieved sources
            model_name: LLM model to use

        Returns:
            Tuple of (answer, sources_display)
        """
        if not self.index_dir.exists():
            return "Index not found. Please run ingestion first.", ""

        # Reinitialize pipeline if parameters changed
        if self.pipeline is None or self.current_top_k != top_k:
            self.initialize_pipeline(top_k=top_k, model_name=model_name)

        # Get answer
        answer = self.pipeline.query(message)

        # Format sources if requested
        sources_display = ""
        if show_sources:
            sources = self.pipeline.get_sources()
            sources_display = format_sources_display(sources)

        return answer, sources_display

    def preview_retrieval(self, query: str, top_k: int) -> str:
        """
        Preview what would be retrieved without calling the LLM.

        Useful for debugging retrieval quality.
        """
        if not self.index_dir.exists():
            return "Index not found. Please run ingestion first."

        if not query.strip():
            return "Enter a query to preview retrieval."

        retriever = create_retriever(self.index_dir, top_k=top_k)
        results = retriever.retrieve(query)

        return format_context(results)


def create_ui(index_dir: str | Path = DEFAULT_INDEX_DIR) -> gr.Blocks:
    """
    Create the Gradio interface.

    Features:
    - Chat interface for Q&A
    - Slider to adjust number of retrieved chunks (k)
    - Toggle to show/hide retrieved sources
    - Model selection dropdown
    - Retrieval preview tab for debugging
    """
    app = RAGApp(index_dir)

    with gr.Blocks(title="RAG Tutor - AI Engineering Q&A") as demo:
        gr.Markdown("# RAG Tutor - AI Engineering Q&A")
        gr.Markdown("Ask questions about AI Engineering concepts!")

        with gr.Tabs():
            # Main Chat Tab
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400
                        )
                        msg = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about AI Engineering...",
                            lines=2
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Submit", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of chunks (k)"
                        )
                        show_sources = gr.Checkbox(
                            value=True,
                            label="Show retrieved sources"
                        )
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "llama-3.1-8b-instant",
                                "llama-3.1-70b-versatile",
                                "mixtral-8x7b-32768"
                            ],
                            value="llama-3.1-8b-instant",
                            label="Model"
                        )

                sources_output = gr.Markdown(label="Sources")

                def respond(message, history, top_k, show_sources, model_name):
                    answer, sources = app.chat_response(
                        message, history, top_k, show_sources, model_name
                    )
                    history.append([message, answer])
                    return "", history, sources

                submit_btn.click(
                    respond,
                    inputs=[msg, chatbot, top_k_slider, show_sources, model_dropdown],
                    outputs=[msg, chatbot, sources_output]
                )
                msg.submit(
                    respond,
                    inputs=[msg, chatbot, top_k_slider, show_sources, model_dropdown],
                    outputs=[msg, chatbot, sources_output]
                )
                clear_btn.click(
                    lambda: ([], ""),
                    outputs=[chatbot, sources_output]
                )

            # Retrieval Preview Tab
            with gr.TabItem("Retrieval Preview"):
                gr.Markdown("### Preview Retrieved Chunks")
                gr.Markdown("Test retrieval without calling the LLM.")

                preview_query = gr.Textbox(
                    label="Query",
                    placeholder="Enter a query to see what would be retrieved..."
                )
                preview_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of chunks"
                )
                preview_btn = gr.Button("Preview Retrieval")
                preview_output = gr.Markdown(label="Retrieved Chunks")

                preview_btn.click(
                    app.preview_retrieval,
                    inputs=[preview_query, preview_k],
                    outputs=[preview_output]
                )

    return demo


def launch_app(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    share: bool = False,
    server_port: int = 7860
):
    """
    Launch the Gradio application.

    Args:
        index_dir: Directory containing saved indexes
        share: Whether to create a public URL
        server_port: Port to run the server on
    """
    demo = create_ui(index_dir)
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch_app()
