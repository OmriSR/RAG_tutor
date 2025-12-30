"""
RAG Chain - LangChain integration for retrieval-augmented generation.

WHY LANGCHAIN:
LangChain provides abstractions for building LLM applications:
- Standard interfaces for different LLM providers
- Prompt templates with variable substitution
- Chain composition for multi-step workflows

We use Groq for LLM inference - it's fast and has a generous free tier.

INTERVIEW INSIGHT:
"I built the RAG chain using LangChain for its flexibility. The retriever
is a custom implementation combining hybrid search with cross-encoder
reranking. The prompt template instructs the LLM to cite sources and
acknowledge uncertainty."
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

from src.ingestion.indexer import load_indexes
from src.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.retrieval.reranker import Reranker

# Load environment variables
load_dotenv()


RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions about AI Engineering.

Use the following retrieved context to answer the question. If the context doesn't contain enough information to answer fully, say so honestly.

When citing information, reference the section and page numbers from the context.

Context:
{context}

Question: {question}

Answer:"""


def format_context(results: list[SearchResult]) -> str:
    """
    Format search results into context string for the prompt.

    Each chunk is formatted with its metadata (section, pages) to enable
    the LLM to provide proper citations.
    """
    formatted_parts = []

    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        meta = chunk.metadata

        pages_str = ", ".join(str(p) for p in meta.page_numbers)
        header = f"[{i}] Section: {meta.section_title} | Pages: {pages_str}"

        formatted_parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(formatted_parts)


class RAGRetriever:
    """
    Custom retriever combining hybrid search and reranking.

    This wraps our hybrid search + reranking pipeline into a single
    callable that can be used in a LangChain chain.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        top_k: int = 5,
        rerank_candidates: int = 20
    ):
        """
        Initialize the RAG retriever.

        Args:
            searcher: HybridSearcher instance
            reranker: Reranker instance
            top_k: Number of final results to return
            rerank_candidates: Number of candidates to retrieve before reranking
        """
        self.searcher = searcher
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def retrieve(self, query: str) -> list[SearchResult]:
        """
        Retrieve and rerank documents for a query.

        Pipeline:
        1. Hybrid search retrieves candidates
        2. Cross-encoder reranks candidates
        3. Return top-k results
        """
        candidates = self.searcher.search(query, k=self.rerank_candidates)
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        return reranked

    def __call__(self, query: str) -> str:
        """
        Make retriever callable for LangChain integration.

        Takes a query string, retrieves documents, and returns
        formatted context string.
        """
        results = self.retrieve(query)
        return format_context(results)


def create_retriever(
    index_dir: str | Path,
    top_k: int = 5,
    rerank_candidates: int = 20
) -> RAGRetriever:
    """
    Create a RAG retriever from saved indexes.

    Args:
        index_dir: Directory containing saved indexes
        top_k: Number of final results to return
        rerank_candidates: Candidates to consider for reranking

    Returns:
        RAGRetriever instance
    """
    faiss_index, bm25_index, chunks, _ = load_indexes(index_dir)

    searcher = HybridSearcher(faiss_index, bm25_index, chunks)
    reranker = Reranker()

    return RAGRetriever(
        searcher=searcher,
        reranker=reranker,
        top_k=top_k,
        rerank_candidates=rerank_candidates
    )


def create_prompt() -> ChatPromptTemplate:
    """
    Create the RAG prompt template.

    The prompt instructs the LLM to:
    - Use the provided context
    - Cite sources with section/page info
    - Acknowledge when context is insufficient
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def create_llm(model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Create a Groq LLM instance.

    WHY llama-3.1-8b-instant:
    - Fast inference (Groq's specialty)
    - Good quality for RAG tasks
    - Free tier available

    For better quality, consider llama-3.1-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )


def create_rag_chain(
    index_dir: str | Path,
    top_k: int = 5,
    model_name: str = "llama-3.1-8b-instant"
):
    """
    Create the complete RAG chain using LCEL.

    CHAIN STRUCTURE (using LangChain Expression Language):
    1. RunnablePassthrough passes the question through
    2. Retriever fetches and formats context
    3. Prompt combines context + question
    4. LLM generates answer
    5. StrOutputParser extracts text

    Args:
        index_dir: Directory containing saved indexes
        top_k: Number of chunks to retrieve
        model_name: Groq model to use

    Returns:
        Runnable chain that takes {"question": str} and returns str
    """
    retriever = create_retriever(index_dir, top_k=top_k)
    prompt = create_prompt()
    llm = create_llm(model_name)
    output_parser = StrOutputParser()

    # Build chain using LCEL pipe syntax
    # RunnablePassthrough passes input through unchanged
    # RunnableLambda wraps our retriever callable
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retriever(x["question"]))
        )
        | prompt
        | llm
        | output_parser
    )

    return chain


class RAGPipeline:
    """
    High-level RAG pipeline for easy usage.

    Wraps the chain and provides methods for querying with
    optional access to retrieved chunks.
    """

    def __init__(
        self,
        index_dir: str | Path,
        top_k: int = 5,
        model_name: str = "llama-3.1-8b-instant"
    ):
        self.retriever = create_retriever(index_dir, top_k=top_k)
        self.chain = create_rag_chain(index_dir, top_k=top_k, model_name=model_name)
        self.last_results: list[SearchResult] = []

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline.

        Args:
            question: User's question

        Returns:
            LLM's answer
        """
        # Store retrieved results for inspection
        self.last_results = self.retriever.retrieve(question)

        # Invoke the chain
        answer = self.chain.invoke({"question": question})

        return answer

    def get_sources(self) -> list[dict]:
        """
        Get source information from the last query.

        Returns list of dicts with section, pages, and text preview.
        """
        sources = []
        for result in self.last_results:
            chunk = result.chunk
            sources.append({
                "section": chunk.metadata.section_title,
                "pages": chunk.metadata.page_numbers,
                "score": result.score,
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        return sources
