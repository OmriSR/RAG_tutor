"""
Reranker - Cross-encoder for precise relevance scoring.

WHY RERANKING:
Bi-encoders (like our embedding model) encode query and document separately.
They're fast but can miss nuanced relevance signals.

Cross-encoders process query + document together, capturing interactions
between them. They're slower but more accurate.

TYPICAL RAG PIPELINE:
1. Retrieve 20-50 candidates with bi-encoder (fast)
2. Rerank top candidates with cross-encoder (accurate)
3. Use top 3-5 for LLM context

INTERVIEW INSIGHT:
"I use a two-stage retrieval: fast initial retrieval with bi-encoders,
then reranking with a cross-encoder. This gives us the speed of approximate
search with the precision of attention-based relevance scoring."
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from .hybrid_search import SearchResult


# Singleton for cross-encoder model
_reranker_model: CrossEncoder | None = None
_reranker_model_name: str | None = None


def load_reranker_model(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> CrossEncoder:
    """
    Load a cross-encoder model for reranking.

    Uses singleton pattern to avoid reloading on each call.

    WHY ms-marco-MiniLM-L-6-v2:
    - Trained on MS MARCO passage ranking
    - Good balance of speed and accuracy
    - Small enough to run on CPU

    For better quality, consider cross-encoder/ms-marco-MiniLM-L-12-v2
    or ms-marco-TinyBERT-L-2 for even faster inference.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded CrossEncoder model
    """
    global _reranker_model, _reranker_model_name

    if _reranker_model is None or _reranker_model_name != model_name:
        _reranker_model = CrossEncoder(model_name)
        _reranker_model_name = model_name

    return _reranker_model


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Takes candidates from hybrid search and reorders them based on
    cross-encoder relevance scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: Cross-encoder model to use
        """
        self.model = load_reranker_model(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        The cross-encoder processes each (query, document) pair and outputs
        a relevance score. We then sort by this score.

        Args:
            query: User's question
            results: Search results from hybrid search
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, result.chunk.text) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Get indices sorted by score (descending)
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Limit to top_k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        # Create SearchResult only for top_k results
        reranked = []
        for idx in sorted_indices:
            result = results[idx]
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(scores[idx]),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank
            ))

        return reranked
