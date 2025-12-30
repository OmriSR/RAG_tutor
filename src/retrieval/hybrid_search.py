"""
Hybrid Search - Combine dense (semantic) and sparse (keyword) retrieval.

WHY HYBRID SEARCH:
Neither dense nor sparse search is perfect alone:
- Dense (FAISS): Great for semantic similarity, misses exact keyword matches
- Sparse (BM25): Great for keywords/acronyms, misses semantic variations

Example: Query "RLHF training"
- BM25 finds docs with exact "RLHF" even if embedding model doesn't know it
- Dense finds docs about "reinforcement learning from human feedback"
- Together: best of both worlds

INTERVIEW INSIGHT:
"I combine dense and sparse retrieval using Reciprocal Rank Fusion. Dense
search handles semantic similarity while BM25 catches exact term matches.
RRF merges their rankings without needing to tune weights - it's robust
and effective."
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.ingestion.indexer import load_embedding_model, tokenize_for_bm25


@dataclass
class SearchResult:
    """
    A single search result with score and ranking info.

    chunk: The retrieved chunk with text and metadata
    score: Combined score from fusion (higher = better)
    dense_rank: Rank from dense search (None if not in dense results)
    sparse_rank: Rank from sparse search (None if not in sparse results)
    """
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class HybridSearcher:
    """
    Hybrid retriever combining FAISS dense search with BM25 sparse search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.
    """

    def __init__(
        self,
        faiss_index: faiss.IndexFlatIP,
        bm25_index: BM25Okapi,
        chunks: list[Chunk],
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the hybrid searcher.

        Args:
            faiss_index: Pre-built FAISS index
            bm25_index: Pre-built BM25 index
            chunks: Original chunks (indexed in same order as indexes)
            model_name: Embedding model for query encoding
        """
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.model = load_embedding_model(model_name)

    def dense_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        Search using FAISS dense embeddings.

        Encodes the query and finds nearest neighbors in embedding space.

        Args:
            query: User's question
            k: Number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search FAISS
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32),
            k
        )

        # Return as list of (index, score) tuples
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((int(idx), float(score)))

        return results

    def sparse_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        Search using BM25 keyword matching.

        Tokenizes query and scores documents based on term frequency.

        Args:
            query: User's question
            k: Number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append((int(idx), float(scores[idx])))

        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k: int = 60
    ) -> list[tuple[int, float, int | None, int | None]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF FORMULA: score = sum(1 / (k + rank)) for each ranking list
        where k is a constant (default 60) that dampens the influence of rank.

        WHY RRF:
        - No need to normalize scores across different systems
        - Robust to outliers
        - Simple and effective
        - Works well empirically

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: RRF constant (higher = more weight to lower ranks)

        Returns:
            List of (chunk_index, rrf_score, dense_rank, sparse_rank)
        """
        scores: dict[int, float] = {}
        dense_ranks: dict[int, int] = {}
        sparse_ranks: dict[int, int] = {}

        # Add dense search contributions
        for rank, (idx, _) in enumerate(dense_results, start=1):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
            dense_ranks[idx] = rank

        # Add sparse search contributions
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
            sparse_ranks[idx] = rank

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (idx, score, dense_ranks.get(idx), sparse_ranks.get(idx))
            for idx, score in sorted_results
        ]

    def search(self, query: str, k: int = 5, candidates: int = 20) -> list[SearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        Pipeline:
        1. Run dense search to get top candidates
        2. Run sparse search to get top candidates
        3. Merge with RRF
        4. Return top-k results

        Args:
            query: User's question
            k: Number of final results to return
            candidates: Number of candidates to retrieve from each search

        Returns:
            List of SearchResult objects with chunks and scores
        """
        # Get candidates from both searches
        dense_results = self.dense_search(query, k=candidates)
        sparse_results = self.sparse_search(query, k=candidates)

        # Merge with RRF
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # Build results
        results = []
        for idx, score, dense_rank, sparse_rank in fused[:k]:
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank
            ))

        return results
