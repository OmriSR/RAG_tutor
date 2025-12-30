"""Tests for the hybrid search module."""

import tempfile

import pytest

from src.ingestion.chunker import Chunk, ChunkMetadata
from src.ingestion.indexer import build_bm25_index, build_faiss_index, load_embedding_model
from src.retrieval.hybrid_search import HybridSearcher, SearchResult


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Machine learning is a subset of artificial intelligence that enables computers to learn.",
            metadata=ChunkMetadata(page_numbers=[1], section_title="ML Basics", chunk_index=0)
        ),
        Chunk(
            text="RLHF stands for Reinforcement Learning from Human Feedback, a technique for training LLMs.",
            metadata=ChunkMetadata(page_numbers=[2], section_title="RLHF", chunk_index=1)
        ),
        Chunk(
            text="Transformers use self-attention mechanisms to process sequences in parallel.",
            metadata=ChunkMetadata(page_numbers=[3], section_title="Transformers", chunk_index=2)
        ),
        Chunk(
            text="Vector databases store embeddings for efficient similarity search.",
            metadata=ChunkMetadata(page_numbers=[4], section_title="Vector DBs", chunk_index=3)
        ),
    ]


@pytest.fixture
def searcher(sample_chunks) -> HybridSearcher:
    """Create a HybridSearcher with sample data."""
    model = load_embedding_model()
    faiss_index, _ = build_faiss_index(sample_chunks, model)
    bm25_index = build_bm25_index(sample_chunks)

    return HybridSearcher(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=sample_chunks
    )


class TestDenseSearch:
    """Tests for dense (FAISS) search."""

    def test_returns_results(self, searcher):
        results = searcher.dense_search("machine learning", k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)

    def test_semantic_matching(self, searcher):
        # Should find ML-related content even without exact words
        results = searcher.dense_search("AI and computers learning", k=2)

        # First result should be the ML chunk (semantically similar)
        top_idx = results[0][0]
        assert top_idx == 0  # ML basics chunk


class TestSparseSearch:
    """Tests for sparse (BM25) search."""

    def test_returns_results(self, searcher):
        results = searcher.sparse_search("machine learning", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, tuple) for r in results)

    def test_exact_keyword_matching(self, searcher):
        # BM25 should find exact term "RLHF"
        results = searcher.sparse_search("RLHF", k=2)

        assert len(results) > 0
        top_idx = results[0][0]
        assert top_idx == 1  # RLHF chunk


class TestReciprocalRankFusion:
    """Tests for RRF score merging."""

    def test_combines_results(self, searcher):
        dense = [(0, 0.9), (1, 0.8)]
        sparse = [(1, 5.0), (2, 4.0)]

        fused = searcher.reciprocal_rank_fusion(dense, sparse)

        # Should have 3 unique results
        assert len(fused) == 3

        # Index 1 appears in both, should have highest score
        idx_1_score = next(score for idx, score, _, _ in fused if idx == 1)
        idx_0_score = next(score for idx, score, _, _ in fused if idx == 0)
        assert idx_1_score > idx_0_score

    def test_preserves_ranks(self, searcher):
        dense = [(0, 0.9), (1, 0.8)]
        sparse = [(1, 5.0), (2, 4.0)]

        fused = searcher.reciprocal_rank_fusion(dense, sparse)

        # Check that ranks are preserved
        for idx, _, dense_rank, sparse_rank in fused:
            if idx == 0:
                assert dense_rank == 1
                assert sparse_rank is None
            elif idx == 1:
                assert dense_rank == 2
                assert sparse_rank == 1
            elif idx == 2:
                assert dense_rank is None
                assert sparse_rank == 2


class TestHybridSearch:
    """Tests for the full hybrid search pipeline."""

    def test_returns_search_results(self, searcher):
        results = searcher.search("machine learning", k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_chunks(self, searcher):
        results = searcher.search("machine learning", k=2)

        for result in results:
            assert result.chunk is not None
            assert result.chunk.text is not None
            assert result.score > 0

    def test_respects_k_parameter(self, searcher):
        results_1 = searcher.search("learning", k=1)
        results_3 = searcher.search("learning", k=3)

        assert len(results_1) == 1
        assert len(results_3) == 3
