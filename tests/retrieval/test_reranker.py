"""Tests for the reranker module."""

import pytest

from src.ingestion.chunker import Chunk, ChunkMetadata
from src.retrieval.hybrid_search import SearchResult
from src.retrieval.reranker import Reranker, load_reranker_model


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    chunks = [
        Chunk(
            text="Machine learning is a field of AI focused on learning from data.",
            metadata=ChunkMetadata(page_numbers=[1], section_title="ML", chunk_index=0)
        ),
        Chunk(
            text="Cooking recipes require careful measurement of ingredients.",
            metadata=ChunkMetadata(page_numbers=[2], section_title="Cooking", chunk_index=1)
        ),
        Chunk(
            text="Deep learning uses neural networks to learn representations.",
            metadata=ChunkMetadata(page_numbers=[3], section_title="DL", chunk_index=2)
        ),
    ]

    return [
        SearchResult(chunk=chunks[0], score=0.8, dense_rank=1, sparse_rank=2),
        SearchResult(chunk=chunks[1], score=0.7, dense_rank=2, sparse_rank=1),
        SearchResult(chunk=chunks[2], score=0.6, dense_rank=3, sparse_rank=3),
    ]


class TestLoadRerankerModel:
    """Tests for reranker model singleton."""

    def test_returns_model(self):
        model = load_reranker_model()
        assert model is not None

    def test_singleton_returns_same_instance(self):
        model1 = load_reranker_model()
        model2 = load_reranker_model()
        assert model1 is model2


class TestReranker:
    """Tests for the Reranker class."""

    @pytest.fixture
    def reranker(self) -> Reranker:
        return Reranker()

    def test_rerank_returns_results(self, reranker, sample_results):
        reranked = reranker.rerank("What is machine learning?", sample_results)

        assert len(reranked) == len(sample_results)
        assert all(isinstance(r, SearchResult) for r in reranked)

    def test_rerank_reorders_by_relevance(self, reranker, sample_results):
        query = "What is machine learning?"
        reranked = reranker.rerank(query, sample_results)

        # ML and DL chunks should rank higher than cooking for ML query
        top_2_indices = [r.chunk.metadata.chunk_index for r in reranked[:2]]
        assert 0 in top_2_indices  # ML chunk
        assert 2 in top_2_indices  # DL chunk
        assert 1 not in top_2_indices  # Cooking should not be in top 2

    def test_rerank_top_k(self, reranker, sample_results):
        reranked = reranker.rerank("machine learning", sample_results, top_k=2)

        assert len(reranked) == 2

    def test_rerank_empty_list(self, reranker):
        reranked = reranker.rerank("query", [])

        assert reranked == []

    def test_rerank_preserves_original_ranks(self, reranker, sample_results):
        reranked = reranker.rerank("machine learning", sample_results)

        for result in reranked:
            # Original dense/sparse ranks should be preserved
            assert result.dense_rank is not None or result.sparse_rank is not None

    def test_scores_are_cross_encoder_scores(self, reranker, sample_results):
        reranked = reranker.rerank("machine learning", sample_results)

        # Cross-encoder scores are typically in a different range than original
        # They should reflect actual relevance
        for result in reranked:
            assert isinstance(result.score, float)
