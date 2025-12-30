"""Tests for the indexer module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunker import Chunk, ChunkMetadata
from src.ingestion.indexer import (
    build_bm25_index,
    build_faiss_index,
    build_indexes,
    load_embedding_model,
    load_indexes,
    save_indexes,
    tokenize_for_bm25,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            text="Machine learning is a subset of artificial intelligence.",
            metadata=ChunkMetadata(page_numbers=[1], section_title="Intro", chunk_index=0)
        ),
        Chunk(
            text="Deep learning uses neural networks with many layers.",
            metadata=ChunkMetadata(page_numbers=[2], section_title="Deep Learning", chunk_index=1)
        ),
        Chunk(
            text="Transformers revolutionized natural language processing.",
            metadata=ChunkMetadata(page_numbers=[3], section_title="NLP", chunk_index=2)
        ),
    ]


class TestTokenizeForBm25:
    """Tests for BM25 tokenization."""

    def test_basic_tokenization(self):
        text = "Hello World"
        tokens = tokenize_for_bm25(text)
        assert tokens == ["hello", "world"]

    def test_lowercase_conversion(self):
        text = "UPPERCASE lowercase MiXeD"
        tokens = tokenize_for_bm25(text)
        assert all(t.islower() for t in tokens)

    def test_empty_string(self):
        tokens = tokenize_for_bm25("")
        assert tokens == []


class TestLoadEmbeddingModel:
    """Tests for embedding model singleton."""

    def test_returns_model(self):
        model = load_embedding_model()
        assert model is not None

    def test_singleton_returns_same_instance(self):
        model1 = load_embedding_model()
        model2 = load_embedding_model()
        assert model1 is model2


class TestBuildFaissIndex:
    """Tests for FAISS index building."""

    def test_creates_index(self, sample_chunks):
        model = load_embedding_model()
        index, embeddings = build_faiss_index(sample_chunks, model)

        assert index is not None
        assert index.ntotal == len(sample_chunks)

    def test_embeddings_shape(self, sample_chunks):
        model = load_embedding_model()
        _, embeddings = build_faiss_index(sample_chunks, model)

        assert embeddings.shape[0] == len(sample_chunks)
        assert embeddings.shape[1] == 384  # MiniLM dimension

    def test_embeddings_normalized(self, sample_chunks):
        model = load_embedding_model()
        _, embeddings = build_faiss_index(sample_chunks, model)

        # Check that embeddings are normalized (L2 norm ~= 1)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(sample_chunks)), decimal=5)


class TestBuildBm25Index:
    """Tests for BM25 index building."""

    def test_creates_index(self, sample_chunks):
        bm25 = build_bm25_index(sample_chunks)
        assert bm25 is not None

    def test_can_score_query(self, sample_chunks):
        bm25 = build_bm25_index(sample_chunks)
        scores = bm25.get_scores(["machine", "learning"])

        assert len(scores) == len(sample_chunks)
        # First chunk should score highest for "machine learning"
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]


class TestSaveLoadIndexes:
    """Tests for index persistence."""

    def test_save_and_load_roundtrip(self, sample_chunks):
        model = load_embedding_model()
        faiss_index, embeddings = build_faiss_index(sample_chunks, model)
        bm25_index = build_bm25_index(sample_chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_indexes(tmpdir, faiss_index, bm25_index, sample_chunks, embeddings)

            # Verify files exist
            assert (Path(tmpdir) / "faiss.index").exists()
            assert (Path(tmpdir) / "bm25.pkl").exists()
            assert (Path(tmpdir) / "chunks.pkl").exists()
            assert (Path(tmpdir) / "embeddings.npy").exists()

            # Load
            loaded_faiss, loaded_bm25, loaded_chunks, loaded_embeddings = load_indexes(tmpdir)

            # Verify loaded data
            assert loaded_faiss.ntotal == faiss_index.ntotal
            assert len(loaded_chunks) == len(sample_chunks)
            np.testing.assert_array_equal(loaded_embeddings, embeddings)

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_indexes(tmpdir)


class TestBuildIndexes:
    """Tests for the main orchestration function."""

    def test_builds_both_indexes(self, sample_chunks):
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_index, bm25_index = build_indexes(sample_chunks, tmpdir)

            assert faiss_index is not None
            assert bm25_index is not None
            assert faiss_index.ntotal == len(sample_chunks)
