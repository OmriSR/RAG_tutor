"""
Indexer - Build and persist FAISS and BM25 indexes for hybrid search.

WHY TWO INDEXES:
Hybrid search combines dense (semantic) and sparse (keyword) retrieval:
- FAISS (dense): Finds semantically similar content via embeddings
- BM25 (sparse): Finds exact keyword matches (great for technical terms)

Together they catch both "what you mean" and "what you say".

INTERVIEW INSIGHT:
"I use hybrid search because semantic search alone misses exact term matches.
If someone asks about 'RLHF', BM25 finds documents with that acronym even if
the embedding model hasn't seen it. Dense search catches conceptual matches
like 'reinforcement learning from human feedback'."
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .chunker import Chunk


# Singleton for embedding model - avoids reloading on each call
_embedding_model: SentenceTransformer | None = None
_embedding_model_name: str | None = None


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a sentence transformer model for dense embeddings.

    Uses singleton pattern to avoid reloading the model on each call.
    Loading a transformer model is expensive (~1-2 seconds, ~300MB memory).

    WHY all-MiniLM-L6-v2:
    - Fast: 80ms per sentence on CPU
    - Small: 80MB model size
    - Good quality: Strong performance on semantic similarity
    - Free: No API costs, runs locally

    For production, consider larger models like all-mpnet-base-v2 or
    OpenAI's text-embedding-3-small for better quality.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded SentenceTransformer model
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is None or _embedding_model_name != model_name:
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model_name = model_name

    return _embedding_model


def build_faiss_index(
    chunks: list[Chunk],
    model: SentenceTransformer
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Build a FAISS index from chunk embeddings.

    WHY IndexFlatIP:
    - IP = Inner Product (equivalent to cosine similarity for normalized vectors)
    - Flat = Exact search (no approximation)
    - For <100k chunks, exact search is fast enough. Use IVF for larger datasets.

    Args:
        chunks: List of Chunk objects to index
        model: SentenceTransformer for generating embeddings

    Returns:
        Tuple of (FAISS index, embedding matrix)
    """
    texts = [chunk.text for chunk in chunks]

    # Generate embeddings - model.encode handles batching internally
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity via IP
    )

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    return index, embeddings


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenization for BM25.

    BM25 works on tokens, not raw text. We do basic preprocessing:
    - Lowercase for case-insensitive matching
    - Split on whitespace and punctuation
    """
    return text.lower().split()


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    """
    Build a BM25 index for sparse keyword search.

    WHY BM25Okapi:
    - Classic information retrieval algorithm
    - Handles term frequency and document length normalization
    - "Okapi" variant is the most commonly used

    Args:
        chunks: List of Chunk objects to index

    Returns:
        BM25Okapi index
    """
    tokenized_corpus = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def save_indexes(
    index_dir: str | Path,
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    chunks: list[Chunk],
    embeddings: np.ndarray
) -> None:
    """
    Persist indexes and chunks to disk.

    We save:
    - FAISS index (.faiss): The vector index
    - BM25 index (.pkl): The keyword index
    - Chunks (.pkl): Original chunks with metadata for retrieval
    - Embeddings (.npy): Raw embeddings for debugging/analysis

    Args:
        index_dir: Directory to save indexes
        faiss_index: FAISS index to save
        bm25_index: BM25 index to save
        chunks: Original chunks for retrieval
        embeddings: Embedding matrix
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    # Save BM25 index
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    # Save chunks with metadata
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Save embeddings (useful for debugging)
    np.save(index_dir / "embeddings.npy", embeddings)


def load_indexes(
    index_dir: str | Path
) -> tuple[faiss.IndexFlatIP, BM25Okapi, list[Chunk], np.ndarray]:
    """
    Load previously saved indexes from disk.

    Args:
        index_dir: Directory containing saved indexes

    Returns:
        Tuple of (FAISS index, BM25 index, chunks, embeddings)

    Raises:
        FileNotFoundError: If any required file is missing
    """
    index_dir = Path(index_dir)

    # Verify all files exist
    required_files = ["faiss.index", "bm25.pkl", "chunks.pkl", "embeddings.npy"]
    for filename in required_files:
        if not (index_dir / filename).exists():
            raise FileNotFoundError(f"Missing index file: {index_dir / filename}")

    # Load FAISS index
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

    # Load BM25 index
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25_index = pickle.load(f)

    # Load chunks
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load embeddings
    embeddings = np.load(index_dir / "embeddings.npy")

    return faiss_index, bm25_index, chunks, embeddings


def build_indexes(
    chunks: list[Chunk],
    index_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[faiss.IndexFlatIP, BM25Okapi]:
    """
    Main orchestration function: build all indexes and save to disk.

    This is the entry point for the indexing pipeline. It:
    1. Loads the embedding model (singleton - only loads once)
    2. Builds FAISS index from embeddings
    3. Builds BM25 index from tokens
    4. Saves everything to disk

    Args:
        chunks: List of Chunk objects to index
        index_dir: Directory to save indexes
        model_name: Embedding model to use

    Returns:
        Tuple of (FAISS index, BM25 index)
    """
    print(f"Building indexes for {len(chunks)} chunks...")

    # Load embedding model (singleton pattern)
    print("Loading embedding model...")
    model = load_embedding_model(model_name)

    # Build FAISS index
    print("Building FAISS index...")
    faiss_index, embeddings = build_faiss_index(chunks, model)
    print(f"  Created index with dimension {embeddings.shape[1]}")

    # Build BM25 index
    print("Building BM25 index...")
    bm25_index = build_bm25_index(chunks)

    # Save to disk
    print(f"Saving indexes to {index_dir}...")
    save_indexes(index_dir, faiss_index, bm25_index, chunks, embeddings)

    print("Done!")
    return faiss_index, bm25_index
