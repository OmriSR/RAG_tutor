"""
CLI Entry Point - Run ingestion or launch the app.

Usage:
    python run.py ingest           # Process PDF and build indexes
    python run.py app              # Launch Gradio interface
    python run.py query "question" # Quick query from command line
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"


def check_indexes_exist() -> bool:
    """
    Verify that all required index files exist.

    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        INDEX_DIR / "faiss.index",
        INDEX_DIR / "bm25.pkl",
        INDEX_DIR / "chunks.pkl",
        INDEX_DIR / "embeddings.npy"
    ]

    for filepath in required_files:
        if not filepath.exists():
            return False

    return True


def find_pdf() -> Path | None:
    """Find a PDF file in the raw data directory."""
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]
    return None


def run_ingestion(pdf_path: Path | None = None):
    """
    Run the full ingestion pipeline.

    1. Parse PDF
    2. Chunk content
    3. Build indexes
    4. Save to disk
    """
    from src.ingestion.pdf_parser import parse_pdf_to_list
    from src.ingestion.chunker import semantic_chunk
    from src.ingestion.indexer import build_indexes

    # Find PDF if not specified
    if pdf_path is None:
        pdf_path = find_pdf()
        if pdf_path is None:
            print(f"No PDF found in {RAW_DIR}")
            print("Please add a PDF file to data/raw/")
            return

    print(f"Processing: {pdf_path.name}")
    print("=" * 50)

    # Step 1: Parse PDF
    print("\n[1/3] Parsing PDF...")
    pages = parse_pdf_to_list(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    # Step 2: Chunk content
    print("\n[2/3] Chunking content...")
    chunks = semantic_chunk(pages)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Build indexes
    print("\n[3/3] Building indexes...")
    build_indexes(chunks, INDEX_DIR)

    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print(f"Indexes saved to: {INDEX_DIR}")


def run_app(share: bool = False, port: int = 7860):
    """Launch the Gradio application."""
    if not check_indexes_exist():
        print("Indexes not found. Running ingestion first...")
        run_ingestion()
        print()

    from src.app import launch_app
    launch_app(INDEX_DIR, share=share, server_port=port)


def run_query(question: str, top_k: int = 5):
    """Run a single query from command line."""
    if not check_indexes_exist():
        print("Indexes not found. Please run 'python run.py ingest' first.")
        return

    from src.chain.rag_chain import RAGPipeline

    print(f"Question: {question}")
    print("-" * 50)

    pipeline = RAGPipeline(INDEX_DIR, top_k=top_k)
    answer = pipeline.query(question)

    print(f"\nAnswer:\n{answer}")

    print("\n" + "-" * 50)
    print("Sources:")
    for source in pipeline.get_sources():
        pages = ", ".join(str(p) for p in source["pages"])
        print(f"  - {source['section']} (pages {pages})")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Tutor - AI Engineering Q&A System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Process PDF and build indexes")
    ingest_parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to PDF file (default: auto-detect in data/raw/)"
    )

    # App command
    app_parser = subparsers.add_parser("app", help="Launch Gradio interface")
    app_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL"
    )
    app_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(args.pdf)
    elif args.command == "app":
        run_app(share=args.share, port=args.port)
    elif args.command == "query":
        run_query(args.question, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
