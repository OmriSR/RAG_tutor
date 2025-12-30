# RAG Tutor - Implementation Tasks

## Phase 1: Setup
- [x] Create project structure and directories
- [x] Create requirements.txt with dependencies
- [x] Create .env template for Groq API key

## Phase 2: Ingestion Pipeline

### pdf_parser.py
- [x] Create LineInfo dataclass (text, font_size, top, is_bold)
- [x] Create PageContent dataclass (page_number, text, lines)
- [x] Implement extract_lines_with_metadata() - extract lines with font info
- [x] Implement detect_heading() - identify headings by font size/bold
- [x] Implement parse_pdf() - generator yielding PageContent per page
- [x] Implement parse_pdf_to_list() - convenience wrapper

### chunker.py
- [x] Create Chunk dataclass (text, metadata with page, section, index)
- [x] Implement detect_sections() - find section boundaries from headings
- [x] Implement split_into_paragraphs() - split text by paragraph breaks
- [x] Implement create_chunks_with_overlap() - add overlap between chunks
- [x] Implement semantic_chunk() - main function combining above

### indexer.py
- [x] Implement load_embedding_model() - load all-MiniLM-L6-v2 (singleton)
- [x] Implement build_faiss_index() - create FAISS IndexFlatIP
- [x] Implement build_bm25_index() - create BM25 index from texts
- [x] Implement save_indexes() - persist to disk
- [x] Implement load_indexes() - load from disk
- [x] Implement build_indexes() - main orchestration function

## Phase 3: Retrieval Pipeline

### hybrid_search.py
- [x] Create HybridSearcher class
- [x] Implement dense_search() - FAISS similarity search
- [x] Implement sparse_search() - BM25 keyword search
- [x] Implement reciprocal_rank_fusion() - merge results with RRF
- [x] Implement search() - combined hybrid search returning top-k

### reranker.py
- [x] Create Reranker class
- [x] Implement load_reranker_model() - load cross-encoder (singleton)
- [x] Implement rerank() - score and reorder candidates

### rag_chain.py
- [x] Implement create_retriever() - custom retriever with hybrid+rerank
- [x] Implement create_prompt() - RAG prompt template
- [x] Implement create_rag_chain() - LangChain chain with LCEL
- [x] Implement RAGPipeline class - high-level interface

## Phase 4: Interface

### app.py
- [x] Implement chat_response() - process query through RAG chain
- [x] Implement format_sources_display() - display source citations
- [x] Implement create_ui() - Gradio interface with controls
- [x] Add "show retrieved chunks" toggle
- [x] Add k slider for retrieval depth
- [x] Add retrieval preview tab

### run.py
- [x] Implement check_indexes_exist() - verify index files
- [x] Implement run_ingestion() - orchestrate PDF processing
- [x] Implement main() - entry point with CLI args
- [x] Add query subcommand for CLI testing
