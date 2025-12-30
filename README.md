# RAG Tutor

A Retrieval-Augmented Generation (RAG) system for querying AI Engineering documents. Built as a learning project to understand RAG concepts through hands-on implementation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Retrieval                             │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   Dense Search      │    │   Sparse Search     │             │
│  │   (FAISS + MiniLM)  │    │   (BM25)            │             │
│  └─────────────────────┘    └─────────────────────┘             │
│              │                        │                          │
│              └──────────┬─────────────┘                          │
│                         ▼                                        │
│              ┌─────────────────────┐                             │
│              │ Reciprocal Rank     │                             │
│              │ Fusion (RRF)        │                             │
│              └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Cross-Encoder Reranking                     │
│                   (ms-marco-MiniLM-L-6-v2)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Generation                           │
│                      (Groq + Llama 3.1)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Answer with Sources                         │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack & Design Choices

### Embeddings: all-MiniLM-L6-v2
- **Why**: Fast (80ms/sentence), small (80MB), good quality, runs locally
- **Trade-off**: Larger models like `all-mpnet-base-v2` offer better quality but slower inference

### Vector Search: FAISS IndexFlatIP
- **Why**: Exact search with inner product (cosine similarity for normalized vectors)
- **Trade-off**: For >100k documents, approximate search (IVF) would be faster

### Keyword Search: BM25Okapi
- **Why**: Classic IR algorithm that catches exact term matches embeddings might miss
- **Example**: Query "RLHF" finds documents with that acronym even if the embedding model hasn't seen it

### Hybrid Search: Reciprocal Rank Fusion
- **Why**: Combines dense and sparse rankings without needing to tune weights
- **Formula**: `score = Σ(1 / (k + rank))` for each ranking list

### Reranking: Cross-Encoder
- **Why**: Bi-encoders encode query and document separately. Cross-encoders process them together, capturing deeper relevance signals
- **Trade-off**: Slower but more accurate. We retrieve 20 candidates, rerank to top 5

### LLM: Groq + Llama 3.1
- **Why**: Fast inference, generous free tier, good quality for RAG tasks
- **Models**: `llama-3.1-8b-instant` (default), `llama-3.1-70b-versatile` (better quality)

### Chunking: Semantic
- **Why**: Respects document structure (sections, paragraphs) rather than arbitrary character splits
- **Parameters**: ~500 tokens per chunk, 100 token overlap

## Project Structure

```
RAG_tutor/
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py   # PDF → structured pages
│   │   ├── chunker.py      # Pages → semantic chunks
│   │   └── indexer.py      # Chunks → FAISS + BM25 indexes
│   ├── retrieval/
│   │   ├── hybrid_search.py # Dense + sparse + RRF
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── chain/
│   │   └── rag_chain.py     # LangChain RAG pipeline
│   └── app.py               # Gradio UI
├── tests/                   # Unit tests
├── data/
│   ├── raw/                 # PDF files
│   └── index/               # Saved indexes
├── run.py                   # CLI entry point
└── requirements.txt
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Groq API key**:
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_key_here" > .env
   ```
   Get a free key at https://console.groq.com

3. **Add a PDF**:
   Place your PDF in `data/raw/`

## Usage

### Option 1: Gradio Web UI

```bash
python run.py app
```

Opens at http://localhost:7860 with:
- Chat interface for Q&A
- Slider to adjust number of retrieved chunks (k)
- Toggle to show/hide source citations
- Model selection dropdown
- Retrieval preview tab for debugging

To create a public URL:
```bash
python run.py app --share
```

### Option 2: Command Line

**Process PDF and build indexes**:
```bash
python run.py ingest
```

**Ask a question**:
```bash
python run.py query "What is RLHF?"
```

**With custom parameters**:
```bash
python run.py query "Explain transformers" --top-k 3
```

### Option 3: Python API

```python
from src.chain.rag_chain import RAGPipeline

pipeline = RAGPipeline("data/index", top_k=5)

# Ask a question
answer = pipeline.query("What is RLHF?")
print(answer)

# Get sources
for source in pipeline.get_sources():
    print(f"- {source['section']} (pages {source['pages']})")
```

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```
