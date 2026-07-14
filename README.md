# Scaler Sage 🌿

> A Retrieval-Augmented Generation (RAG) system for Scaler's learner support knowledge base.

Builds a semantic search index over PDFs, text files, and structured documents to provide grounded, source-backed answers using Gemini.

---

## Features

- Multi-format document ingestion (PDF, TXT, JSON, YAML)
- Intelligent document chunking with overlap
- Semantic search using Gemini Embeddings
- Persistent ChromaDB vector store
- Grounded answer generation with Gemini
- Source citations for retrieved context
- Query tracing and observability
- Automated evaluation pipeline
- Hallucination prevention using similarity thresholding

---

## Setup

```bash
# Clone the repository
git clone https://github.com/TarzanAbhi/scaler-rag-assignment.git
cd scaler-rag-assignment

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key
export SCALER_LLM_API_KEY="your_api_key"

# Add documents to corpus/
# Supported formats: PDF, TXT, JSON, YAML

# Build the vector index
python src/main.py --ingest
```

---

## Usage

```bash
# Ask a single question
python src/main.py --query "What is the refund policy?"

# Interactive CLI
python src/main.py

# Rebuild the index after adding documents
python src/main.py --ingest
```

---

## Project Structure

```text
scaler-rag-assignment/
│
├── corpus/                  # Knowledge base documents
├── src/
│   ├── config.py            # Configuration constants
│   ├── ingest.py            # Parse → Chunk → Embed → Store
│   ├── retrieve.py          # Semantic retrieval
│   ├── generate.py          # Prompt construction + Gemini
│   ├── trace.py             # Structured tracing
│   └── main.py              # CLI entry point
│
├── eval/
│   ├── test_cases.json
│   └── run_eval.py
│
├── index/                   # Persistent ChromaDB index
├── traces.jsonl             # Query traces
├── DESIGN.md                # Architecture and design decisions
├── README.md
└── requirements.txt
```

---

## System Pipeline

```text
                    Documents
                        │
                        ▼
          Parse (PDF / TXT / JSON / YAML)
                        │
                        ▼
       Chunk (400 tokens, 80-token overlap)
                        │
                        ▼
       Embed (gemini-embedding-001)
                        │
                        ▼
      ChromaDB Vector Store (Cosine Similarity)
                        │
                        ▼
        Query → Embed → Retrieve Top-K Chunks
                        │
                        ▼
        Similarity Threshold (< 0.3 → No Match)
                        │
                        ▼
 Generate (Gemini 2.0 Flash, Temperature = 0)
                        │
                        ▼
        Grounded Answer + Source Citation + Trace
```

---

## Evaluation

```bash
python eval/run_eval.py
```

The evaluation suite executes test queries against the corpus and measures:

- **Keyword Match** – verifies expected concepts appear in the generated answer.
- **Faithfulness** – checks whether the answer is grounded in the retrieved context using an LLM-as-a-judge approach.
- **Context Precision** – evaluates whether the retrieved chunks are relevant to the user's query.

Results are written to:

```text
eval/results.json
```

---

## Query Tracing

Every query produces a structured trace in `traces.jsonl`.

Example:

```json
{
  "timestamp": "2026-07-14T10:00:00",
  "query": "What is the refund policy?",
  "latency_ms": 1243.5,
  "top_similarity": 0.87,
  "top_k_chunks": [
    {
      "source": "refund_policy.txt",
      "similarity": 0.87,
      "text_preview": "..."
    }
  ],
  "no_match_triggered": false,
  "prompt_tokens": 512,
  "completion_tokens": 180,
  "answer_preview": "..."
}
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunk Size | 400 tokens | Preserves a complete semantic unit while maintaining retrieval precision |
| Chunk Overlap | 80 tokens | Prevents loss of context at chunk boundaries |
| Embedding Model | `gemini-embedding-001` | High-quality semantic embeddings with a single API provider |
| Vector Store | ChromaDB | Lightweight, persistent, and serverless |
| Similarity Metric | Cosine Similarity | Standard metric for semantic text retrieval |
| Retrieval | Top-K = 5 | Balances recall and context size |
| Similarity Threshold | 0.3 | Prevents hallucinations from weak retrievals |
| Generation Model | Gemini 2.0 Flash | Fast, cost-efficient, and well suited for grounded Q&A |
| Temperature | 0 | Deterministic and reproducible responses |

---

## Edge Cases

The system gracefully handles:

- Empty user queries
- Empty document corpus
- Unsupported or malformed documents
- PDF extraction failures
- Out-of-domain questions
- Low-confidence retrievals using similarity thresholding
- Missing context without hallucinating answers

---

## Technology Stack

- **Python**
- **Google Gemini API**
- **ChromaDB**
- **PyPDF2**
- **tiktoken**
- **FastAPI**
- **sentence-transformers** *(optional for local embeddings)*

---

## Production Improvements

Given additional time, the following enhancements would make the system production-ready:

- Hybrid Retrieval (Dense + BM25)
- Cross-Encoder Reranking
- PostgreSQL + pgvector / Pinecone / Qdrant
- Metadata-aware retrieval
- Semantic chunking
- Incremental document indexing
- Redis caching
- Kafka-based asynchronous ingestion
- Langfuse & OpenTelemetry observability
- Streaming LLM responses
- Role-based access control
