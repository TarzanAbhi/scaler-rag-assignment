# scaler-sage — Design Document
*RAG-powered Q&A system for Scaler's learner support knowledge base*

---

## The Problem

Scaler's learner support team handles hundreds of repetitive queries daily — about 
refund policies, eligibility, payment plans, and placement guarantees. The answers 
exist across PDFs, FAQs, and structured documents, but finding them is slow and 
inconsistent.

This system lets anyone ask a natural language question and get a grounded, cited 
answer in seconds — without a support agent in the loop.

---

## System Architecture

The pipeline has two phases: **ingestion** (runs once) and **query** (runs on demand).
INGESTION
─────────
corpus/ (PDFs, TXT, JSON)
↓
Parse — extract raw text per document type
↓
Chunk — split into 400-token segments with 80-token overlap
↓
Embed — convert each chunk to a vector (all-MiniLM-L6-v2)
↓
Store — persist vectors + metadata to ChromaDB
QUERY
─────
User question
↓
Embed query — same model as ingestion
↓
Retrieve — cosine similarity search, top-5 chunks
↓
Threshold gate — similarity < 0.3 → return no-match, skip LLM
↓
Generate — GPT-4o-mini with context-only system prompt
↓
Trace — emit structured log to traces.jsonl
↓
Answer + source citation → user

---

## Chunking Strategy

**Chunk size: 400 tokens. Overlap: 80 tokens.**

The size was chosen deliberately. Too small (under 100 tokens) and a chunk loses 
its context — a sentence like "No refund after this period" means nothing without 
the sentence before it that says which period. Too large (over 700 tokens) and 
retrieval becomes imprecise — you pull in three topics when you needed one.

400 tokens is roughly one complete idea: a policy clause, a FAQ answer, a pricing 
entry with its explanation.

The 80-token overlap handles boundary cases. If an answer straddles two chunks, 
the overlapping region ensures at least one chunk contains the full thought.

**Document-type handling:**

| Type | Strategy | Reason |
|---|---|---|
| PDF | Sliding window by token count | Preserves narrative flow |
| TXT | Sliding window with overlap | Same as PDF |
| JSON / YAML | One top-level entry per chunk | A pricing row split mid-entry is meaningless |

Every chunk is stored with metadata: `source`, `chunk_index`, `doc_type`. This 
metadata drives filtering and debugging — when retrieval fails, the trace shows 
exactly which chunks were pulled and from where.

---

## Retrieval Design

**Embedding model: `all-MiniLM-L6-v2` (local, via sentence-transformers)**

A local model was chosen over a hosted API for a specific reason: it eliminates 
a class of failure. No API key, no quota, no network dependency. The model downloads 
once (~80MB) and runs entirely on the machine. For a corpus of 15-20k tokens, its 
quality is more than sufficient.

The same model is used for both ingestion and query. This is not optional — mixing 
embedding models means the vector spaces don't align and similarity scores become 
meaningless.

**Vector store: ChromaDB (local persistent)**

ChromaDB was chosen because it requires no server, no Docker, and no configuration. 
It installs with pip and persists the index to disk automatically. The index survives 
process restarts, which means `--ingest` runs once and every subsequent query is fast.

At production scale (millions of chunks), ChromaDB would be replaced with Pinecone 
or Weaviate. For this corpus, it is more than sufficient.

**Similarity metric: cosine**

Cosine similarity measures the angle between vectors, not their magnitude. A short 
chunk and a long chunk about the same topic score similarly against a matching query. 
Euclidean distance would penalise the longer chunk just for being longer. For 
semantic text search, cosine is always the right choice.

**Similarity threshold: 0.3**

Below 0.3, the retrieved chunks are essentially unrelated to the query. Passing them 
to the LLM is actively harmful — the model will synthesise a plausible-sounding answer 
from irrelevant content rather than admitting it doesn't know. The threshold gate 
prevents this and returns an honest no-match response instead.

---

## Answer Generation

**Model: GPT-4o-mini. Temperature: 0.**

This is a synthesis task, not a reasoning task. The LLM reads five chunks and writes 
a coherent answer. That does not require a frontier model. GPT-4o-mini is fast, cheap, 
and more than capable for this use case.

Temperature 0 makes the system deterministic. The same question should return the 
same answer every time. Randomness serves no purpose in factual Q&A.

**System prompt:**
You are a helpful assistant for Scaler's learner support team.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer
confidently, say exactly:
"I don't have enough information to answer this accurately
based on the available documents."
Do not make up facts. Do not use prior knowledge.
At the end of your answer, cite the source document(s) like this:
Sources: [filename.pdf, filename.txt]

The phrase "ONLY the context provided" is load-bearing. Without it, the LLM draws 
on its training knowledge to fill gaps — and will confidently state a plausible but 
wrong refund policy based on what edtech companies generally do. The constraint 
forces grounding.

The exact no-match phrase is also intentional. It gives the eval script a reliable 
signal to check programmatically.

**No-match handling:**

If the top retrieved chunk has similarity below 0.3, the LLM call is skipped entirely. 
The system returns a fixed response immediately. This saves cost and prevents 
hallucination in the same step.

---

## Instrumentation

Every query emits a structured trace to `traces.jsonl` (one JSON line per query):

```json
{
  "timestamp": "2026-07-14T10:00:00",
  "query": "What is the refund policy?",
  "latency_ms": 1243.5,
  "top_similarity": 0.87,
  "top_k_chunks": [
    { "source": "refund_policy.txt", "similarity": 0.87, "text_preview": "..." }
  ],
  "no_match_triggered": false,
  "prompt_tokens": 512,
  "completion_tokens": 180,
  "prompt_preview": "Context:\nSource: refund_policy.txt...",
  "answer_preview": "Scaler offers a full refund if..."
}
```

The most important field for debugging is `top_similarity`. It immediately tells 
you which layer failed:

- **Low similarity + wrong answer** → retrieval failure. Check chunking or whether 
  the document was ingested.
- **High similarity + wrong answer** → generation failure. Check the system prompt 
  or whether the right chunks are in the top-5.

These are two completely different failure modes with different fixes. Without 
similarity scores in the trace, you cannot tell them apart.

---

## Evaluation

The eval suite runs 10 test cases covering every document in the corpus, plus one 
deliberate out-of-corpus query to test the no-match path.

Two automated metrics are computed per query:

**1. Keyword check (deterministic)**
Fast and free. Confirms the answer contains expected terms. Catches obvious failures 
without an LLM call.

**2. Faithfulness judge (LLM)**
An LLM reads the retrieved context and the generated answer and determines whether 
the answer contradicts or extrapolates beyond the context. This is the core RAG 
failure mode — hallucination — and keyword matching cannot detect it.

**3. Context precision judge (LLM)**
Checks whether the retrieved chunks were actually relevant to the question. A 
faithful answer from the wrong chunks is a retrieval problem, not a generation 
problem. This metric separates the two.

**Known limits:**
- The LLM judge can itself hallucinate. Its output is a signal, not ground truth.
- 10 test cases cannot cover every query pattern in a real support context.
- There is no human-labelled ground truth to calibrate against. In production, 
  a sample of judge outputs would be reviewed by a human grader and disagreements 
  would become regression test cases.

---

## What I would do with more time

**1. Hybrid retrieval**

The current system uses dense retrieval only (embeddings). For keyword-heavy queries 
like "EMI option" or "refund after 30 days", sparse retrieval (BM25) outperforms 
embeddings because it matches exact terms. Combining both — hybrid retrieval with 
score fusion — consistently outperforms either approach alone. This would be the 
first production improvement.

**2. Cross-encoder reranker**

Cosine similarity is a fast approximation of relevance. A cross-encoder reranker 
reads the query and each chunk together and scores them more accurately. The pattern 
is: retrieve top-20 by cosine, rerank to top-5, pass top-5 to the LLM. This 
significantly improves precision, especially for longer or more specific queries.

**3. Query rewriting**

Raw user queries are often poorly formed. "what about money back" does not embed 
close to "refund and cancellation policy". An LLM rewrite step before retrieval — 
one extra API call — improves recall significantly for colloquial or ambiguous 
questions. For a learner support context where questions come from stressed students, 
this matters.

**4. Learner-aware retrieval**

The system currently retrieves the same chunks for every user. In Scaler's context, 
a Week 2 learner asking about recursion should not receive Week 8 advanced content. 
Filtering retrieved chunks by learner metadata (current module, known weak areas) 
would make the system genuinely personalised rather than just semantically accurate.

---