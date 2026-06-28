# Module 2: Vector Search — Study Notes

**Course:** LLM Zoomcamp 2026
**Module:** 02 — Vector Search
**Purpose:** Long-term reference. Return to this before building any retrieval layer.

---

## Module Overview

Module 1 built a working RAG system using keyword search — fast and simple, but limited:
it only finds what you literally typed, not what you meant.

Module 2 fixes that. The core idea: convert text into numbers that encode meaning, then find
the numbers closest to your question. Two phrases that mean the same thing produce similar
numbers even when they share no words. This is semantic search.

### Where this fits in the RAG pipeline

```
[User Question]
       |
       v
[Retrieval Layer]  <-- THIS MODULE
  - Embed the question
  - Search the vector index (or text index, or both)
  - Return the most relevant chunks
       |
       v
[Context Assembly]
       |
       v
[LLM Generation]  (Groq / Llama 3.3 70B in our setup)
       |
       v
[Answer]
```

---

## Embeddings

An embedding is a list of numbers representing the meaning of a piece of text.

```
"How do I store vectors in PostgreSQL?"
→  [-0.12, 0.81, -0.34, 0.07, ..., 0.22]   # 384 numbers
```

The numbers are not random. A model trained on enormous amounts of text learned to place
sentences with similar meanings close together in a high-dimensional space, and sentences
with different meanings far apart.

Individual values have no human-interpretable meaning. Position 47 does not mean "database".
The meaning is distributed across all 384 values collectively.

**Why it works:** During training, the model saw that "store vectors in PostgreSQL" and
"save embeddings in pgvector" appear in similar contexts — documentation, tutorials, technical
articles. It adjusted its weights so both produce similar vectors.

---

## Why 384 Dimensions?

The model used here is `all-MiniLM-L6-v2`. 384 is a design trade-off:

- More dimensions → richer representations, but larger storage, slower computation
- Fewer dimensions → faster and cheaper, but less expressive

384 is where the designers found a good balance. Larger models like `all-mpnet-base-v2`
use 768 dimensions. The practical consequence: 295 chunks → a `(295, 384)` matrix.

---

## Cosine Similarity

Two vectors, one for the query and one for a document — how similar are they?

Cosine similarity measures the angle between two vectors:

```
Same direction    → similarity ≈  1.0   (same meaning)
Perpendicular     → similarity ≈  0.0   (unrelated)
Opposite          → similarity ≈ -1.0   (opposite meaning)
```

**Why the dot product works here:** The embedder L2-normalizes every vector (scales to
unit length). For normalized vectors, `dot product == cosine similarity`.

```python
similarity = query_vector.dot(doc_vector)
```

This is 384 multiplications and 383 additions — trivially fast. For 295 chunks, 295 of these.

**What 0.36 means in Q2:** The ANN query and the SQLite lesson are genuinely related but not
identical. The lesson covers many topics, so its embedding is diluted. 0.36 is moderate
positive similarity, correctly reflecting what the document actually contains.

---

## Chunking

**Chunking is often more important than the choice of embedding model.**

### Why whole-document embeddings are weak

Embedding a full document compresses all its topics into one vector — a weighted average of
everything. The result is pulled equally toward every topic in the document, making it a
weak match for any specific query.

### What chunking does

Split each document into smaller, overlapping windows. Each chunk gets its own focused
embedding representing only its own content.

```
Full document
  → Chunk 1: chars    0-2000  [embedding 1]
  → Chunk 2: chars 1000-3000  [embedding 2]   ← overlaps chunk 1 by 1000 chars
  → Chunk 3: chars 2000-4000  [embedding 3]
```

**The Q3 proof:** whole-document similarity was 0.36; best chunk similarity was 0.65.
Nearly double, with no change to the model or query.

### Overlap (step parameter)

`step=1000` with `size=2000` means each window slides 1000 chars, giving 1000-char overlap
between consecutive chunks. This ensures no passage is silently cut in half at a boundary.

### The trade-off

| Smaller chunks | Larger chunks |
|---|---|
| More focused embeddings | Richer context per chunk |
| More precise retrieval | More diluted embeddings |
| More chunks to store | Fewer chunks to store |
| Less context for LLM | More noise in LLM context |

Right chunk size is domain-dependent — determine by evaluation, not intuition.

---

## Vector Search

### Brute-force

```python
scores = X.dot(query_vector)   # X is (295, 384), result is (295,)
best_idx = scores.argmax()
```

One line. That's the entire search. For 295 chunks this is instant. For 295 million chunks,
this is where ANN libraries (FAISS, Qdrant, pgvector) become necessary.

### What VectorSearch adds

`minsearch.VectorSearch` handles the matrix algebra, manages the payload (chunk dicts), and
returns ranked results with metadata. Clean interface: you provide vectors, it searches.
Swap in any embedding model without touching search code.

---

## Text Search vs Vector Search

| Property | Text Search | Vector Search |
|---|---|---|
| Matching | Exact word overlap (TF-IDF) | Semantic similarity (dot product) |
| Vocabulary | Must match index terms | Handles synonyms naturally |
| Exact terms | Excellent | Can miss rare/domain terms |
| Speed | Very fast | Fast for small corpora; needs ANN at scale |
| Infrastructure | Minimal | Requires embedding model + vector store |

**When to use text search:** Users know exact terminology — error codes, product names,
legal citations, technical identifiers.

**When to use vector search:** Users might phrase queries differently from how content is
written. "How do I save embeddings?" needs to find docs that say "store vectors in pgvector."

**The Q5 demonstration:** Searching "How do I store vectors in PostgreSQL?" — text search
missed the pgvector lesson because it uses different vocabulary. Vector search found it first.

**Key principle:** Neither method is universally superior. The right choice depends on your
data and users' query patterns — measure with evaluation (Module 4).

---

## Hybrid Search and Reciprocal Rank Fusion (RRF)

### The motivation

Text search is good at exact terms. Vector search is good at meaning. Hybrid search asks:
which documents do both methods agree are relevant?

### How RRF works

For each document in each result list:
```
score += 1 / (k + rank)    # rank starts at 0, k=60 default
```

- Raw similarity scores are discarded — only position (rank) matters
- This solves the scale incompatibility between TF-IDF weights and cosine similarities
- A document appearing in two lists accumulates contributions from both
- `k=60` smooths the curve — from the original 2009 RRF paper, works well across most tasks

```
rank 0 → 1/(60+0) = 0.0167
rank 4 → 1/(60+4) = 0.0156
appears in 2 lists → both contributions added
```

### The Q6 insight

For "How do I give the model access to tools?" — `13-function-calling.md` was 2nd in text
search and 5th in vector search. It was the only chunk that ranked high in both lists. RRF
placed it first. Consensus beat individual dominance.

**Do not tune k=60 without an evaluation set.** Tuning on a single query is overfitting.

---

## ONNX Runtime

**Development vs production gap:**
`sentence-transformers` is convenient but requires PyTorch (~4.8 GB).
ONNX Runtime produces identical vectors at ~147 MB. No PyTorch. No CUDA. Runs anywhere.

**Why this matters:**
- Smaller Docker images → faster CI/CD, lower storage costs
- Runs on any CPU — no GPU, no driver compatibility issues
- Fewer dependencies → simpler security auditing

**The lesson:** Development tooling values ergonomics. Production tooling values minimalism.
They are different concerns. Design for the transition between them.

---

## Key Takeaways

- Embeddings encode semantic meaning as lists of numbers — not literal words
- Similar meanings → vectors pointing in similar directions
- For normalized vectors, `dot product == cosine similarity`
- 384 dimensions is a quality-vs-cost design choice; individual dimensions have no meaning
- Chunking before embedding dramatically improves retrieval — focused chunks beat diluted docs
- Overlapping chunks prevent passages from being silently split at boundaries
- Brute-force vector search is `X.dot(query_vector)` — that's it
- Text search matches words; vector search matches meaning; each has strengths
- Hybrid search (RRF) combines ranked lists, not raw scores
- RRF finds documents that consistently rank well across methods — consensus over dominance
- ONNX Runtime = same embeddings as sentence-transformers, 30x smaller deployment
- Which retrieval method is best is an empirical question answered by evaluation, not intuition
