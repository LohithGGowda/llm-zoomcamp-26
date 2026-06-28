# Homework 2: Vector Search — Solutions & Notes

**Course:** LLM Zoomcamp 2026 — Module 2
**Full notebook:** `vector_search.ipynb`
**Course material:** https://github.com/DataTalksClub/llm-zoomcamp/tree/main/02-vector-search

---

## Setup

Uses pip + venv with the ONNX-based embedder instead of sentence-transformers.
Same vectors, no PyTorch, no CUDA dependency — runs anywhere.

```bash
pip install -r requirements.txt

# Download the model once (saves ~90 MB to models/)
cd homework/module_02
python download.py
```

The download fetches `Xenova/all-MiniLM-L6-v2` — the ONNX version of the model used in the module lessons.

---

## Q1. Embedding a Query

**Answer: -0.02**

### Why

Embedding converts a text string into a fixed-length vector of numbers. The shape `(384,)`
confirms the correct model was loaded — `all-MiniLM-L6-v2` always produces 384-dimensional
vectors. The first value is a deterministic fingerprint: same model, same input, same output,
every time.

### How

```python
from embedder import Embedder

embed = Embedder()

query = "How does approximate nearest neighbor search work?"
query_vector = embed.encode(query)

print(f"Embedding shape: {query_vector.shape}")   # (384,)
print(f"First value: {query_vector[0]}")           # -0.02058203437252893
```

### What to remember

Individual embedding values have no standalone meaning. Only the complete vector, compared
against other vectors, encodes semantic information. Shape and first value checks are
pipeline sanity checks, not insight into the embedding itself.

---

## Q2. Cosine Similarity

**Answer: 0.37**

### Why

The embedder returns normalized vectors (each scaled to unit length). For normalized vectors,
the dot product equals cosine similarity. A similarity of ~0.36 between the ANN query and the
full SQLite vector search lesson reflects genuine but diluted relevance — the lesson covers
vector search concepts but also SQLite tables, SQL queries, and persistence details, all of
which pull the embedding away from the specific ANN question.

### How

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]

sqlite_doc = next(
    doc for doc in documents
    if doc["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
)

doc_vector = embed.encode(sqlite_doc["content"])
similarity = query_vector.dot(doc_vector)

print(similarity)   # 0.36107027225589694
```

### What to remember

This builds intuition before retrieval at scale. At search time, this same dot product is
computed between the query and every document in the index. The highest dot product wins.

---

## Q3. Chunking and Search by Hand

**Answer: `02-vector-search/lessons/07-sqlitesearch-vector.md`**

### Why

The full-page embedding gave similarity ~0.36. After chunking, the best chunk from the same
lesson gave ~0.65 — nearly double, with no change to the embedding model or query. Chunking
creates smaller, focused semantic units. The relevant passage gets its own vector, undiluted
by the rest of the document.

The winning chunk was chunk index 94, starting at character position 1000 within that lesson.

### How

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
print(len(chunks))   # 295

texts = [chunk["content"] for chunk in chunks]
X = embed.encode_batch(texts)

print(X.shape)   # (295, 384)

scores = X.dot(query_vector)

print(scores.max())   # 0.6489017718578813

best_idx = scores.argmax()

print(best_idx)                          # 94
print(chunks[best_idx]["filename"])      # 02-vector-search/lessons/07-sqlitesearch-vector.md
print(chunks[best_idx]["start"])         # 1000
```

### What to remember

`X.dot(query_vector)` is the entire brute-force vector search operation — a matrix
multiplication producing one similarity score per chunk simultaneously. `argmax()` returns
the index of the highest score.

The `step=1000` parameter means consecutive chunks overlap by 1000 characters, preventing
relevant passages from being silently split across chunk boundaries.

---

## Q4. Vector Search with minsearch

**Answer: `04-evaluation/lessons/05-search-metrics.md`**

### Why

`VectorSearch` from minsearch wraps the matrix multiplication into a proper search interface:
fit once, search many times, return ranked results with payloads attached. The query
"What metric do we use to evaluate a search engine?" semantically matched the lesson literally
titled "Search Evaluation Metrics" — which covers Hit Rate and MRR.

### How

```python
from minsearch import VectorSearch

vector_index = VectorSearch()
vector_index.fit(X, chunks)

query = "What metric do we use to evaluate a search engine?"
query_vector = embed.encode(query)

results = vector_index.search(query_vector)

print(results[0]["filename"])   # 04-evaluation/lessons/05-search-metrics.md
```

### What to remember

`VectorSearch.search()` takes a query vector, not raw text. The library handles indexing and
ranking; the embedding model handles text-to-vector conversion. You can swap either component
independently — this separation of concerns is what makes retrieval systems maintainable.

---

## Q5. Text Search vs Vector Search

**Answer: `02-vector-search/lessons/08-pgvector.md`**

### Why

Keyword search matched words — "vectors", "PostgreSQL", "store" — in documents that contain
those terms. The pgvector lesson uses different vocabulary: "pgvector", "embeddings",
"pg_vector extension". Vector search matched meaning: both the query and the pgvector lesson
discuss the same concept regardless of exact phrasing.

This is the vocabulary mismatch problem.

### How

```python
from minsearch import Index

text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

query = "How do I store vectors in PostgreSQL?"

text_results = text_index.search(query=query, num_results=5)
query_vector = embed.encode(query)
vector_results = vector_index.search(query_vector, num_results=5)

print("TEXT SEARCH")
for result in text_results:
    print(result["filename"])
# 02-vector-search/lessons/02-embeddings.md
# 03-orchestration/lessons/05-rag.md
# 02-vector-search/lessons/01-intro.md  (pgvector NOT here)

print("VECTOR SEARCH")
for result in vector_results:
    print(result["filename"])
# 02-vector-search/lessons/08-pgvector.md  <-- found here
```

### What to remember

Text search is not wrong — it worked correctly given its design. The pgvector lesson simply
uses different words to describe the same concept. Neither method is universally better.

---

## Q6. Hybrid Search with RRF

**Answer: `01-agentic-rag/lessons/13-function-calling.md`**

### Why

For "How do I give the model access to tools?", the function-calling lesson ranked 2nd in
text search and 5th in vector search — first in neither. But it was the only chunk that
appeared high in both lists. RRF surfaced it as #1 because the consensus signal outweighed
any single method's top result.

### How

```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


query = "How do I give the model access to tools?"

text_results = text_index.search(query, num_results=5)
query_vector = embed.encode(query)
vector_results = vector_index.search(query_vector, num_results=5)

results = rrf([vector_results, text_results])

for i, doc in enumerate(results):
    print(i + 1, doc["filename"], doc["start"])

# 1  01-agentic-rag/lessons/13-function-calling.md   4000
# 2  01-agentic-rag/lessons/01-intro.md              2000
# 3  01-agentic-rag/lessons/14-agentic-loop.md       0
```

### What to remember

RRF uses ranks, not raw scores. TF-IDF weights and cosine similarities are not on comparable
scales and cannot be summed directly. RRF sidesteps this by working only with position.
`k=60` is the paper's default — do not tune it without a proper evaluation set.

---

## Summary

| Method | Strong when | Weak when |
|---|---|---|
| Text search | Users know exact terminology; error codes, names, IDs | Users paraphrase; synonyms; different vocabulary |
| Vector search | Vocabulary mismatch between queries and documents | Rare exact terms the model may not embed well |
| Hybrid (RRF) | You want the benefits of both | You need to tune for very specific query distributions |

The right method is always determined by evaluation (Module 4), not intuition.

---

*Full notebook: [vector_search.ipynb](vector_search.ipynb)*
