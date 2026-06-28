# Module 2 (Vector Search) — FAQ

Questions that came up while working through the Module 2 homework.
None of these touch graded answers — just the concepts and reasoning behind each step.

---

### Why does the homework print the embedding shape and first value?

The goal is not to inspect the embedding itself but to verify the pipeline is working.

Printing `(384,)` confirms the right model loaded — `all-MiniLM-L6-v2` always produces
384-dimensional vectors.

Printing the first value is a deterministic fingerprint. Same model + same input = same
output, always. If it changes unexpectedly, it usually means a different model, tokenizer,
or preprocessing pipeline is running.

Individual embedding values have no human-interpretable meaning.

---

### Why are we only comparing the query with one document in Q2?

Q2 is not retrieval — it is model validation. Comparing one query against one known document
builds intuition for what cosine similarity means before scaling to 295 chunks.

Retrieval systems repeat this same comparison against every document in the corpus and rank
by score. Q2 shows you the unit operation; Q3 shows it applied at scale.

---

### Why wasn't the cosine similarity in Q2 close to 1?

Because the query and the document are related but not the same thing.

The query asks specifically about approximate nearest neighbor search. The SQLite vector
search lesson discusses related ideas but spends most of its length on SQLite tables, SQL
syntax, persistence, and implementation details. Embedding the whole document compresses all
those topics into one vector — a weighted average pulled away from the specific ANN concept.

~0.36 correctly reflects what the document actually contains.

---

### Why does chunking improve retrieval?

Embedding a full document forces the model to compress many different topics into one vector.
The result is a blurred average — a weak match for any specific question.

Chunking breaks documents into smaller, focused pieces. Each chunk gets its own embedding
representing only its own content. The query is then compared against many specialized
vectors instead of one general-purpose one.

Q3 makes this concrete: whole-document similarity was ~0.36; best chunk was ~0.65.
That improvement came entirely from better document preparation — no model change, no query change.

---

### Why chunk before embedding rather than after?

Embedding happens per unit of text. If you embed first, you already have one vector per whole
document, and you cannot split that vector after the fact.

Chunking must happen on raw text before embedding, so each chunk becomes its own independent
vector. The pipeline is always:

```
Raw document
  → Chunk 1 text  →  Embedding 1
  → Chunk 2 text  →  Embedding 2
  → Chunk 3 text  →  Embedding 3
```

---

### Why do we use argmax() after computing similarities?

`X.dot(query_vector)` returns one similarity score per chunk — a vector of 295 numbers.
`argmax()` returns the index of the highest score, identifying which chunk is most similar
to the query.

This is the entire core of brute-force vector search:
1. Compute similarities between the query and every chunk
2. Find the highest score
3. Return the corresponding chunk

Vector search libraries automate this and add efficiency at scale, but the operation is identical.

---

### Why does VectorSearch.search() take a vector instead of raw text?

`VectorSearch` is only responsible for indexing and ranking vectors. It does not know how to
convert text into embeddings, and it should not — that is the embedding model's job.

This separation keeps the system flexible. You can use any embedding model (ONNX,
sentence-transformers, OpenAI, a domain-specific model) as long as it produces vectors of
the correct dimension. Switching embedding models does not require touching the search code.

---

### Why did vector search find the pgvector lesson but text search missed it?

The query asked "How do I store vectors in PostgreSQL?" Keyword search found documents
containing those exact words. The pgvector lesson uses different vocabulary — "pgvector",
"pg_vector extension", "storing embeddings" — so keyword search missed it.

Vector search converted both the query and the lesson into embeddings capturing their meaning.
Because they discuss the same concept, their vectors pointed in similar directions.

This is the vocabulary mismatch problem:
- Text search matches words
- Vector search matches meaning

---

### Why does the same filename appear multiple times in vector search results?

Vector search runs over chunks, not whole documents. A long document is split into overlapping
chunks, each with its own embedding. Multiple chunks from the same file can all be highly
relevant and appear separately in the results, each with a different `start` value.

This is expected behaviour. The RAG system later decides how many chunks from the same
document to include in the prompt.

---

### Why does RRF often outperform either search method individually?

RRF asks a different question than either individual method. Instead of "which document
scored highest?", it asks: "which documents consistently appear near the top across methods?"

A document that ranks well in both text search and vector search receives contributions from
both ranked lists. A document strong in only one list collects just one contribution.

RRF also sidesteps incomparable scores. Text search and vector search produce raw scores
on completely different scales. Rather than normalising or weighting them, RRF discards the
raw scores entirely and works only with rank positions — which are always comparable.

The Q6 result: `13-function-calling.md` ranked 2nd in text and 5th in vector. It was the
only chunk that ranked high in both lists, so RRF placed it first overall.

---

### Why use pip instead of uv?

Personal preference and toolchain familiarity. Both install the same packages from PyPI.
The `requirements.txt` file covers all dependencies used across modules.

`uv` is faster for dependency resolution but requires an additional tool installation.
`pip` + `venv` works everywhere Python is installed, with no extra setup.
