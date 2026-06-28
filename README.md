# LLM Zoomcamp 2026

My learning repo for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
homework/
├── module_01/
│   └── notebook.ipynb        ← Module 01: Agentic RAG (Groq API)
└── module_02/
    ├── download.py           ← Download ONNX model (run once)
    ├── embedder.py           ← Lightweight ONNX embedder class
    └── vector_search.ipynb   ← Module 02: Vector Search homework
```

## Modules

### Module 01 — Agentic RAG
Basic RAG pipeline using keyword search + Groq LLM (Llama 3.3 70B).

### Module 02 — Vector Search
Semantic retrieval using ONNX embeddings, chunking, VectorSearch, and hybrid RRF search.

```bash
# One-time model download (saves ~90 MB to homework/module_02/models/)
cd homework/module_02
python download.py
```

## LLM Provider
Uses [Groq](https://console.groq.com) free tier via the OpenAI-compatible API.
Set your key in `.env`:
```
GROQ_API_KEY=your_key_here
```
