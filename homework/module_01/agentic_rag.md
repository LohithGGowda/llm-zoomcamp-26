# Module 1: Agentic RAG — Notes & Solutions

**Course:** LLM Zoomcamp 2026 — Module 1
**Full notebook:** `notebook.ipynb`
**Course material:** https://github.com/DataTalksClub/llm-zoomcamp/tree/main/01-agentic-rag

---

## What This Module Builds

A working RAG (Retrieval-Augmented Generation) pipeline from scratch:

```
[User Question]
       |
       v
[Search Index]  ← keyword search over FAQ documents (minsearch)
       |
       v
[Context Builder]  ← formats top results into a readable prompt
       |
       v
[LLM]  ← Groq API, Llama 3.3 70B
       |
       v
[Answer]
```

No frameworks. No magic. Every component is written explicitly so you understand what
each piece does before reaching for a library.

---

## Setup

Uses the Groq API (free tier) via the OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your_groq_key",
    base_url="https://api.groq.com/openai/v1"
)
```

Set your key in `.env` at the repo root:
```
GROQ_API_KEY=your_key_here
```

Model used: `llama-3.3-70b-versatile` — Groq's free-tier production model.

---

## Components

### 1. LLM Wrapper

```python
def llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

Wraps the raw API call into a simple function. Takes a prompt string, returns a string.

**Why this pattern:** Centralises the model name and API call. To switch models, change
one line. To add system instructions, modify one function.

---

### 2. Data Loading

FAQ data from the DataTalks.Club website:

```python
import requests

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"{url_prefix}{course['path']}"
    course_data = requests.get(course_url).json()
    documents.extend(course_data)
```

Each document has: `question`, `answer`, `section`, `course`.

---

### 3. Search Index

```python
from minsearch import Index

index = Index(
    text_fields=['question', 'section', 'answer'],
    keyword_fields=['course']
)
index.fit(documents)
```

`text_fields` are TF-IDF searched. `keyword_fields` are exact-match filtered.

```python
def search(question, course='llm-zoomcamp'):
    return index.search(
        question,
        boost_dict={'question': 2.0, 'section': 0.5},
        filter_dict={'course': course},
        num_results=5
    )
```

`boost_dict` gives the `question` field twice the weight of `answer`, and `section` half.
This reflects the intuition that matching the question field is more relevant than matching
the answer body.

---

### 4. Context Builder

```python
def build_context(search_results):
    lines = []
    for doc in search_results:
        lines.append(doc['section'])
        lines.append('Q: ' + doc['question'])
        lines.append('A: ' + doc['answer'])
        lines.append('')
    return '\n'.join(lines).strip()
```

Formats search results into a readable block the LLM can use as context.

---

### 5. Prompt Builder

```python
INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

USER_PROMPT_TEMPLATE = '''
Question:
{question}

Context:
{context}
'''

def build_prompt(question, search_results):
    context = build_context(search_results)
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    ).strip()
```

**Why separate INSTRUCTIONS from the user prompt:** INSTRUCTIONS describe the task and
constraints — they don't change between requests. The user prompt carries the specific
question and retrieved context — it changes every request. Keeping them separate makes it
easy to update the task description without touching the per-request logic.

---

## The Full Pipeline

```python
question = "I just discovered the course. Can I join now?"

search_results = search(question)
prompt = build_prompt(question, search_results)
answer = llm(prompt)

print(answer)
```

Five lines. That's RAG.

---

## What Went Wrong (and Why)

The notebook has cells with `input=prompt` instead of `messages=[...]`:

```python
# WRONG — this is not how the chat completions API works
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    input=prompt          # ← no such parameter
)
```

The correct call:
```python
# CORRECT
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content
```

`input=` and `output_text` are from the newer Responses API (OpenAI), not the chat
completions endpoint. Groq uses the chat completions interface.

---

## Key Takeaways

- RAG grounds LLM answers in retrieved facts, reducing hallucination
- `minsearch.Index` builds a lightweight keyword search engine with no external dependencies
- `boost_dict` controls field weights — questions matching the `question` field are more
  relevant than questions that only appear in the answer body
- `filter_dict` narrows results to one course — prevents documents from other courses
  contaminating results
- The LLM is the last step, not the first — retrieval quality determines answer quality
- The prompt structure matters: clear instructions + structured context = better answers
- Groq uses the chat completions API — `messages=[{"role": "user", "content": ...}]`

---

*Course material: [01-agentic-rag](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/01-agentic-rag)*
