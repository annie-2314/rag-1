# Text RAG Pipeline — Step-by-Step Guide

This document explains **`rag-pipeline.ipynb`** in plain language. Read it while you run the notebook cell by cell.

---

## What is RAG?

**RAG = Retrieval Augmented Generation**

1. You have **private data** (here: text from a web page).
2. You **search** that data for pieces related to a user question.
3. You give those pieces to an **LLM** as context.
4. The LLM **answers using that context** (not from memory alone).

```
User question
    → Embed question (vector)
    → Find similar text chunks in FAISS
    → Put chunks + question into prompt
    → Gemini writes answer
```

---

## Frameworks & libraries used

| Piece | Library / tool | Role |
|--------|----------------|------|
| Orchestration | **LangChain** | Chains, prompts, documents, vector store wrappers |
| LLM (chat) | **langchain-google-genai** → `ChatGoogleGenerativeAI` | Talks to **Google Gemini** |
| Embeddings | **GoogleGenerativeAIEmbeddings** | Turns text → numbers (vectors) |
| Text splitting | **langchain-text-splitters** → `RecursiveCharacterTextSplitter` | Chunking + overlap |
| Loading web pages | **langchain-community** → `WebBaseLoader` | HTML → `Document` |
| Vector database | **FAISS** (`faiss-cpu` + `langchain_community.vectorstores.FAISS`) | Fast similarity search in memory |
| Config / secrets | **python-dotenv** | Loads `.env` (API keys) |
| Combine step | **langchain-classic** → `create_stuff_documents_chain` | Puts all retrieved text into one prompt |

---

## Models used (defaults in notebook)

| Purpose | Default model | Notes |
|---------|---------------|--------|
| **Chat / answers** | `gemini-2.5-flash-lite` | Set via `GEMINI_MODEL` |
| **Embeddings** | `models/gemini-embedding-001` | Set via `GEMINI_EMBEDDING_MODEL` |
| **Max answer length** | 256 tokens | `GEMINI_MAX_OUTPUT_TOKENS` |

You need a **Gemini API key** in `.env`:

```env
GEMINI_API_KEY=your_key_here
```

The notebook copies it to `GOOGLE_API_KEY` for LangChain.

---

## Big picture: 7 steps

| Step | Name | What happens |
|------|------|----------------|
| 0 | Setup | Load `.env`, configure models and rate limits |
| 1 | **Load** | Download HTML from a URL → one big `Document` |
| 2 | **Chunking** | Split long text into smaller pieces (`chunk_size`) |
| 3 | **Overlapping** | Share characters between chunks (`chunk_overlap`) |
| 4 | **Embed** | Each chunk → vector via Gemini embeddings API |
| 5 | **Index** | Store vectors in **FAISS** |
| 6 | **Retrieve** | User query → find closest chunks |
| 7 | **Generate** | LLM answers using retrieved chunks only |

---

## Part A — Setup cells (before RAG)

These cells teach **LangChain + Gemini basics** (not full RAG yet).

### Cell: `load_dotenv()`

- Loads variables from `.env` into `os.environ`.
- So `GEMINI_API_KEY` is available later.

### Cell: Environment defaults

- Syncs `GEMINI_API_KEY` → `GOOGLE_API_KEY`.
- Sets chat model, token limit, and **embedding rate limits** (important on free tier):
  - `GEMINI_EMBED_MAX_CHUNKS=12` — only index first 12 chunks
  - `GEMINI_EMBED_API_PAUSE_SEC=1.05` — pause between embed calls
  - `GEMINI_EMBED_ON_429_WAIT=48` — wait if rate-limited

### Cell: Create `llm`

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_output_tokens=256)
```

This is the **generator** — it writes answers in Step 7.

### Cells: Prompt templates & chains (practice)

- `ChatPromptTemplate` — system + user messages.
- `prompt | llm` — simple chain.
- `StrOutputParser` — get plain string instead of message object.
- `JsonOutputParser` — structured JSON output.

**Takeaway:** LangChain chains are `step1 | step2 | step3`. RAG adds retrieval before the final LLM call.

---

## Part B — RAG pipeline (main workflow)

### Step 1 — Load

**Goal:** Get raw text into LangChain `Document` objects.

```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/docs/tutorials/llm_chain/")
document = loader.load()
```

- **`Document`** has:
  - `page_content` — the text
  - `metadata` — e.g. `source` URL, title

**Why load?** The model cannot read a whole website at once; we index pieces.

---

### Steps 2 & 3 — Chunking and overlapping

**Goal:** Break one huge page into searchable pieces without losing context at boundaries.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4500,      # max characters per chunk
    chunk_overlap=300,    # shared characters between neighbors
)
documents = text_splitter.split_documents(document)
documents = documents[:12]  # cap for API quota
```

| Term | Meaning |
|------|--------|
| **Chunking** | Cutting long text into smaller segments. |
| **chunk_size** | Max length of each segment (characters). |
| **chunk_overlap** | End of chunk N overlaps start of chunk N+1 so sentences are not cut in half. |
| **RecursiveCharacterTextSplitter** | Tries splits on `\n\n`, `\n`, space, then characters — good for generic text. |

**Analogy:** A book split into pages; overlap = each page repeats the last paragraph of the previous page.

---

### Step 4 — Embedding

**Goal:** Turn each text chunk into a **vector** (list of numbers) so similar meanings are **close** in math space.

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

Custom class `_RateLimitedGeminiEmbeddings`:

- Embeds **one chunk per API call** + sleep → avoids bursting Gemini free-tier limits.
- Retries on `429` / `RESOURCE_EXHAUSTED`.

| Term | Meaning |
|------|--------|
| **Embedding** | A fixed-size vector representing meaning of text. |
| **embed_documents** | Vectors for your indexed chunks. |
| **embed_query** | Vector for the user's question (used at search time). |

**Important:** Query and documents should use the **same embedding model** so comparison is valid.

---

### Step 5 — Index (FAISS)

**Goal:** Store all chunk vectors so we can search quickly.

```python
vectorstore = FAISS.from_documents(documents, embeddings)
```

| Term | Meaning |
|------|--------|
| **FAISS** | Facebook AI Similarity Search — library for nearest-neighbor search. |
| **Vector store** | Database of vectors + link back to original text. |
| **Index** | Building that store (happens once per notebook run). |

**In memory:** When you close the notebook, the index is gone unless you save it to disk (not done in this beginner notebook).

---

### Step 6 — Retrieve

**Goal:** Given a question, find the most relevant chunks.

```python
query = "This is a relatively simple LLM application"
results = vectorstore.similarity_search(query)
```

What happens inside:

1. Embed the **query** string → vector Q.
2. Compare Q to every chunk vector (distance / similarity).
3. Return top **k** `Document` objects (default k may be 4).

**You do not send the whole website to Gemini here** — only a few chunks.

---

### Step 7 — Generate

**Goal:** Answer the question using **only** retrieved text.

**7a — Prompt template**

```python
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>
{context}
</context>
""")
```

**7b — Stuff documents chain**

```python
document_chain = create_stuff_documents_chain(llm, prompt)
```

- **Stuff** = put all retrieved documents into `{context}` in one prompt.
- Then `llm` generates the answer.

**Flow:**

```
question → retriever (similarity_search) → context string → prompt → Gemini → answer
```

---

## Environment variables cheat sheet

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | API key |
| `GEMINI_MODEL` | Chat model name |
| `GEMINI_EMBEDDING_MODEL` | Embedding model |
| `GEMINI_MAX_OUTPUT_TOKENS` | Max length of answers |
| `GEMINI_EMBED_MAX_CHUNKS` | Limit chunks indexed |
| `GEMINI_EMBED_API_PAUSE_SEC` | Delay between embed calls |
| `GEMINI_EMBED_ON_429_WAIT` | Wait on rate limit |

---

## How to run the notebook

1. Create virtual env, `pip install -r requirements.txt`
2. Add `.env` with `GEMINI_API_KEY`
3. Open `rag-pipeline.ipynb`
4. **Run All** top to bottom (or run in order)
5. Wait on Step 4–5 (embedding many chunks calls the API)

---

## Common issues

| Problem | Cause | Fix |
|---------|--------|-----|
| 429 / quota | Too many embed calls | Lower chunks, increase pause, wait |
| Short answers | `max_output_tokens=256` | Raise in `.env` |
| Wrong context | Query unlike indexed text | Rephrase query; improve source data |
| Empty index | Skipped cells | Re-run from load through FAISS |

---

## One-sentence summary

**Load web text → split into overlapping chunks → embed with Gemini → store in FAISS → search by question → Gemini answers from retrieved chunks only.**

For the image version, see **[multimodal-rag-pipeline-explained.md](./multimodal-rag-pipeline-explained.md)**.
