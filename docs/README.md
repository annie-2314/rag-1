# RAG Practice — Documentation

Step-by-step guides for the notebooks in this repo:

| Notebook | Guide |
|----------|--------|
| [rag-pipeline.ipynb](../rag-pipeline.ipynb) | [rag-pipeline-explained.md](./rag-pipeline-explained.md) — **text RAG** (web page → chunks → FAISS → Gemini) |
| [multimodal-rag-pipeline.ipynb](../multimodal-rag-pipeline.ipynb) | [multimodal-rag-pipeline-explained.md](./multimodal-rag-pipeline-explained.md) — **image RAG** (`data/multimodal/` → vision → FAISS → Gemini) |

## Quick start

1. `pip install -r requirements.txt`
2. Create `.env` with `GEMINI_API_KEY=...`
3. Run notebooks top to bottom
4. Put images in `data/multimodal/` for the multimodal notebook

GitHub repo: [annie-2314/rag-1](https://github.com/annie-2314/rag-1)
