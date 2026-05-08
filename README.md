# RAG-Pipeline-From-Scratch

Production-grade Retrieval-Augmented Generation pipeline implemented from scratch in Python with:

- token-aware recursive chunking via `tiktoken`
- embedding caching and retry handling
- custom NumPy vector search with persistence
- grounded generation with citations and streaming output
- a CLI for indexing and interactive chat

## Project Layout

```text
config.py
chunker.py
embedder.py
vector_store.py
retriever.py
generator.py
main.py
requirements.txt
```

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Export your Groq API key for generation.

```bash
export GROQ_API_KEY="your-groq-api-key"
```

4. Optional: override defaults with environment variables.

```bash
export RAG_CHUNK_SIZE_TOKENS=400
export RAG_CHUNK_OVERLAP_TOKENS=60
export RAG_TOP_K=5
export RAG_SIMILARITY_THRESHOLD=0.2
export RAG_INDEX_DIR="$(pwd)/artifacts/index"
export RAG_EMBEDDING_CACHE_PATH="$(pwd)/artifacts/embedding_cache.json"
export RAG_LLM_MODEL="openai/gpt-oss-120b"
export RAG_LLM_BASE_URL="https://api.groq.com/openai/v1"
export RAG_LLM_MAX_REQUESTS_PER_MINUTE=30
export RAG_LLM_MAX_REQUESTS_PER_DAY=1000
export RAG_LLM_MAX_TOKENS_PER_MINUTE=8000
export RAG_LLM_MAX_TOKENS_PER_DAY=200000
```

## Index Documents

The indexer reads local `.txt` files by default.

```bash
python3 main.py index --docs-dir ./docs
```

Useful flags:

```bash
python3 main.py index --docs-dir ./docs --index-dir ./artifacts/index --glob "**/*.txt"
```

## Chat Against the Index

Load an existing persisted index:

```bash
python3 main.py chat --index-dir ./artifacts/index
```

Build the index if it does not exist yet:

```bash
python3 main.py chat --docs-dir ./docs
```

Force a rebuild before chatting:

```bash
python3 main.py chat --docs-dir ./docs --rebuild
```

Show the exact retrieved context block for debugging:

```bash
python3 main.py chat --index-dir ./artifacts/index --show-context
```

## Default Model Setup

The project now defaults to:

- Groq-compatible chat generation via `openai/gpt-oss-120b`
- local `sentence-transformers` embeddings via `all-MiniLM-L6-v2`

This keeps embedding traffic local, so you can run the full pipeline with only a Groq chat key.

## Embedding Backends

Local sentence-transformers embeddings are the default:

```bash
export RAG_EMBEDDING_PROVIDER="sentence-transformers"
export RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

The generator still uses an OpenAI-compatible chat completion endpoint for answer generation.

If you want remote embeddings instead, point the embedding client at a separate OpenAI-compatible endpoint:

```bash
export RAG_EMBEDDING_PROVIDER="openai"
export RAG_EMBEDDING_MODEL="text-embedding-3-small"
export RAG_EMBEDDING_BASE_URL="https://your-embedding-endpoint/v1"
export RAG_EMBEDDING_API_KEY="your-embedding-api-key"
```

## Groq Notes

- chat generation reads `GROQ_API_KEY`, `RAG_LLM_API_KEY`, or `OPENAI_API_KEY`
- the default chat base URL is `https://api.groq.com/openai/v1`
- the default chat model is `openai/gpt-oss-120b`
- the generator enforces the supplied Groq budgets: `30` requests/minute, `1000` requests/day, `8000` tokens/minute, and `200000` tokens/day
- the API key is not stored in the repository; keep it in your shell environment

## Persistence Details

- vector matrix: `artifacts/index/vectors.npy`
- metadata payloads: `artifacts/index/metadata.json`
- embedding cache: `artifacts/embedding_cache.json`

## Notes

- cosine similarity is computed with a vectorized dot product because all vectors are L2 normalized
- retrieval drops hits below `RAG_SIMILARITY_THRESHOLD`
- the generator is explicitly instructed to answer only from retrieved context and to cite source tags