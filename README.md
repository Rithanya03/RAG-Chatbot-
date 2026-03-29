# Labeeb RAG Chatbot — Backend

A production-ready **Retrieval-Augmented Generation (RAG)** backend for the Labeeb business chatbot. Built with **FastAPI**, **FAISS**, and **NVIDIA NIM** (LLM + Embeddings).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  /documents  │  │    /chat     │  │      /health        │  │
│  │   (upload,   │  │  (stream,    │  │   (health check,    │  │
│  │  list, del)  │  │  non-stream) │  │   FAQs, styles)     │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────────────────┘  │
│         │                 │                                     │
│  ┌──────▼───────┐  ┌──────▼───────┐                           │
│  │   Document   │  │  RAG Service │                           │
│  │  Processor   │  │              │                           │
│  │ (PDF/DOCX/   │  │  Retrieve →  │                           │
│  │  TXT/CSV/MD) │  │  Augment →   │                           │
│  └──────┬───────┘  │  Generate    │                           │
│         │          └──────┬───────┘                           │
│  ┌──────▼───────┐         │                                   │
│  │ FAISS Vector │◄────────┘                                   │
│  │    Store     │                                             │
│  │  (cosine     │                                             │
│  │  similarity) │                                             │
│  └──────┬───────┘                                             │
│         │                                                     │
│  ┌──────▼───────────────────────────────────────────────────┐ │
│  │              NVIDIA NIM API                              │ │
│  │  Embeddings: nvidia/nv-embedqa-e5-v5 (1024-dim)         │ │
│  │  LLM:        meta/llama-3.1-70b-instruct                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  SQLite (chat sessions, messages, document metadata)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

```bash
cd rag_backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your NVIDIA_API_KEY
```

Get a free NVIDIA NIM API key at **https://build.nvidia.com**

### 3. Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | System health, vector store stats, NVIDIA API status |
| GET | `/api/v1/faqs` | Suggested FAQ questions for the UI |
| GET | `/api/v1/styles` | Available chat response styles |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/documents/upload` | Upload a document (PDF, DOCX, TXT, MD, CSV) |
| GET | `/api/v1/documents/` | List all documents |
| GET | `/api/v1/documents/{id}` | Get document status/details |
| DELETE | `/api/v1/documents/{id}` | Delete document + its vectors |

**Document ingestion flow:**
1. File is saved to disk
2. Text is extracted (PyPDF / python-docx / csv / text)
3. Text is split into overlapping chunks (512 words, 64 overlap)
4. Chunks are embedded via NVIDIA NIM in batches of 32
5. Embeddings are stored in FAISS (L2-normalised, cosine similarity)
6. DB record updated: `status: ready`

### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/sessions` | Create a chat session |
| GET | `/api/v1/chat/sessions/{id}` | Get session + message history |
| DELETE | `/api/v1/chat/sessions/{id}` | Delete session |
| POST | `/api/v1/chat/` | Send message (full JSON response) |
| POST | `/api/v1/chat/stream` | Send message (SSE stream) |

#### Chat Request Body

```json
{
  "message": "What is an Inventory Aging Trends Report?",
  "session_id": "optional-existing-session-id",
  "style": "professional",
  "use_rag": true,
  "document_ids": ["optional", "filter", "to", "specific", "docs"]
}
```

#### Streaming (SSE) Events

```
data: {"type": "sources", "sources": [...], "session_id": "...", "message_id": "..."}

data: {"type": "token", "content": "The "}
data: {"type": "token", "content": "Inventory "}
...

data: {"type": "done", "session_id": "...", "message_id": "...", "confidence": 0.87}
```

#### Chat Styles

| Style | Behaviour |
|-------|-----------|
| `professional` | Formal, structured responses |
| `friendly` | Warm, conversational tone |
| `technical` | Detailed, precise, technical |
| `concise` | Brief, direct answers only |

---

## Supported Document Types

| Extension | Parser |
|-----------|--------|
| `.pdf` | pypdf |
| `.docx` | python-docx |
| `.txt` | built-in |
| `.md` | built-in |
| `.csv` | built-in csv module |

Max file size: **50 MB** (configurable via `MAX_FILE_SIZE_MB`)

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | — | **Required.** Your NIM API key |
| `NVIDIA_LLM_MODEL` | `meta/llama-3.1-70b-instruct` | LLM model ID |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embedding model ID |
| `EMBEDDING_DIMENSION` | `1024` | Must match embedding model output |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlapping words between chunks |
| `TOP_K_RESULTS` | `5` | Retrieved chunks per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include |
| `MAX_TOKENS` | `1024` | Max LLM output tokens |
| `TEMPERATURE` | `0.2` | LLM temperature (lower = more factual) |

---

## Project Structure

```
rag_backend/
├── main.py                  # FastAPI app, CORS, router mounts
├── config.py                # Pydantic settings (reads .env)
├── database.py              # SQLAlchemy async models + session
├── schemas.py               # Pydantic request/response models
├── nvidia_client.py         # NVIDIA NIM async client (LLM + embeddings)
├── vector_store.py          # FAISS index with persistence
├── document_processor.py    # Text extraction + chunking
├── rag_service.py           # RAG orchestration (retrieve → augment → generate)
├── routes/
│   ├── chat.py              # Chat & session endpoints
│   ├── documents.py         # Document upload/list/delete
│   └── health.py            # Health check, FAQs, styles
├── requirements.txt
└── .env.example
```

---

## Switching NVIDIA Models

**LLMs available on NIM:**
- `meta/llama-3.1-70b-instruct` (recommended, best quality)
- `meta/llama-3.1-8b-instruct` (faster, lighter)
- `mistralai/mixtral-8x7b-instruct-v0.1`
- `nvidia/nemotron-4-340b-instruct` (largest)

**Embedding models:**
- `nvidia/nv-embedqa-e5-v5` — 1024-dim (recommended)
- `nvidia/embed-qa-4` — 1024-dim

> ⚠️ If you change the embedding model, delete the `vector_store/` directory and re-upload your documents since dimension may differ.
