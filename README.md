# Doc-Bot — RAG-Powered Document Chatbot

A local AI assistant that lets you upload PDFs and chat with them using a Retrieval-Augmented Generation (RAG) pipeline. Built as part of an 8-week ML/AI Engineer learning roadmap.

---

## What It Does

- Upload one or more PDF documents
- Ask questions in plain English
- Get accurate answers with page-level source citations
- Remembers the last 3 turns of your conversation

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.2 via Ollama (runs locally, free) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Framework | LangChain |
| API | FastAPI |
| UI | Embedded HTML (dark theme, served at `/`) |

---

## Project Structure

```
doc-bot/
├── main.py          # FastAPI app — REST endpoints + embedded chat UI
├── rag.py           # RAG logic — load, index, retrieve, answer
├── chatbot.py       # CLI version of the chatbot
├── docs/            # Drop your PDFs here
├── chroma_db/       # Vector store (auto-generated, gitignored)
└── README.md
```

---

## Setup

### 1. Install Ollama and pull models
```bash
# Download Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Python dependencies
```bash
pip install langchain langchain-community langchain-chroma langchain-ollama
pip install fastapi uvicorn python-multipart pypdf
```

### 3. Run the API server
```bash
python -m uvicorn main:app --reload
```

### 4. Open the chat UI
Go to `http://localhost:8000` in your browser.

---

## How to Use

1. Drop a PDF into the sidebar (drag & drop or click to upload)
2. Wait for the indexing confirmation
3. Type your question and hit Send
4. The bot answers and cites which pages it used

To use the CLI version instead:
```bash
# Put PDFs in the docs/ folder first
python chatbot.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/chat` | Ask a question |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/session/{id}` | Clear conversation history |
| `GET` | `/docs` | Auto-generated API docs (Swagger) |

### Example: Chat via curl
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "session_id": "default"}'
```

### Example response
```json
{
  "answer": "This document is a letter requesting...",
  "sources": ["document.pdf p.1"],
  "session_id": "default",
  "turn": 1
}
```

---

## Weekly Progress

### ✅ Week 1 — First RAG App
- PDF loading with LangChain `PyPDFLoader`
- Text splitting (`RecursiveCharacterTextSplitter`, 1000 chars / 100 overlap)
- Embeddings with `nomic-embed-text` stored in ChromaDB
- RAG chain: retrieve → prompt → LLM → answer
- Streaming output, page source citations
- **Deliverable:** CLI chatbot (`chatbot.py`)

### ✅ Week 2 — Vector DB & RAG Deep Dive
- Multi-document support — scans `docs/` folder for all PDFs
- Switched to **MMR retrieval** (fetch 20, return 6 diverse results)
- Conversation history — last 3 turns injected into every prompt
- Source citations now show filename + page number
- **Deliverable:** Upgraded CLI chatbot with multi-doc and memory

### ✅ Week 3 — REST API & Chat UI
- Extracted RAG logic into `rag.py` (shared by API and CLI)
- FastAPI backend with 5 clean endpoints
- Session-based conversation history (each browser tab gets its own session)
- Embedded dark-themed chat UI — drag & drop upload, real-time chat
- Auto-generated Swagger docs at `/docs`
- **Deliverable:** `python -m uvicorn main:app --reload` → open `localhost:8000`

### 🔜 Week 4 — Docker + Frontend Polish
- Dockerize the FastAPI app
- `docker-compose up` spins everything up in one command
- No more manual `uvicorn` commands

### 🔜 Week 5 — MLOps & Experiment Tracking
- Fine-tune a classifier with PyTorch / Scikit-learn
- Track experiments with MLflow

### 🔜 Week 6 — Cloud Deployment (AWS)
- Deploy to AWS (ECS or EC2)
- PDFs stored in S3
- Public URL — no local server needed

### 🔜 Week 7 — CI/CD + Kubernetes
- GitHub Actions pipeline: push → test → deploy automatically
- Kubernetes basics with minikube

### 🔜 Week 8 — Agents + Monitoring
- Upgrade to LangChain Agent (web search, database tools)
- Prometheus/Grafana or CloudWatch monitoring
- Production hardening

---

## Roadmap Goal

Build a full ML/AI Engineer portfolio project covering every skill in an Ottawa ML/AI Engineer job posting — Python, LangChain, RAG, vector databases, FastAPI, Docker, AWS, MLflow, CI/CD, and LangChain Agents.
