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

### ✅ Week 4 — Docker + Container Deployment
- Multi-stage Dockerfile (builder → slim runtime)
- `docker-compose.yml` with volume mounts for docs and chroma_db
- `OLLAMA_HOST` env var bridges Docker container → host Ollama
- **Deliverable:** `docker-compose up` spins up the entire app locally

### ✅ Week 5 — MLOps & Experiment Tracking
- Trained a 4-class topic classifier on 20 Newsgroups dataset (sci.med, sci.space, rec.sport.hockey, talk.politics.guns)
- 6 model runs: LogisticRegression (C=0.1, C=1.0, bigrams), NaiveBayes, LinearSVC (unigrams + bigrams)
- 8 alpha-tuning runs for NaiveBayes (alpha: 0.001 → 5.0) — 14 MLflow runs total
- Best model: NaiveBayes alpha=0.5 → accuracy=91.26%
- **Deliverable:** MLflow dashboard at `http://localhost:5000` with 14 runs and metric charts

### ✅ Week 6 — Cloud Deployment (AWS)
- S3 bucket `doc-bot-pdfs-ahmadissa` (us-east-2) — every uploaded PDF is stored in the cloud
- IAM user `doc-bot-app` with scoped S3 access
- EC2 t3.micro instance (Amazon Linux 2023, us-east-2) running Docker
- App deployed via `docker run` on EC2 — publicly accessible
- **Live URL:** http://18.227.122.170:8000 (Elastic IP — permanent)
- **Deliverable:** Doc-Bot accessible from any browser in the world

### ✅ Week 3 — REST API & Chat UI
- Extracted RAG logic into `rag.py` (shared by API and CLI)
- FastAPI backend with 5 clean endpoints
- Session-based conversation history (each browser tab gets its own session)
- Embedded dark-themed chat UI — drag & drop upload, real-time chat
- Auto-generated Swagger docs at `/docs`
- **Deliverable:** `python -m uvicorn main:app --reload` → open `localhost:8000`

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

---

## 8-Week ML/AI Engineer Roadmap

### Week 1 — Python Foundations & Your First RAG App
**Theme:** Get a working chatbot running locally by end of week.

**What to build:** A simple Q&A chatbot that reads a PDF and answers questions about it.

**Tools:** Python, LangChain (document loaders, text splitters, chains), Ollama (free local LLMs), ChromaDB

**Deliverable:** CLI chatbot that answers questions about any PDF you feed it.

---

### Week 2 — Vector Databases & RAG Deep Dive
**Theme:** Understand how retrieval actually works under the hood.

**What to build:** Upgrade your chatbot — add multi-document support, conversation history, and source citations.

**Tools:** Embeddings (nomic-embed-text / sentence-transformers), ChromaDB, retrieval strategies (similarity search, MMR, hybrid)

**Key concepts:** Embedding distance, chunking strategies, context window limits

**Deliverable:** Multi-doc chatbot that cites which document/page it used.

---

### Week 3 — Wrap It in an API (Software Engineering Layer)
**Theme:** Turn your script into a real backend service.

**What to build:** REST API around your RAG chatbot — file upload endpoint + chat endpoint + embedded UI.

**Tools:** FastAPI, Pydantic, uvicorn

**Deliverable:** Working API + chat UI accessible at `localhost:8000`.

---

### Week 4 — Containerize & Add a Frontend
**Theme:** Package your app so it runs anywhere.

**What to build:** Dockerize the FastAPI app, docker-compose setup with app + vector DB as separate services.

**Tools:** Docker (Dockerfile, layers, multi-stage builds), docker-compose (networking, volumes, env vars), Streamlit or Gradio

**Deliverable:** `docker-compose up` spins up your entire app locally.

---

### Week 5 — MLOps: Experiment Tracking & Model Management
**Theme:** Start doing ML the professional way.

**What to build:** Fine-tune a small classifier (sentiment or topic tagger), track every run with MLflow.

**Tools:** Scikit-learn / PyTorch (training loops, evaluation metrics), MLflow (experiment tracking, model registry, artifact logging)

**Key concepts:** Train/val/test splits, overfitting, model versioning

**Deliverable:** MLflow dashboard showing 10+ experiment runs with metrics compared.

---

### Week 6 — Cloud Deployment (AWS)
**Theme:** Put your app on the internet.

**What to build:** Deploy FastAPI to AWS, store PDFs in S3, run MLflow on EC2.

**Tools:** AWS S3, EC2, IAM, ECR, ECS or SageMaker

**Deliverable:** Your chatbot accessible via a public URL.

---

### Week 7 — CI/CD Pipeline & Kubernetes Basics
**Theme:** Automate testing and deployment like a real engineering team.

**What to build:** GitHub Actions pipeline (push → test → build → deploy), deploy to Kubernetes.

**Tools:** GitHub Actions, pytest, Kubernetes (pods, deployments, services, ingress), minikube or EKS

**Deliverable:** Push a code change → app auto-deploys within minutes, no manual steps.

---

### Week 8 — Agents, Monitoring & Polish
**Theme:** Add advanced AI features and make it production-ready.

**What to build:** Upgrade chatbot to an agent (web search, database queries, external APIs), add monitoring, write polished docs.

**Tools:** LangChain Agents / LangGraph, Prometheus + Grafana or AWS CloudWatch

**Deliverable:** A polished GitHub repo you'd be proud to show in an interview.

---

## Skills Checklist

| Job Requirement | Where You Learn It |
|---|---|
| Python | Week 1 + throughout |
| PyTorch / Scikit-learn | Week 5 |
| AWS / Cloud | Week 6 |
| Docker / Kubernetes | Week 4 + Week 7 |
| MLflow (MLOps) | Week 5 |
| CI/CD pipelines | Week 7 |
| LLMs / RAG / LangChain | Week 1–2 |
| Vector databases | Week 2 |
| Agent-based architectures | Week 8 |
| FastAPI / REST APIs | Week 3 |
| Model monitoring | Week 8 |
