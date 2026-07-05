# Doc-Bot — RAG-Powered Document Chatbot

A production-deployed AI assistant that lets you upload PDFs and chat with them using a full Retrieval-Augmented Generation (RAG) pipeline. Built end-to-end as an 8-week ML/AI Engineer portfolio project.

**Live:** http://18.227.122.170:8000

---

## Architecture

```
User Browser
     │
     │  HTTP
     ▼
FastAPI (EC2 t3.micro, us-east-2)
     │
     ├── /upload ──► Local disk + S3 (doc-bot-pdfs-ahmadissa)
     │
     └── /chat
           │
           ├── ChromaDB (vector store, MMR retrieval)
           │       └── nomic-embed-text embeddings (Ollama)
           │
           └── llama3.2 LLM (Ollama)
                   └── Strict document-only prompt
```

**Deployment pipeline:**
```
git push → GitHub Actions → SSH into EC2 → git pull → docker build → docker run
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.2 via Ollama (runs locally, free) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB |
| Retrieval | MMR (Maximal Marginal Relevance, k=6, fetch_k=20) |
| Framework | LangChain |
| API | FastAPI + Pydantic |
| UI | Embedded dark-theme HTML/CSS/JS (mobile-responsive) |
| Container | Docker (multi-stage build) |
| Cloud | AWS EC2 + S3 + IAM + Elastic IP |
| CI/CD | GitHub Actions (auto-deploy on every push) |
| ML Tracking | MLflow (14 experiment runs, SQLite backend) |

---

## Features

- Upload PDFs via drag & drop or file picker
- Ask questions in plain English — answers cite exact page numbers
- Session-based conversation memory (last 3 turns)
- Strict document-only answers — refuses to answer from outside knowledge
- Every uploaded PDF backed up to AWS S3
- `/metrics` endpoint tracks questions, response times, errors
- API key authentication on all write endpoints
- Auto-deploys on every GitHub push via CI/CD

---

## Project Structure

```
doc-bot/
├── main.py                          # FastAPI app — REST endpoints + embedded chat UI
├── rag.py                           # RAG logic — load, index, retrieve, answer
├── chatbot.py                       # CLI version of the chatbot
├── train.py                         # MLflow experiment tracking (14 runs)
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Local development setup
├── requirements.txt
├── .github/workflows/deploy.yml     # GitHub Actions CI/CD pipeline
├── docs/                            # PDF storage (gitignored)
├── chroma_db/                       # Vector store (gitignored)
└── mlflow.db                        # MLflow SQLite backend (gitignored)
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Chat UI |
| `POST` | `/upload` | Yes | Upload and index a PDF |
| `POST` | `/chat` | Yes | Ask a question |
| `GET` | `/documents` | No | List indexed documents |
| `DELETE` | `/session/{id}` | No | Clear conversation history |
| `GET` | `/metrics` | No | Usage stats |
| `GET` | `/health` | No | Health check |
| `GET` | `/docs` | No | Swagger API docs |

**Authentication:** Pass `X-API-Key: docbot-secret-123` header on upload and chat requests.

---

## Local Setup

### 1. Install Ollama and pull models
```bash
# Download from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Clone and install dependencies
```bash
git clone https://github.com/issa89ai/doc-bot.git
cd doc-bot
pip install -r requirements.txt
```

### 3. Run the API
```bash
python -m uvicorn main:app --reload
```

### 4. Open the chat UI
Go to `http://localhost:8000`

### Docker (alternative)
```bash
docker-compose up
```

---

## MLflow Experiment Tracking

Trained a 4-class topic classifier on the 20 Newsgroups dataset (sci.med, sci.space, rec.sport.hockey, talk.politics.guns) with 14 tracked runs:

| Model | Accuracy |
|---|---|
| NaiveBayes (alpha=0.5) | 91.26% — best |
| LinearSVC bigrams | 90.8% |
| LogisticRegression bigrams | 90.4% |
| LinearSVC | 89.9% |
| LogisticRegression C=1.0 | 89.1% |
| LogisticRegression C=0.1 | 85.3% |

```bash
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## AWS Infrastructure

| Resource | Details |
|---|---|
| EC2 | t3.micro, Amazon Linux 2023, us-east-2 |
| Elastic IP | 18.227.122.170 (permanent) |
| S3 Bucket | doc-bot-pdfs-ahmadissa (us-east-2) |
| IAM User | doc-bot-app (AmazonS3FullAccess) |
| Security Group | Ports 22 (SSH), 8000 (app) |

---

## Weekly Progress

### Week 1 — First RAG App
- PDF loading with LangChain `PyPDFLoader`
- Text splitting (1000 chars / 100 overlap)
- ChromaDB vector store + nomic-embed-text embeddings
- **Deliverable:** CLI chatbot (`chatbot.py`)

### Week 2 — Vector DB & RAG Deep Dive
- Multi-document support
- MMR retrieval (fetch 20, return 6 diverse results)
- Conversation history (last 3 turns)
- Source citations with filename + page number
- **Deliverable:** Multi-doc CLI chatbot with memory

### Week 3 — REST API & Chat UI
- FastAPI backend with 7 endpoints
- Session-based conversation history
- Embedded dark-theme chat UI with drag & drop
- API key authentication
- **Deliverable:** `localhost:8000` chat interface

### Week 4 — Docker
- Multi-stage Dockerfile (builder → slim runtime)
- docker-compose with volume mounts
- OLLAMA_HOST env var for Docker → host Ollama bridge
- **Deliverable:** `docker-compose up` spins up everything

### Week 5 — MLOps & Experiment Tracking
- 14 MLflow runs across 3 model families
- NaiveBayes alpha hyperparameter sweep (8 runs)
- Best: NaiveBayes alpha=0.5 → 91.26% accuracy
- **Deliverable:** MLflow dashboard with metric charts

### Week 6 — Cloud Deployment (AWS)
- S3 bucket for PDF backup on every upload
- EC2 t3.micro deployment with Docker
- **Deliverable:** App live at http://18.227.122.170:8000

### Week 7 — CI/CD Pipeline
- GitHub Actions workflow: push → SSH → rebuild → redeploy
- Elastic IP for permanent fixed address
- Docker `--restart always` for auto-start on boot
- **Deliverable:** Push code → deployed in ~15 seconds automatically

### Week 8 — Monitoring & Polish
- Strict document-only prompt (no hallucination)
- Request logging to `requests.log`
- `/metrics` endpoint: total questions, avg response time, errors
- **Deliverable:** Production-hardened, monitored, polished repo

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Python | Throughout |
| LangChain / RAG / LLMs | Weeks 1–2, rag.py |
| Vector databases (ChromaDB) | Weeks 1–2 |
| FastAPI / REST APIs | Week 3, main.py |
| Docker | Week 4, Dockerfile |
| Scikit-learn / MLflow | Week 5, train.py |
| AWS (EC2, S3, IAM) | Week 6 |
| CI/CD (GitHub Actions) | Week 7 |
| Monitoring / Observability | Week 8 |
