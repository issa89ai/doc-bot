# 8-Week ML/AI Engineer Learning Roadmap
**Goal:** Build a RAG-powered AI Assistant with full MLOps pipeline — covering every key skill in the Ottawa ML/AI Engineer job posting.

---

## Week 1 — Python Foundations & Your First RAG App
**Theme:** Get a working chatbot running locally by end of week.

**What to build:**
- A simple Q&A chatbot that reads a PDF and answers questions about it

**Tools to learn:**
- Python (data structures, async, type hints)
- LangChain — document loaders, text splitters, chains
- OpenAI API or Ollama (free, runs Llama locally)
- ChromaDB as your vector store

**Deliverable:** CLI chatbot that answers questions about any PDF you feed it.

**Resources:**
- LangChain docs: https://docs.langchain.com
- Ollama (free LLM locally): https://ollama.ai

---

## Week 2 — Vector Databases & RAG Deep Dive
**Theme:** Understand how retrieval actually works under the hood.

**What to build:**
- Upgrade your chatbot: add multi-document support, conversation history, and source citations

**Tools to learn:**
- Embeddings (OpenAI text-embedding-3-small or sentence-transformers)
- ChromaDB → migrate to Pinecone (cloud vector DB)
- Retrieval strategies: similarity search, MMR, hybrid search

**Deliverable:** Multi-doc chatbot that cites which document/page it used.

**Key concepts:** embedding distance, chunking strategies, context window limits

---

## Week 3 — Wrap It in an API (Software Engineering Layer)
**Theme:** Turn your script into a real backend service.

**What to build:**
- REST API around your RAG chatbot
- File upload endpoint (user posts a PDF, it gets indexed)
- Chat endpoint (user sends a question, gets an answer + sources)

**Tools to learn:**
- FastAPI — routing, request/response models, async endpoints
- Pydantic — data validation
- Basic auth with API keys

**Deliverable:** Working API you can call with curl or Postman.

---

## Week 4 — Containerize & Add a Frontend
**Theme:** Package your app so it runs anywhere.

**What to build:**
- Dockerize your FastAPI app
- Add a simple Streamlit or Gradio frontend
- docker-compose setup: app + vector DB as separate services

**Tools to learn:**
- Docker — Dockerfile, layers, multi-stage builds
- docker-compose — networking, volumes, environment variables
- Streamlit or Gradio for quick UIs

**Deliverable:** `docker-compose up` spins up your entire app locally.

---

## Week 5 — MLOps: Experiment Tracking & Model Management
**Theme:** Start doing ML the professional way.

**What to build:**
- Fine-tune a small classifier (e.g., sentiment or topic tagger) using PyTorch or Scikit-learn
- Track every experiment run with MLflow

**Tools to learn:**
- Scikit-learn / PyTorch — training loops, evaluation metrics
- MLflow — experiment tracking, model registry, artifact logging
- Hyperparameter tuning basics

**Deliverable:** MLflow dashboard showing 10+ experiment runs with metrics compared.

**Key concepts:** train/val/test splits, overfitting, model versioning

---

## Week 6 — Cloud Deployment (AWS)
**Theme:** Put your app on the internet.

**What to build:**
- Deploy your FastAPI app to AWS
- Store uploaded PDFs in S3
- Run MLflow on an EC2 instance

**Tools to learn:**
- AWS core: S3, EC2, IAM roles, security groups
- AWS ECR — push your Docker image to the cloud registry
- Either AWS ECS (simpler) or try SageMaker for model serving

**Deliverable:** Your chatbot accessible via a public URL.

**Free tier note:** AWS free tier covers most of this for 12 months.

---

## Week 7 — CI/CD Pipeline & Kubernetes Basics
**Theme:** Automate testing and deployment like a real engineering team.

**What to build:**
- GitHub Actions pipeline: on every push → run tests → build Docker image → deploy to AWS
- Deploy your app to a Kubernetes cluster (use minikube locally or EKS on AWS)

**Tools to learn:**
- GitHub Actions — workflows, jobs, secrets
- pytest — write unit and integration tests for your API
- Kubernetes basics — pods, deployments, services, ingress

**Deliverable:** Push a code change → app auto-deploys within minutes, no manual steps.

---

## Week 8 — Agents, Monitoring & Polish
**Theme:** Add advanced AI features and make it production-ready.

**What to build:**
- Upgrade chatbot to an agent: can search the web, query a database, or call an external API as tools
- Add model performance monitoring (latency, token usage, answer quality)
- Write a README that explains architecture, setup, and design decisions

**Tools to learn:**
- LangChain Agents / LangGraph — tool calling, multi-step reasoning
- Prometheus + Grafana OR AWS CloudWatch for monitoring
- Responsible AI basics: prompt injection defense, output filtering

**Deliverable:** A polished GitHub repo you'd be proud to show in an interview.

---

## Skills Checklist (mapped to the job posting)

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

---

## Tips for Success

1. **Commit code every day** — your GitHub contribution graph matters to technical recruiters.
2. **Write a short blog post or LinkedIn post** after each week summarizing what you built. This signals communication skills (also in the job requirements).
3. **Don't skip Week 3** — many ML engineers are weak on software engineering fundamentals. A clean API separates you from data scientists.
4. **Use free tiers** — AWS Free Tier, OpenAI $5 credit, Pinecone free tier, and Ollama (fully free, local) keep costs near zero.
5. **By Week 8, you have a portfolio project** — walk through it in interviews as a real system you architected end-to-end.
