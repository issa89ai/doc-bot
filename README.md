# Doc-Bot

A single-owner PDF assistant built with FastAPI, LangChain, ChromaDB and Ollama.
Focused on retrieval-augmented generation (RAG) for selectable-text PDFs.

## Scope and architecture

PDF upload -> text extraction -> chunks -> embeddings -> Chroma retrieval ->
Llama answer with retrieved-page labels.

This repository owns the PDF interface, API, document storage, retrieval pipeline,
and RAG evaluation. It does not train topic classifiers or require a training
repository, MLflow database, or shared model artifacts. Its runtime dependencies
are declared in `requirements.txt`; test dependencies are in `requirements-dev.txt`.
Use a dedicated virtual environment for this application.

See [Development and evaluation record](PROGRESS.md) for failures, improvements,
manual test results (8 receipt + 5 CV checks), and remaining limitations.

**Status:** learning project under hardening, not a production-ready public service.
The previous AWS deployment has not been updated or verified by this repair pass.
Both indexing and answering require a reachable Ollama server with the models below.

## Local setup

Use Python 3.11 or 3.12 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

If `.env` already exists, **do not overwrite it**: edit its settings locally instead.
Put the generated secret in `API_KEY` locally. Never paste it into chat or commit it.
The server refuses missing or weak keys. AWS credentials are not required for local use;
leave `S3_BUCKET` blank to disable backups.

Start Ollama and install both models:

```text
ollama pull llama3.2
ollama pull nomic-embed-text
python -m uvicorn main:app --reload
```

Open http://localhost:8000 and enter your API key. It stays in tab memory, not browser
storage. Connect, upload a text PDF, select documents, and ask a question. Scanned
image-only PDFs need OCR (not implemented). Maximum upload: 10 MB. Filenames must
use letters, numbers, spaces, dots, parentheses, dashes or underscores.

The frontend requires HTTPS outside localhost. Do not send credentials to a public HTTP URL.
The owner key grants access to all documents; this is **not multi-user authentication**.

## Organization

| File | Responsibility |
| --- | --- |
| `main.py` | Authenticated API, request limits, bounded in-memory sessions |
| `rag.py` | Shared PDF indexing, retrieval and model calls |
| `exact_fields.py` | Conservative comparisons of labeled identifiers and single-item receipt tables |
| `storage.py` | Optional S3 backup and deletion using boto3's credential chain |
| `static/` | HTML, responsive CSS and browser JavaScript |
| `chatbot.py` | CLI using the shared RAG module |
| `tests/` | Local regression tests using mocked external services |
| `.github/workflows/ci.yml` | Test-and-build CI; no AWS deployment |

Use one API worker. Do not run the CLI against the same Chroma store while the API
is running. Sessions and metrics reset on restart. Document files and vectors persist.

## API

All routes below except `/`, `/static/*`, `/health` and API documentation require
the `X-API-Key` header. No server secret is embedded in the frontend.

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Validate, index and optionally back up a PDF |
| `POST /chat` | Answer a question; requires `question` and `session_id` |
| `GET /documents` | List filenames present in the vector index |
| `DELETE /documents/{filename}` | Remove cloud copy, index entries and local PDF |
| `DELETE /session/{session_id}` | Clear conversation |
| `GET /metrics` | In-memory usage metrics |
| `GET /health` | Process liveness only |
| `GET /ready` | Check Ollama model inventory and Chroma access |

Omit `selected_docs` to query all documents; an explicit empty list is rejected.
Source labels identify retrieved pages, not proof that every generated claim is correct.
Document-only prompts reduce unsupported answers but do not guarantee correctness or
provide complete protection against prompt injection.

PDFs containing broken Unicode mappings are retried with PDFium rather than
guessing missing characters. Existing uploads must be re-uploaded to refresh
their indexed text after this extraction change. Supported single-item receipt
tables and invoice/receipt-number comparisons use deterministic code; other
layouts still use the model and require evaluation. These narrow helpers do not
establish reliability for arbitrary financial documents.

Re-uploading replaces the document's previous chunks after new embeddings succeed.
Failed embeddings preserve the old document. Filesystem and vector updates are not a
cross-system transaction; crash recovery and multi-process writes remain future work.
S3 backup failures return `backup_status: failed`, not a false backup success.
Cloud deletion failures retain the local document for retry. Versioned S3 buckets can
retain old versions after deletion; lifecycle/version cleanup is an AWS policy decision.

## Docker

```text
docker compose up --build
```

The Compose setup targets Ollama on your host computer. It passes `API_KEY`, `S3_BUCKET`
and `AWS_REGION`, and bind-mounts `docs/` and `chroma_db/`. It does not launch Ollama or
automatically pass host AWS credentials into the container. For the first local test,
leave S3 disabled. Configure cloud credentials through an appropriate role/profile separately.

## Tests and evaluation

```text
python -m pytest -q
node --test tests/frontend.test.cjs
```

Default tests mock AI and AWS calls; they do not establish cloud or model availability.
Set `RUN_OLLAMA_TESTS=1` to enable the synthetic local-model grounding checks.
For private end-to-end evaluation, run
`python scripts/evaluate_receipt.py PATH_TO_PDF PATH_TO_PRIVATE_CASES_JSON`.
The runner exercises upload, extraction, embeddings, retrieval, and chat using local
Ollama and an ephemeral index (no AWS). Add `--isolated` for separate sessions.
Keep cases and generated results under the ignored `tmp/` directory, never in Git.
After updating the extractor, restart the app and re-upload older PDFs: existing
saved chunks are not automatically rewritten. Corrupted retrieved text now prompts
re-upload rather than returning unreliable extracted fields.
Doc-Bot uses pretrained generation and embedding models through Ollama. Uploading
PDFs builds a retrieval index; it does not train or fine-tune model weights.

Dependencies are currently not fully locked. On October 4, 2026, an isolated Docker
smoke test passed upload, host Ollama inference, source citation, restart persistence,
authentication, and deletion. See `scripts/smoke_docker.py` and PROGRESS.md. This does
not verify cloud deployment or data recovery after container replacement.

## AWS deployment safety gates

Automatic deployment is disabled in the repository changes. CI tests/builds only.
Before re-enabling deployment:

1. Replace exposed AWS access keys and the SSH private key. Verify replacement access
   before removing the old SSH public key from EC2 `authorized_keys`. Update GitHub secrets.
2. Prefer a bucket-scoped IAM instance role to static AWS access keys. Restrict IAM
   permissions to the required bucket/prefix, not `AmazonS3FullAccess`.
3. Configure HTTPS and restrict direct port 8000 access. Set a reverse-proxy body-size
   limit and request-rate limits (application limits alone are not sufficient).
4. Configure a reachable Ollama service and enough memory, then test upload and chat.
5. Build a candidate image before replacing a running container; verify readiness and
   provide rollback. The old stop-before-build workflow was removed.
6. Verify billing in AWS. Do not assume EC2, disks or Elastic IPs are free.

No AWS resources or credentials are changed by these local repairs. Cloud operation,
HTTPS, credential rotation, backup restore and billing must be checked separately.

## RAG application progress

Progress below refers only to the PDF assistant, not a broader learning syllabus.

| Component | Actual status |
| --- | --- |
| PDF chatbot | Local receipt API evaluation and 13 browser checks passed; broader evaluation pending |
| Multi-doc retrieval/history | Implemented; Pinecone/hybrid search not implemented |
| API/auth | Hardened locally; shared owner key, not separate user accounts |
| Docker/frontend | Docker configuration and responsive custom UI; no separate DB service |
| Cloud | Prior UI deployment; working cloud inference/HTTPS not verified |
| CI/CD | Test/build CI added; deployment paused; Kubernetes not implemented |
| Monitoring | Basic metrics only; agents, external monitoring and production hardening incomplete |
