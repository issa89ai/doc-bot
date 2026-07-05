from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
import os
import shutil
import time
import logging
from datetime import datetime, timezone

load_dotenv()

import rag

logging.basicConfig(
    filename="requests.log",
    level=logging.INFO,
    format="%(message)s"
)

metrics = {"total_questions": 0, "total_response_time": 0.0, "errors": 0}

# ── S3 client ─────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION", "us-east-2"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# ── API Key auth ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "docbot-secret-123")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass it as X-API-Key header.")

os.makedirs(rag.DOCS_DIR, exist_ok=True)
os.makedirs(rag.CHROMA_DIR, exist_ok=True)

app = FastAPI(
    title="Doc-Bot API",
    description="Upload PDFs and chat with them using local LLMs.",
    version="2.0.0",
)

# In-memory session store: session_id → list of (question, answer) tuples
sessions: dict[str, list[tuple[str, str]]] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the main topics in this document?",
                "session_id": "default"
            }
        }
    }


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    turn: int


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int


class DocumentsResponse(BaseModel):
    documents: list[str]
    count: int


class ClearResponse(BaseModel):
    message: str
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTML_UI


@app.post("/upload", response_model=UploadResponse, summary="Upload a PDF to index")
async def upload(file: UploadFile = File(..., description="A PDF file to index"), _=Depends(require_api_key)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = os.path.join(rag.DOCS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Upload to S3 for persistent cloud storage
    if S3_BUCKET:
        try:
            s3.upload_file(save_path, S3_BUCKET, f"pdfs/{file.filename}")
        except Exception as e:
            print(f"S3 upload warning: {e}")

    chunks = rag.load_and_index(save_path)
    return UploadResponse(
        message=f"'{file.filename}' indexed successfully.",
        filename=file.filename,
        chunks_indexed=chunks,
    )


@app.post("/chat", response_model=ChatResponse, summary="Ask a question about your documents")
def chat(req: ChatRequest, _=Depends(require_api_key)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = sessions.get(req.session_id, [])
    start = time.time()

    try:
        reply, sources = rag.answer(req.question, history)
    except Exception as e:
        metrics["errors"] += 1
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    elapsed = round(time.time() - start, 2)
    metrics["total_questions"] += 1
    metrics["total_response_time"] += elapsed

    logging.info({
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": req.question,
        "sources": sources,
        "response_time_s": elapsed,
        "session": req.session_id,
    })

    history.append((req.question, reply))
    sessions[req.session_id] = history

    return ChatResponse(
        answer=reply,
        sources=sources,
        session_id=req.session_id,
        turn=len(history),
    )


@app.get("/metrics", summary="Usage metrics")
def get_metrics():
    total = metrics["total_questions"]
    avg = round(metrics["total_response_time"] / total, 2) if total else 0
    return {
        "total_questions": total,
        "average_response_time_s": avg,
        "errors": metrics["errors"],
    }


@app.get("/documents", response_model=DocumentsResponse, summary="List indexed documents")
def documents():
    files = rag.list_indexed_files()
    return DocumentsResponse(documents=files, count=len(files))


@app.delete("/session/{session_id}", response_model=ClearResponse, summary="Clear conversation history")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return ClearResponse(message="Session cleared.", session_id=session_id)


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


# ── Embedded HTML UI ──────────────────────────────────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Doc-Bot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  #sidebar {
    width: 260px;
    background: #1a1d27;
    border-right: 1px solid #2a2d3a;
    display: flex;
    flex-direction: column;
    padding: 20px 16px;
    gap: 16px;
    flex-shrink: 0;
  }

  #sidebar h1 { font-size: 1.2rem; color: #7c8cf8; font-weight: 700; }
  #sidebar p  { font-size: 0.75rem; color: #666; }

  #upload-area {
    border: 2px dashed #2a2d3a;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    font-size: 0.82rem;
    color: #888;
  }
  #upload-area:hover, #upload-area.drag-over {
    border-color: #7c8cf8;
    background: #1e2235;
    color: #aab;
  }
  #upload-area span { display: block; font-size: 1.6rem; margin-bottom: 6px; }
  #file-input { display: none; }

  #upload-status {
    font-size: 0.78rem;
    min-height: 18px;
    color: #7c8cf8;
  }

  #doc-list { flex: 1; overflow-y: auto; }
  #doc-list h3 { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .doc-item {
    font-size: 0.8rem;
    padding: 6px 8px;
    border-radius: 6px;
    background: #22253a;
    margin-bottom: 4px;
    color: #aab;
    word-break: break-all;
  }

  #metrics-panel { border-top: 1px solid #2a2d3a; padding-top: 12px; }
  #metrics-panel h3 { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    padding: 3px 0;
    color: #888;
  }
  .metric-row span { color: #7c8cf8; font-weight: 600; }

  .btn-ghost {
    background: none;
    border: 1px solid #2a2d3a;
    color: #777;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.78rem;
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s;
  }
  .btn-ghost:hover { border-color: #7c8cf8; color: #7c8cf8; }

  /* ── Main chat area ── */
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #chat-header {
    padding: 14px 24px;
    border-bottom: 1px solid #1e2130;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0f1117;
  }
  #chat-header h2 { font-size: 0.95rem; color: #ccc; font-weight: 500; }
  #session-label { font-size: 0.75rem; color: #555; }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .msg {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  .msg.user {
    align-self: flex-end;
    background: #7c8cf8;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.bot {
    align-self: flex-start;
    background: #1a1d27;
    color: #dde;
    border-bottom-left-radius: 4px;
  }
  .msg .sources {
    margin-top: 8px;
    font-size: 0.75rem;
    color: #7c8cf8;
    opacity: 0.8;
  }

  .msg.thinking {
    align-self: flex-start;
    background: #1a1d27;
    color: #555;
    font-style: italic;
    font-size: 0.82rem;
  }

  #empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #333;
    gap: 8px;
    pointer-events: none;
  }
  #empty-state span { font-size: 2.5rem; }
  #empty-state p { font-size: 0.9rem; }

  /* ── Input bar ── */
  #input-bar {
    padding: 16px 24px;
    border-top: 1px solid #1e2130;
    display: flex;
    gap: 10px;
    background: #0f1117;
  }

  #question-input {
    flex: 1;
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    color: #e0e0e0;
    padding: 12px 16px;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
    resize: none;
    height: 48px;
    font-family: inherit;
  }
  #question-input:focus { border-color: #7c8cf8; }

  #send-btn {
    background: #7c8cf8;
    border: none;
    border-radius: 10px;
    color: #fff;
    padding: 0 20px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
    height: 48px;
  }
  #send-btn:hover { background: #6070e0; }
  #send-btn:disabled { background: #333; cursor: not-allowed; }

  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 4px; }
</style>
</head>
<body>

<!-- Sidebar -->
<div id="sidebar">
  <div>
    <h1>🤖 Doc-Bot</h1>
    <p>Chat with your documents</p>
  </div>

  <div id="upload-area" onclick="document.getElementById('file-input').click()"
       ondragover="event.preventDefault(); this.classList.add('drag-over')"
       ondragleave="this.classList.remove('drag-over')"
       ondrop="handleDrop(event)">
    <span>📄</span>
    Drop a PDF here<br/>or click to upload
  </div>
  <input type="file" id="file-input" accept=".pdf" onchange="uploadFile(this.files[0])"/>

  <div id="upload-status"></div>

  <div id="doc-list">
    <h3>Indexed Documents</h3>
    <div id="doc-items"></div>
  </div>

  <div id="metrics-panel">
    <h3>Usage Stats</h3>
    <div class="metric-row">Questions asked <span id="m-questions">—</span></div>
    <div class="metric-row">Avg response time <span id="m-avg">—</span></div>
    <div class="metric-row">Errors <span id="m-errors">—</span></div>
  </div>

  <button class="btn-ghost" onclick="clearSession()">🗑 Clear conversation</button>
</div>

<!-- Main -->
<div id="main">
  <div id="chat-header">
    <h2>Chat</h2>
    <span id="session-label">Session: default</span>
  </div>

  <div id="messages">
    <div id="empty-state">
      <span>💬</span>
      <p>Upload a PDF and start asking questions</p>
    </div>
  </div>

  <div id="input-bar">
    <textarea id="question-input" placeholder="Ask something about your documents..."
              onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault(); sendMessage();}"></textarea>
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
  const API_KEY = "docbot-secret-123";
  const sessionId = "session_" + Math.random().toString(36).slice(2, 8);
  document.getElementById("session-label").textContent = "Session: " + sessionId;

  async function uploadFile(file) {
    if (!file) return;
    const status = document.getElementById("upload-status");
    status.textContent = "Uploading " + file.name + "...";

    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch("/upload", { method: "POST", body: fd, headers: { "X-API-Key": API_KEY } });
    const data = await res.json();

    if (res.ok) {
      status.textContent = "✓ " + data.message + " (" + data.chunks_indexed + " chunks)";
      loadDocs();
    } else {
      status.textContent = "✗ " + (data.detail || "Upload failed");
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    document.getElementById("upload-area").classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  }

  async function loadDocs() {
    const res = await fetch("/documents");
    const data = await res.json();
    const container = document.getElementById("doc-items");
    container.innerHTML = data.documents.length
      ? data.documents.map(d => `<div class="doc-item">📄 ${d}</div>`).join("")
      : `<p style="color:#444; font-size:0.78rem;">No documents yet</p>`;
  }

  function addMessage(role, text, sources) {
    const empty = document.getElementById("empty-state");
    if (empty) empty.remove();

    const box = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;

    if (sources && sources.length) {
      const s = document.createElement("div");
      s.className = "sources";
      s.textContent = "Sources: " + sources.join(", ");
      div.appendChild(s);
    }

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }

  async function sendMessage() {
    const input = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question) return;

    input.value = "";
    input.style.height = "48px";
    document.getElementById("send-btn").disabled = true;

    addMessage("user", question);
    const thinking = addMessage("thinking", "Thinking...");

    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    thinking.remove();
    document.getElementById("send-btn").disabled = false;

    if (res.ok) {
      const data = await res.json();
      addMessage("bot", data.answer, data.sources);
    } else {
      const err = await res.json();
      addMessage("bot", "Error: " + (err.detail || "Something went wrong."));
    }
  }

  async function clearSession() {
    await fetch("/session/" + sessionId, { method: "DELETE" });
    document.getElementById("messages").innerHTML =
      `<div id="empty-state" style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#333;gap:8px;pointer-events:none">
        <span style="font-size:2.5rem">💬</span>
        <p style="font-size:0.9rem">Upload a PDF and start asking questions</p>
      </div>`;
  }

  async function loadMetrics() {
    const res = await fetch("/metrics");
    const d = await res.json();
    document.getElementById("m-questions").textContent = d.total_questions;
    document.getElementById("m-avg").textContent = d.average_response_time_s + "s";
    document.getElementById("m-errors").textContent = d.errors;
  }

  loadDocs();
  loadMetrics();
  setInterval(loadMetrics, 10000);
</script>
</body>
</html>
"""
