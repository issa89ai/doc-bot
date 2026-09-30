"""Single-owner API. Use one worker with the embedded Chroma store."""
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import secrets
import tempfile
import threading
import time
import traceback
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
import rag
from storage import backup_pdf, delete_backup

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)
lock = threading.RLock()
sessions = OrderedDict()
metrics = {"total_questions": 0, "total_response_time": 0.0, "errors": 0}


def configured_key():
    key = os.getenv("API_KEY", "")
    if len(key) < 32 or key == "docbot-secret-123":
        raise RuntimeError("Set API_KEY to a random secret of at least 32 characters.")
    return key


@asynccontextmanager
async def lifespan(app):
    configured_key()
    Path(rag.DOCS_DIR).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Doc-Bot", version="3.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(key_header)):
    if not secrets.compare_digest((key or "").encode(), configured_key().encode()):
        raise HTTPException(401, "Invalid or missing API key.")


@app.middleware("http")
async def security_headers(request, call_next):
    from fastapi.responses import JSONResponse
    # Authenticate before multipart parsing/spooling for private API routes.
    if request.url.path in {"/upload", "/chat", "/documents", "/metrics", "/ready"} or request.url.path.startswith(("/documents/", "/session/")):
        try:
            require_api_key(request.headers.get("X-API-Key"))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    length = request.headers.get("content-length")
    if request.url.path == "/upload" and length:
        try:
            if int(length) > MAX_UPLOAD_BYTES + 65536:
                return JSONResponse({"detail": "Maximum PDF size is 10 MB."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid content length."}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    return response


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=100)
    selected_docs: list[str] | None = Field(default=None, max_length=100)


def checked_filename(filename):
    try:
        return rag.validate_filename(filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}  # Liveness only.


@app.get("/ready", dependencies=[Depends(require_api_key)])
def ready():
    try:
        rag.check_ready()
    except Exception:
        raise HTTPException(503, "AI dependencies unavailable. Check Ollama and both models.")
    return {"status": "ready"}


@app.post("/upload", dependencies=[Depends(require_api_key)])
def upload(file: UploadFile = File(...)):
    filename = checked_filename(file.filename)
    temp_path = None
    stage = "temporary file"
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as target:
            temp_path = Path(target.name)
            size = 0
            while chunk := file.file.read(65536):
                if size == 0 and not chunk.startswith(b"%PDF-"):
                    raise HTTPException(400, "File must contain a PDF.")
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Maximum PDF size is 10 MB.")
                target.write(chunk)
            if size == 0:
                raise HTTPException(400, "The PDF is empty.")
        with lock:
            stage = "PDF indexing"
            chunks = rag.load_and_index(str(temp_path), filename)
            try:
                backup = backup_pdf(str(temp_path), filename)
            except Exception:
                logger.warning("S3 backup failed; local indexing succeeded.")
                backup = "failed"
        return {"filename": filename, "chunks_indexed": chunks, "backup_status": backup,
                "message": "Document indexed."}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Exception messages/locals may contain document text or credentials.
        # Log only the exception type and code locations, never their values.
        locations = " -> ".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in traceback.extract_tb(exc.__traceback__)
        )
        logger.error("Upload failed at %s [%s]; code path: %s",
                     stage, type(exc).__name__, locations)
        raise HTTPException(503, "Could not index PDF. Check PDF validity and Ollama availability.")
    finally:
        file.file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if req.selected_docs == []:
        raise HTTPException(400, "Select at least one document.")
    with lock:
        history, turns = sessions.get(req.session_id, ([], 0))
        start = time.monotonic()
        try:
            reply, sources = rag.answer(req.question, history, req.selected_docs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            metrics["errors"] += 1
            raise HTTPException(503, "AI service unavailable. Check Ollama and installed models.")
        metrics["total_questions"] += 1
        metrics["total_response_time"] += time.monotonic() - start
        sessions[req.session_id] = ((history + [(req.question, reply)])[-3:], turns + 1)
        sessions.move_to_end(req.session_id)
        while len(sessions) > 100:
            sessions.popitem(last=False)
    return {"answer": reply, "sources": sources, "session_id": req.session_id, "turn": turns + 1}


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents():
    with lock:
        files = rag.list_indexed_files()
    return {"documents": files, "count": len(files)}


@app.delete("/documents/{filename}", dependencies=[Depends(require_api_key)])
def delete_document(filename: str):
    filename = checked_filename(filename)
    with lock:
        try:
            delete_backup(filename)
        except Exception:
            raise HTTPException(503, "Cloud deletion failed. Document retained locally; retry.")
        rag.delete_document(filename)
        sessions.clear()
    return {"message": "Document deleted.", "filename": filename}


@app.delete("/session/{session_id}", dependencies=[Depends(require_api_key)])
def clear_session(session_id: str):
    with lock:
        sessions.pop(session_id, None)
    return {"message": "Conversation cleared."}


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def get_metrics():
    with lock:
        total = metrics["total_questions"]
        return {"total_questions": total, "errors": metrics["errors"],
                "average_response_time_s": round(metrics["total_response_time"] / total, 2) if total else 0}
