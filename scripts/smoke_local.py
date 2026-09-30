"""Opt-in integration check against local Ollama; no AWS or real documents.

Run: python scripts/smoke_local.py path/to/synthetic.pdf
The PDF must state that the project code is BLUEBIRD-742.
"""
from pathlib import Path
import os
import secrets
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Override dotenv settings before importing the app. Never print real credentials.
os.environ["API_KEY"] = secrets.token_urlsafe(32)
os.environ["S3_BUCKET"] = ""
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi.testclient import TestClient
import main
import rag


def run(pdf):
    # Chroma may hold Windows file handles until process exit; use its ephemeral
    # client instead of opening the user's persistent store.
    import chromadb
    from langchain_chroma import Chroma
    client = chromadb.EphemeralClient()
    store = Chroma(client=client, collection_name="smoke-" + secrets.token_hex(8),
                   embedding_function=rag.embeddings())
    rag.vectorstore = lambda: store
    with tempfile.TemporaryDirectory(prefix="doc-bot-smoke-") as directory:
        rag.DOCS_DIR = str(Path(directory) / "docs")
        headers = {"X-API-Key": os.environ["API_KEY"]}
        with TestClient(main.app) as api:
            assert api.get('/').status_code == 200
            assert api.get('/documents').status_code == 401
            response = api.get('/ready', headers=headers)
            assert response.status_code == 200, response.text
            print('PASS: homepage, authentication, model readiness', flush=True)
            for attempt in range(2):
                response = api.post('/upload', headers=headers,
                                    files={'file': ('synthetic.pdf', pdf, 'application/pdf')})
                assert response.status_code == 200, response.text
                assert response.json()['backup_status'] == 'disabled'
                assert len(store.get()['ids']) == response.json()['chunks_indexed']
            print('PASS: PDF indexing and re-upload without duplicate chunks', flush=True)
            response = api.post('/chat', headers=headers, json={
                'question':'What is the project code?', 'session_id':'smoke',
                'selected_docs':['synthetic.pdf']})
            assert response.status_code == 200, response.text
            data = response.json()
            assert 'BLUEBIRD-742' in data['answer'], data['answer']
            assert data['sources'] == ['synthetic.pdf p.1'], data['sources']
            print('PASS: real model answer contains BLUEBIRD-742 and cites page 1', flush=True)
            assert api.delete('/documents/synthetic.pdf', headers=headers).status_code == 200
            assert api.get('/documents', headers=headers).json()['documents'] == []
            print('PASS: document deletion and empty index', flush=True)
        store.delete_collection()
    print('All live integration checks passed. Existing docs/index and AWS untouched.', flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]).read_bytes())
