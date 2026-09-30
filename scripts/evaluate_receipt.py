"""Run private PDF answer-key cases through upload, retrieval and chat APIs.

Usage: python scripts/evaluate_receipt.py PDF CASES_JSON [--isolated]
Cases JSON: [{"name": "...", "question": "...", "contains": [...],
              "excludes": [...], "patterns": [...]}]
Keep personal documents, cases and results out of Git. Uses local Ollama,
an ephemeral Chroma collection, temporary documents, and no AWS calls.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['API_KEY'] = secrets.token_urlsafe(32)
os.environ['S3_BUCKET'] = ''
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from fastapi.testclient import TestClient
import chromadb
from langchain_chroma import Chroma
import main
import rag


def run(pdf_path, cases_path, isolated=False):
    cases = json.loads(Path(cases_path).read_text(encoding='utf-8'))
    store = Chroma(client=chromadb.EphemeralClient(),
                   collection_name='evaluation-' + secrets.token_hex(8),
                   embedding_function=rag.embeddings())
    rag.vectorstore = lambda: store
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix='doc-bot-evaluation-') as root:
            rag.DOCS_DIR = str(Path(root) / 'docs')
            name = Path(pdf_path).name
            headers = {'X-API-Key': os.environ['API_KEY']}
            with TestClient(main.app) as api:
                response = api.post('/upload', headers=headers,
                    files={'file': (name, Path(pdf_path).read_bytes(), 'application/pdf')})
                response.raise_for_status()
                print('Upload/indexing passed:', response.json()['chunks_indexed'], 'chunks', flush=True)
                for index, case in enumerate(cases):
                    response = api.post('/chat', headers=headers, json={
                        'question': case['question'], 'session_id': str(index) if isolated else 'sequence',
                        'selected_docs': [name]})
                    data = response.json()
                    answer = data.get('answer', '')
                    ok = (response.status_code == 200
                          and all(s.lower() in answer.lower() for s in case.get('contains', []))
                          and all(s.lower() not in answer.lower() for s in case.get('excludes', []))
                          and all(re.search(p, answer, re.I) for p in case.get('patterns', []))
                          and data.get('sources') == [name + ' p.1'])
                    results.append({'name':case['name'], 'pass':bool(ok), 'answer':answer,
                                    'sources':data.get('sources'), 'status':response.status_code})
                    print(('PASS' if ok else 'FAIL') + ': ' + case['name'], flush=True)
                # Results stay beside the user-supplied private case file.
                output = Path(cases_path).with_suffix('.isolated-results.json' if isolated else '.sequence-results.json')
                output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    finally:
        store.delete_collection()
    print(f"{sum(r['pass'] for r in results)}/{len(results)} passed. Private results saved beside cases.")
    return all(r['pass'] for r in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf')
    parser.add_argument('cases')
    parser.add_argument('--isolated', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.pdf, args.cases, args.isolated) else 1)
