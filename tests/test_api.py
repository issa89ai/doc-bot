import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import main
import rag

KEY = 'test-only-not-a-real-secret-' + 'x' * 32
HEADERS = {'X-API-Key': KEY}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('API_KEY', KEY)
    monkeypatch.delenv('S3_BUCKET', raising=False)
    monkeypatch.setattr(rag, 'DOCS_DIR', str(tmp_path / 'docs'))
    main.sessions.clear()
    with TestClient(main.app) as client:
        yield client


@pytest.mark.parametrize('method,path', [('get','/documents'), ('get','/metrics'), ('get','/ready'),
    ('delete','/session/example'), ('delete','/documents/example.pdf'), ('post','/chat')])
def test_protected_routes(client, method, path):
    kwargs = {'json': {'question':'Hi', 'session_id':'s'}} if method == 'post' else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_upload_requires_key(client):
    assert client.post('/upload', files={'file': ('test.pdf', b'%PDF-1.4')}).status_code == 401


def test_ui_contains_no_key(client):
    response = client.get('/')
    assert response.status_code == 200
    assert KEY not in response.text
    assert 'docbot-secret-123' not in client.get('/static/app.js').text
    assert 'Content-Security-Policy' in response.headers


@pytest.mark.parametrize('filename', ['../escape.pdf', '..\\escape.pdf', '/tmp/escape.pdf', 'C:\\escape.pdf', '<script>.pdf', 'CON.pdf'])
def test_unsafe_filename(filename):
    with pytest.raises(ValueError):
        rag.validate_filename(filename)


def test_upload_path_traversal(client):
    assert client.post('/upload', headers=HEADERS,
                       files={'file': ('../escape.pdf', b'%PDF-1.4')}).status_code == 400


@pytest.mark.parametrize('content', [b'', b'not a pdf'])
def test_non_pdf(client, content):
    assert client.post('/upload', headers=HEADERS, files={'file': ('a.pdf', content)}).status_code == 400


def test_large_pdf(client, monkeypatch):
    monkeypatch.setattr(main, 'MAX_UPLOAD_BYTES', 10)
    assert client.post('/upload', headers=HEADERS,
                       files={'file': ('a.pdf', b'%PDF-' + b'x' * 100)}).status_code == 413


def test_backup_failure_is_visible(client):
    with patch.object(rag, 'load_and_index', return_value=2), patch.object(main, 'backup_pdf', side_effect=RuntimeError):
        response = client.post('/upload', headers=HEADERS, files={'file': ('a.pdf', b'%PDF-1.4')})
    assert response.status_code == 200
    assert response.json()['backup_status'] == 'failed'


def test_empty_selection_rejected(client):
    response = client.post('/chat', headers=HEADERS, json={'question':'Hi', 'session_id':'s', 'selected_docs':[]})
    assert response.status_code == 400


def test_model_failure_is_safe(client):
    with patch.object(rag, 'answer', side_effect=RuntimeError('sensitive internal detail')):
        response = client.post('/chat', headers=HEADERS, json={'question':'Hi', 'session_id':'s'})
    assert response.status_code == 503
    assert 'sensitive' not in response.text


def test_upload_diagnostics_do_not_leak_exception_text(client, caplog):
    with patch.object(rag, 'load_and_index', side_effect=RuntimeError('PRIVATE_DOCUMENT_OR_KEY')):
        response = client.post('/upload', headers=HEADERS,
                               files={'file': ('a.pdf', b'%PDF-1.4')})
    assert response.status_code == 503
    assert 'PDF indexing [RuntimeError]' in caplog.text
    assert 'PRIVATE_DOCUMENT_OR_KEY' not in caplog.text + response.text


def test_readiness_failure(client):
    with patch.object(rag, 'check_ready', side_effect=RuntimeError):
        assert client.get('/ready', headers=HEADERS).status_code == 503
    assert client.get('/health').status_code == 200


def test_history_bounded(client):
    with patch.object(rag, 'answer', return_value=('answer', [])):
        for _ in range(5):
            response = client.post('/chat', headers=HEADERS, json={'question':'Hi', 'session_id':'s'})
    assert response.json()['turn'] == 5
    assert len(main.sessions['s'][0]) == 3


def test_cloud_delete_failure_preserves_local(client):
    with patch.object(main, 'delete_backup', side_effect=RuntimeError), patch.object(rag, 'delete_document') as delete:
        assert client.delete('/documents/a.pdf', headers=HEADERS).status_code == 503
        delete.assert_not_called()


def test_delete_document_succeeds_and_clears_history(client):
    main.sessions['example'] = ([('question', 'answer')], 1)
    with patch.object(main, 'delete_backup') as backup, patch.object(rag, 'delete_document') as delete:
        response = client.delete('/documents/example.pdf', headers=HEADERS)
    assert response.status_code == 200
    backup.assert_called_once_with('example.pdf')
    delete.assert_called_once_with('example.pdf')
    assert not main.sessions


def test_reject_weak_key(monkeypatch):
    monkeypatch.setenv('API_KEY', 'docbot-secret-123')
    with pytest.raises(RuntimeError):
        main.configured_key()
