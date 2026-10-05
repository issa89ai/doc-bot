"""Isolated Docker smoke test. Usage: python scripts/smoke_docker.py SYNTHETIC_PDF.

Requires doc-bot:local-check and host Ollama. Never mounts the user's documents.
The fixture must contain BLUEBIRD-742. Removes only its own disposable container.
"""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def run(pdf):
    name = 'doc-bot-check-' + secrets.token_hex(6)
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, API_KEY=key)
    container = None
    def docker(*args):
        return subprocess.check_output(['docker', *args], env=env, text=True).strip()
    try:
        container = docker('run', '-d', '--name', name, '--restart=no',
            '-p', '127.0.0.1::8000', '-e', 'API_KEY', '-e', 'S3_BUCKET=',
            '-e', 'OLLAMA_HOST=http://host.docker.internal:11434',
            'doc-bot:local-check')
        def endpoint():
            binding = json.loads(docker('inspect', '--format',
                '{{json .NetworkSettings.Ports}}', container))['8000/tcp'][0]
            return 'http://127.0.0.1:' + binding['HostPort']
        base = endpoint()
        def request(path, method='GET', data=None, content_type='application/json', auth=True):
            headers = {'Content-Type': content_type}
            if auth:
                headers['X-API-Key'] = key
            with urlopen(Request(base + path, data=data, headers=headers, method=method), timeout=180) as r:
                return json.load(r)
        def wait_ready():
            for _ in range(45):
                try:
                    request('/health')
                    break
                except (URLError, ConnectionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Container did not become healthy')
            assert request('/ready')['status'] == 'ready'
        wait_ready()
        print('PASS: container readiness and host Ollama connection', flush=True)
        try:
            request('/documents', auth=False)
            raise AssertionError('Unauthenticated request accepted')
        except HTTPError as exc:
            assert exc.code == 401
        boundary = 'smoke-' + secrets.token_hex(8)
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic.pdf"\r\n'
                'Content-Type: application/pdf\r\n\r\n').encode() + pdf + f'\r\n--{boundary}--\r\n'.encode()
        result = request('/upload', 'POST', body, 'multipart/form-data; boundary=' + boundary)
        assert result['chunks_indexed'] > 0 and result['backup_status'] == 'disabled'
        result = request('/chat', 'POST', json.dumps({'question':'What is the project code?',
            'session_id':'docker-smoke', 'selected_docs':['synthetic.pdf']}).encode())
        assert 'BLUEBIRD-742' in result['answer']
        assert result['sources'] == ['synthetic.pdf p.1']
        print('PASS: authentication, upload, embeddings, real answer, citation; no cloud backup', flush=True)
        docker('restart', container)
        base = endpoint()
        wait_ready()
        assert request('/documents')['documents'] == ['synthetic.pdf']
        print('PASS: document index survives container restart', flush=True)
        request('/documents/synthetic.pdf', 'DELETE')
        assert request('/documents')['documents'] == []
        print('PASS: document deletion', flush=True)
    finally:
        if container:
            docker('rm', '-f', container)
            print('Test container removed; existing containers and user data untouched.', flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]).read_bytes())
