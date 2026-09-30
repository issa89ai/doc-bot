let apiKey = '';
let busy = false;
const newId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
const sessionId = newId();
const el = id => document.getElementById(id);
const status = text => { el('status').textContent = text; };

async function request(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...options.headers, 'X-API-Key': apiKey}});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  return data;
}

function message(role, text, sources = []) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  if (sources.length) {
    const label = document.createElement('div');
    label.className = 'sources'; label.textContent = `Retrieved pages: ${sources.join(', ')}`;
    div.append(label);
  }
  el('messages').append(div);
  div.scrollIntoView({block: 'nearest'});
}

async function loadDocs() {
  const {documents} = await request('/documents');
  const container = el('doc-items');
  container.replaceChildren();
  for (const name of documents) {
    const row = document.createElement('div'); row.className = 'doc';
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = true; check.value = name; check.id = newId();
    const label = document.createElement('label'); label.htmlFor = check.id; label.textContent = name;
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'delete-doc'; remove.textContent = 'Delete PDF'; remove.setAttribute('aria-label', `Delete ${name}`);
    remove.addEventListener('click', async () => {
      if (!confirm(`Delete ${name}, including its current S3 copy?`)) return;
      remove.disabled = true; remove.textContent = 'Deleting…';
      try { await request(`/documents/${encodeURIComponent(name)}`, {method:'DELETE'}); await loadDocs(); status('Document deleted.'); }
      catch (error) { status(error.message); } finally { remove.disabled = false; remove.textContent = 'Delete PDF'; }
    });
    row.append(check, label, remove); container.append(row);
  }
}

el('connect-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!window.isSecureContext) { status('Use HTTPS (or localhost) before entering your private key.'); return; }
  apiKey = el('api-key').value;
  try {
    await loadDocs(); el('api-key').value = '';
    el('file-input').disabled = false; el('send-btn').disabled = false; el('clear-btn').disabled = false;
    try { await request('/ready'); status('Connected. AI service is ready.'); }
    catch (error) { status(`Connected, but ${error.message}`); }
  } catch (error) { apiKey = ''; el('file-input').disabled = true; el('send-btn').disabled = true; el('clear-btn').disabled = true; status(error.message); }
});

el('file-input').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 10 * 1024 * 1024) { status('Maximum PDF size is 10 MB.'); return; }
  event.target.disabled = true; status('Uploading and indexing…');
  try {
    const fd = new FormData(); fd.append('file', file);
    const data = await request('/upload', {method:'POST', body:fd});
    await loadDocs(); status(`${data.chunks_indexed} chunks indexed. Cloud backup: ${data.backup_status}.`);
  } catch (error) { status(error.message); }
  finally { event.target.disabled = false; event.target.value = ''; }
});

el('question').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  if (!event.repeat && !busy && apiKey && !el('send-btn').disabled) {
    sendMessage();
  }
});

el('chat-form').addEventListener('submit', event => {
  event.preventDefault();
  sendMessage();
});

async function sendMessage() {
  if (busy || !apiKey) return;
  const question = el('question').value.trim(); if (!question) return;
  const selected_docs = [...document.querySelectorAll('#doc-items input:checked')].map(x => x.value);
  if (!selected_docs.length) { status('Select at least one document.'); return; }
  busy = true; el('send-btn').disabled = true; message('user', question); status('Thinking…');
  try {
    const data = await request('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question, session_id:sessionId, selected_docs})});
    message('bot', data.answer, data.sources); el('question').value = ''; status('Answer ready.');
  } catch (error) { message('bot', error.message); status('Request failed. You can retry.'); }
  finally { busy = false; el('send-btn').disabled = false; }
}

el('clear-btn').addEventListener('click', async () => {
  try { await request(`/session/${sessionId}`, {method:'DELETE'}); el('messages').replaceChildren(); status('Conversation cleared.'); }
  catch (error) { status(error.message); }
});
