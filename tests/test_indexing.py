"""Exercise replacement failures without connecting to Ollama or Chroma."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import rag


@pytest.fixture
def setup_index(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, 'DOCS_DIR', str(tmp_path / 'docs'))
    destination = tmp_path / 'docs' / 'a.pdf'
    destination.parent.mkdir()
    destination.write_bytes(b'old document')
    source = tmp_path / 'incoming.pdf'
    source.write_bytes(b'new document')
    page = SimpleNamespace(metadata={}, page_content='Some PDF text')
    loader = Mock(return_value=SimpleNamespace(load=lambda: [page]))
    splitter = Mock(return_value=SimpleNamespace(split_documents=lambda pages: pages))
    monkeypatch.setitem(sys.modules, 'langchain_community.document_loaders', SimpleNamespace(PyPDFLoader=loader))
    monkeypatch.setitem(sys.modules, 'langchain_text_splitters', SimpleNamespace(RecursiveCharacterTextSplitter=splitter))
    store = Mock()
    store.get.return_value = {'ids':['old-id']}
    monkeypatch.setattr(rag, 'vectorstore', lambda: store)
    return source, destination, store


def test_reupload_removes_old_chunks(setup_index):
    source, destination, store = setup_index
    assert rag.load_and_index(str(source), 'a.pdf') == 1
    assert destination.read_bytes() == b'new document'
    store.delete.assert_called_once_with(ids=['old-id'])


def test_failed_embedding_keeps_old_document(setup_index):
    source, destination, store = setup_index
    store.add_documents.side_effect = RuntimeError('Ollama offline')
    with pytest.raises(RuntimeError):
        rag.load_and_index(str(source), 'a.pdf')
    assert destination.read_bytes() == b'old document'
    assert 'old-id' not in store.delete.call_args.kwargs['ids']
    assert not list(destination.parent.glob('*.upload'))


def test_list_comes_from_index(monkeypatch):
    store = Mock()
    store.get.return_value = {'metadatas':[{'source_file':'a.pdf'}, {'source_file':'a.pdf'}, None]}
    monkeypatch.setattr(rag, 'vectorstore', lambda: store)
    assert rag.list_indexed_files() == ['a.pdf']


def test_empty_selection_never_searches_all():
    with pytest.raises(ValueError):
        rag.get_retriever([])
