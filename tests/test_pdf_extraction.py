import sys
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock
import pytest
import rag


@pytest.mark.parametrize('replacement,valid', [('Invoice number DEMO-123', True),
    ('Invoice number DEMO\x00123', False), ('', False)])
def test_fallback_does_not_guess_characters(monkeypatch, replacement, valid):
    page = SimpleNamespace(page_content='Invoice number DEMO\x00123', metadata={})
    loader = Mock(return_value=SimpleNamespace(load=lambda: [page]))
    monkeypatch.setitem(sys.modules, 'langchain_community.document_loaders', SimpleNamespace(PyPDFLoader=loader))
    textpage = Mock()
    textpage.get_text_range.return_value = replacement
    pdfpage = Mock()
    pdfpage.get_textpage.return_value = textpage
    document = MagicMock()
    document.__enter__.return_value = document
    document.__getitem__.return_value = pdfpage
    monkeypatch.setitem(sys.modules, 'pypdfium2', SimpleNamespace(PdfDocument=lambda path: document))
    if valid:
        result = rag.load_pdf_pages('synthetic.pdf')
        assert result[0].page_content == replacement
        assert result[0].metadata['extraction_fallback'] == 'pdfium'
    else:
        with pytest.raises(ValueError, match='unreadable'):
            rag.load_pdf_pages('synthetic.pdf')
    textpage.close.assert_called_once()
    pdfpage.close.assert_called_once()
