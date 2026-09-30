"""Receipt grounding regression. Live checks are opt-in and use synthetic text."""
import os
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import rag

RECEIPT = 'Item: Max plan - 20x. Subtotal: $200.00. Tax: 13%, $26.00. Total: $226.00. Payment method: Mastercard. Merchant: San Francisco.'
QUESTION = 'List the purchased item, subtotal, tax, total, and currency.'


def retrieval(monkeypatch, text):
    doc = SimpleNamespace(page_content=text, metadata={'source_file':'receipt.pdf', 'page':0})
    monkeypatch.setattr(rag, 'get_retriever', lambda selected: SimpleNamespace(invoke=lambda question: [doc]))


def test_missing_field_rules_reach_model(monkeypatch):
    retrieval(monkeypatch, RECEIPT)
    model = Mock()
    model.invoke.return_value.content = 'Currency: Not specified in the provided document.'
    monkeypatch.setattr(rag, 'llm', lambda: model)
    answer, sources = rag.answer(QUESTION, [('What currency?', 'It is USD.')])
    system = model.invoke.call_args.args[0].to_messages()[0].content
    assert 'a bare $ symbol does not identify' in system
    assert 'Never infer currency from a card brand' in system
    assert 'Conversation history helps interpret the question but is not evidence' in system
    assert RECEIPT in system
    assert 'It is USD.' not in model.invoke.call_args.args[0].to_string()
    assert sources == ['receipt.pdf p.1']


def test_unrelated_previous_question_is_not_repeated(monkeypatch):
    retrieval(monkeypatch, RECEIPT)
    model = Mock()
    model.invoke.return_value.content = 'Subtotal: $200.00'
    monkeypatch.setattr(rag, 'llm', lambda: model)
    rag.answer('What are the subtotal, tax, and total?',
               [('What item was purchased, and how many?', 'Quantity: 20x')])
    prompt = model.invoke.call_args.args[0].to_string()
    assert 'What item was purchased' not in prompt
    assert 'Quantity: 20x' not in prompt


def test_explicit_followup_keeps_user_context(monkeypatch):
    retrieval(monkeypatch, RECEIPT)
    model = Mock()
    model.invoke.return_value.content = 'Summary'
    monkeypatch.setattr(rag, 'llm', lambda: model)
    rag.answer('Summarize it please', [('Tell me about the plan', 'UNTRUSTED_PREVIOUS_OUTPUT')])
    prompt = model.invoke.call_args.args[0].to_string()
    assert 'Tell me about the plan' in prompt
    assert 'UNTRUSTED_PREVIOUS_OUTPUT' not in prompt


def test_stale_corrupted_index_requests_reupload(monkeypatch):
    retrieval(monkeypatch, 'Invoice number DEMO\x00123')
    model = Mock()
    monkeypatch.setattr(rag, 'llm', lambda: model)
    answer, _ = rag.answer('What is the invoice number?', [])
    assert 'upload' in answer and 'unreadable' in answer
    model.invoke.assert_not_called()


@pytest.mark.skipif(os.getenv('RUN_OLLAMA_TESTS') != '1', reason='Opt-in real local model test')
@pytest.mark.parametrize('text,history,question,expected', [
    (RECEIPT, [], QUESTION, 'not specified'),
    (RECEIPT, [('What currency?', 'It is USD.')], QUESTION, 'not specified'),
    (RECEIPT + ' Currency: CAD.', [], 'What is the currency?', 'cad'),
    (RECEIPT, [], 'What is the refund policy?', 'not specified'),
    ('Description Qty Unit price\nStarter plan - 50x 3 $12.00', [], 'What item was purchased, and how many?', '3'),
])
def test_live_receipt_grounding(monkeypatch, text, history, question, expected):
    monkeypatch.setenv('OLLAMA_HOST', 'http://localhost:11434')
    rag.llm.cache_clear()
    retrieval(monkeypatch, text)
    answer, _ = rag.answer(question, history)
    assert expected in answer.lower(), answer
    if expected == 'not specified':
        assert 'usd' not in answer.lower(), answer
        assert 'united states dollar' not in answer.lower(), answer
