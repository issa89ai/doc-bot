from types import SimpleNamespace
import pytest
from exact_fields import compare_receipt_identifiers, single_receipt_item

QUESTION = 'Are the invoice number and receipt number the same? List both.'


def doc(text, name='sample.pdf'):
    return SimpleNamespace(page_content=text, metadata={'source_file': name})


@pytest.mark.parametrize('invoice,receipt,same', [('INV-AX9', 'REC-77', False),
    ('ABC-123', 'ABC-123', True), ('007', '7', False), ('a-1', 'A-1', False)])
def test_comparison(invoice, receipt, same):
    answer = compare_receipt_identifiers(QUESTION, [doc(f'Invoice number {invoice}\nReceipt number {receipt}')])
    assert answer.startswith('Yes' if same else 'No')
    assert invoice in answer and receipt in answer


@pytest.mark.parametrize('text', ['Invoice number INV-1',
    'Invoice number INV\x001\nReceipt number REC-2',
    'Invoice number INV-1\nInvoice number INV-2\nReceipt number REC-3'])
def test_missing_corrupt_or_conflicting_values(text):
    assert compare_receipt_identifiers(QUESTION, [doc(text)]).startswith('I cannot')


def test_no_cross_document_comparison():
    assert 'single document' in compare_receipt_identifiers(QUESTION,
        [doc('Invoice number INV-1', 'a.pdf'), doc('Receipt number REC-2', 'b.pdf')])


def test_unrelated_question_uses_model():
    assert compare_receipt_identifiers('What is the subtotal?', [doc('Subtotal $200')]) is None


@pytest.mark.parametrize('row,name,quantity', [
    ('Starter 50x 3 $12.00 10% $36.00', 'Starter 50x', '3'),
    ('Pro - 80x\nOct 1-Oct 31, 2026\n2 $15.00 5% $30.00', 'Pro - 80x', '2'),
    ('Example - 5x\n1 $9.00 0% $9.00', 'Example - 5x', '1'),
])
def test_quantity_is_not_product_suffix(row, name, quantity):
    text = 'Description Qty Unit price Tax Amount\n' + row + '\nSubtotal $36.00'
    assert single_receipt_item('What item was purchased, and how many?', [doc(text)]) == f'Item: {name}\nQuantity: {quantity}'


def test_multiple_rows_are_not_silently_reduced():
    text = 'Description Qty Unit price Tax Amount\nA 1 $10.00 0% $10.00\nB 2 $10.00 0% $20.00\nSubtotal $30.00'
    assert single_receipt_item('What items and how many?', [doc(text)]) is None


def test_missing_quantity_not_assumed():
    text = 'Description Qty Unit price Tax Amount\nPlan 20x\nSubtotal $20.00'
    assert single_receipt_item('What item and how many?', [doc(text)]) is None
