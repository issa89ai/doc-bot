"""Conservative exact comparison of clearly labeled receipt identifiers.

No currency, quantity, or missing character is inferred here.
"""
import re


def single_receipt_item(question, docs):
    """Read an unambiguous single-row Description/Qty table, not product digits."""
    query = question.lower()
    if not (re.search(r'\b(item|items|product|products)\b', query)
            and re.search(r'\b(quantity|how many)\b', query)):
        return None
    sources = {d.metadata.get('source_file') for d in docs}
    if len(sources) != 1 or None in sources:
        return None
    found = set()
    for doc in docs:
        table = re.search(r'Description\s+Qty\s+Unit price(?:\s+Tax\s+Amount)?[ \t]*\n(.*?)(?:\n\s*Subtotal\b|\Z)',
                          doc.page_content, re.I | re.S)
        if not table:
            continue
        lines = [line.strip() for line in table.group(1).splitlines() if line.strip()]
        # Match quantities directly before the unit-price column. Do not read
        # product suffixes as quantities or assume quantity=1 when absent.
        rows = [(i, re.fullmatch(r'(?:(.+?)\s+)?(\d+)\s+[$\u20ac\u00a3]\d[\d,.]*(?:\s+\d+(?:\.\d+)?%\s+[$\u20ac\u00a3]\d[\d,.]*)?', line))
                for i, line in enumerate(lines)]
        rows = [(i, match) for i, match in rows if match]
        if len(rows) != 1:
            return None
        i, match = rows[0]
        name = match.group(1)
        if name is None:
            # Separate service-period line allowed; otherwise ambiguous names
            # or multi-line descriptions go back to the general answer path.
            before = lines[:i]
            if not before or len(before) > 2:
                return None
            if len(before) == 2 and not re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', before[1], re.I):
                return None
            name = before[0]
        if '\x00' in name or '\ufffd' in name:
            return None
        found.add((name, match.group(2)))
    if len(found) == 1:
        name, quantity = found.pop()
        return f'Item: {name}\nQuantity: {quantity}'
    return None


def compare_receipt_identifiers(question, docs):
    query = question.lower()
    if not (re.search(r'\binvoice\b', query) and re.search(r'\breceipt\b', query)
            and re.search(r'\b(number|numbers|identifier|identifiers)\b', query)
            and re.search(r'\b(same|different|equal|identical|compare)\b', query)):
        return None
    sources = {d.metadata.get('source_file') for d in docs}
    if len(sources) != 1 or None in sources:
        return 'Select a single document to compare its invoice and receipt numbers.'
    fields = {}
    for label in ('invoice', 'receipt'):
        # Anchor to a complete labeled line; do not combine fragments across chunks.
        pattern = rf'^\s*{label}[ \t]+(?:number|no\.?)[ \t]*:?[ \t]+([A-Za-z0-9][A-Za-z0-9._/-]*)[ \t]*$'
        values = {match for doc in docs for match in re.findall(pattern, doc.page_content, re.I | re.M)}
        fields[label] = next(iter(values)) if len(values) == 1 else None
    lines = [f'{name.title()} number: {value or "Not unambiguously specified in the retrieved excerpts."}'
             for name, value in fields.items()]
    if all(fields.values()):
        verdict = ('Yes, the invoice and receipt numbers are the same.' if fields['invoice'] == fields['receipt']
                   else 'No, the invoice and receipt numbers are different.')
    else:
        verdict = 'I cannot reliably compare both numbers from the retrieved excerpts.'
    return verdict + '\n' + '\n'.join(lines)
