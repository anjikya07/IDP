# pipeline/extractor.py

import re

def extract_fields(text: str, doc_type: str = "invoice"):
    """
    Extracts key fields from OCR text of an invoice.
    Currently supports: invoice_number, date, total_amount
    """
    fields = {
        "invoice_number": None,
        "invoice_date": None,
        "total_amount": None
    }

    # Invoice number
    match = re.search(r"(Invoice\s*(No|#|Number)?[:\s]*)\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if match:
        fields["invoice_number"] = match.group(3)

    # Date (supports formats like 2024-06-01, 01/06/2024, etc.)
    date_match = re.search(r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b", text)
    if date_match:
        fields["invoice_date"] = date_match.group(0)

    # Total amount (looks for "Total" followed by ₹ or $ or digits)
    total_match = re.search(r"(Total\s*(Amount)?[:\s]*)[₹$]?\s?([0-9,]+\.\d{2})", text, re.IGNORECASE)
    if total_match:
        fields["total_amount"] = total_match.group(3)

    return fields
