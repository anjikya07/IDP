

import re
import logging

logger = logging.getLogger(__name__)

def extract_fields(text: str, doc_type: str = "invoice"):
    """
    Extracts key fields from OCR text of an invoice.
    Currently supports: invoice_number, date, total_amount
    
    Args:
        text (str): OCR extracted text
        doc_type (str): Type of document (currently only "invoice")
        
    Returns:
        dict: Dictionary containing extracted fields
    """
    
    # Validate input
    if not isinstance(text, str):
        logger.error(f"Invalid input type: expected str, got {type(text)}")
        return {"error": "Invalid text input"}
    
    if not text.strip():
        logger.warning("Empty text provided for field extraction")
        return {
            "invoice_number": None,
            "invoice_date": None,
            "total_amount": None,
            "warning": "No text available for extraction"
        }
    
    # Initialize fields dictionary
    fields = {
        "invoice_number": None,
        "invoice_date": None,
        "total_amount": None
    }
    
    try:
        # Clean text for better matching
        cleaned_text = ' '.join(text.split())  # Remove extra whitespace
        
        # Invoice number extraction
        try:
            invoice_patterns = [
                r"(Invoice\s*(No|#|Number)?[:\s]*)\s*([A-Z0-9\-]+)",
                r"(INV\s*(No|#)?[:\s]*)\s*([A-Z0-9\-]+)",
                r"(Bill\s*(No|#)?[:\s]*)\s*([A-Z0-9\-]+)"
            ]
            
            for pattern in invoice_patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    fields["invoice_number"] = match.group(3).strip()
                    logger.info(f"Invoice number found: {fields['invoice_number']}")
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting invoice number: {e}")

        # Date extraction
        try:
            date_patterns = [
                r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b",  # DD/MM/YYYY or MM/DD/YYYY
                r"\b(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b",    # YYYY/MM/DD
                r"\b(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b"  # DD Month YYYY
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if date_match:
                    fields["invoice_date"] = date_match.group(1).strip()
                    logger.info(f"Date found: {fields['invoice_date']}")
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting date: {e}")

        # Total amount extraction
        try:
            amount_patterns = [
                r"(Total\s*(Amount)?[:\s]*)[₹$]?\s?([0-9,]+\.?\d{0,2})",
                r"(Grand\s*Total[:\s]*)[₹$]?\s?([0-9,]+\.?\d{0,2})",
                r"(Amount\s*Due[:\s]*)[₹$]?\s?([0-9,]+\.?\d{0,2})",
                r"[₹$]\s?([0-9,]+\.?\d{0,2})\s*(Total|Final)"
            ]
            
            for i, pattern in enumerate(amount_patterns):
                total_match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if total_match:
                    # Different capture groups for different patterns
                    if i < 3:
                        amount = total_match.group(3) if total_match.group(3) else total_match.group(2)
                    else:
                        amount = total_match.group(1)
                    
                    fields["total_amount"] = amount.strip()
                    logger.info(f"Total amount found: {fields['total_amount']}")
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting total amount: {e}")

        # Log extraction summary
        extracted_count = sum(1 for v in fields.values() if v is not None)
        logger.info(f"Field extraction complete: {extracted_count}/3 fields extracted")
        
        return fields
        
    except Exception as e:
        logger.error(f"Unexpected error during field extraction: {e}")
        return {
            "invoice_number": None,
            "invoice_date": None,
            "total_amount": None,
            "error": f"Extraction failed: {str(e)}"
        }
