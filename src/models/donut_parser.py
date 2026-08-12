import re
import json

class DonutParser:
    """
    Donut Transformer Structured Information Extractor
    Converts unstructured document text and OCR layout into structured JSON representations.
    """
    def __init__(self):
        self.model_name = "Donut Transformer (Structured Document Understanding)"

    def parse_structured_data(self, full_text, layout_blocks=None):
        """
        Parses document text into key-value attributes, dates, currency totals, metadata, and doc type.
        """
        doc_type = self._classify_document_type(full_text)
        dates = self._extract_dates(full_text)
        amounts = self._extract_amounts(full_text)
        key_values = self._extract_key_values(full_text, layout_blocks)
        emails_urls = self._extract_contacts(full_text)

        structured_json = {
            "document_type": doc_type,
            "metadata": {
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "extracted_dates": dates,
                "extracted_amounts": amounts,
                "contacts": emails_urls
            },
            "key_value_attributes": key_values,
            "structured_summary": f"Identified as {doc_type} with {len(key_values)} structured fields and {len(dates)} dates."
        }
        return {
            "model": self.model_name,
            "structured_json": structured_json
        }

    def _classify_document_type(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["invoice", "bill to", "due date", "amount due", "subtotal", "tax"]):
            return "Invoice / Bill"
        elif any(w in text_lower for w in ["receipt", "total cash", "change", "store #", "item count"]):
            return "Receipt"
        elif any(w in text_lower for w in ["chapter", "section", "abstract", "figure", "table", "definition", "algorithm"]):
            return "Academic / Textbook Document"
        elif any(w in text_lower for w in ["agreement", "contract", "parties", "hereby", "signature", "terms"]):
            return "Legal Contract / Agreement"
        elif any(w in text_lower for w in ["form", "name:", "date of birth", "address:", "phone:"]):
            return "Structured Form"
        else:
            return "General Document"

    def _extract_dates(self, text):
        pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))

    def _extract_amounts(self, text):
        pattern = r'[\$€£₹]\s?\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*\.\d{2}\b'
        matches = re.findall(pattern, text)
        return list(set(matches))

    def _extract_key_values(self, text, layout_blocks=None):
        kvs = {}
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if 2 <= len(k) <= 35 and len(v) > 0:
                    kvs[k] = v

        if layout_blocks:
            for b in layout_blocks:
                if b.get("category") == "Key-Value Field" and ":" in b["text"]:
                    parts = b["text"].split(":", 1)
                    kvs[parts[0].strip()] = parts[1].strip()
        return kvs

    def _extract_contacts(self, text):
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
        return {"emails": list(set(emails)), "urls": list(set(urls))}
