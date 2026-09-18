"""
file_parser.py — Extract clean text from PDF, DOCX, and TXT files.
Includes defenses against PDF steganography (white/tiny text) and extraction of DOCX tables.
"""

import io
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Basic text cleaning: collapse extra whitespace, remove null bytes.
    Preserves newlines for section separation.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_extract_text(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Safely extract plain text from file bytes with comprehensive error handling.

    Returns:
        {
            "success": bool,
            "text": str,
            "chars_extracted": int,
            "error_type": str | None,
            "error_message": str | None
        }
    """
    if not file_bytes or len(file_bytes) == 0:
        return {
            "success": False,
            "text": "",
            "chars_extracted": 0,
            "error_type": "empty_file",
            "error_message": "Uploaded file is 0 bytes.",
        }

    ext = (filename or "").rsplit(".", 1)[-1].lower()

    if ext not in ("pdf", "docx", "doc", "txt"):
        return {
            "success": False,
            "text": "",
            "chars_extracted": 0,
            "error_type": "unsupported_format",
            "error_message": f"Unsupported file extension '.{ext}'. Supported: pdf, docx, txt.",
        }

    raw_text = ""
    error_type = None
    error_message = None

    try:
        if ext == "pdf":
            raw_text = _parse_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            raw_text = _parse_docx(file_bytes)
        elif ext == "txt":
            raw_text = _parse_txt(file_bytes)
    except Exception as e:
        logger.exception("Error parsing file %s: %s", filename, str(e))
        error_type = "corrupted_file"
        error_message = f"Failed to parse file: {str(e)}"

    cleaned = clean_text(raw_text)

    # Scanned / Image-based PDF or empty extraction detection
    if not cleaned or len(cleaned) < 50:
        if not error_type:
            error_type = "unreadable_or_scanned"
            error_message = (
                "Could not extract sufficient text from file (less than 50 characters). "
                "The file may be scanned, image-only, or empty."
            )
        return {
            "success": False,
            "text": cleaned,
            "chars_extracted": len(cleaned),
            "error_type": error_type,
            "error_message": error_message,
        }

    return {
        "success": True,
        "text": cleaned,
        "chars_extracted": len(cleaned),
        "error_type": None,
        "error_message": None,
    }


def _parse_pdf(data: bytes) -> str:
    """
    Extract text from PDF using PyPDF2 with pdfminer layout filter fallback.
    Strips invisible/white text and tiny font text (< 4pt).
    """
    # 1. First try PyPDF2
    text_pypdf = ""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("PDF is password protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
        text_pypdf = "\n".join(pages)
    except Exception as e:
        logger.warning("PyPDF2 parse notice: %s", e)

    # 2. Try pdfminer to filter steganography (white text / tiny font) if possible
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LTChar

        filtered_text_chunks = []
        for page_layout in extract_pages(io.BytesIO(data)):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        line_chars = []
                        for character in text_line:
                            if isinstance(character, LTChar):
                                # Font size check: ignore font sizes smaller than 4pt (often used for prompt injection)
                                if character.size < 4.0:
                                    continue

                                # Color check: if ncolor is pure white / near white (1.0, 1.0, 1.0)
                                ncolor = getattr(character, "ncolor", None)
                                if ncolor is not None:
                                    if isinstance(ncolor, (list, tuple)) and len(ncolor) >= 3:
                                        r, g, b = ncolor[:3]
                                        if r > 0.98 and g > 0.98 and b > 0.98:
                                            continue  # White text stripped
                                    elif isinstance(ncolor, (int, float)) and ncolor > 0.98:
                                        continue  # Grayscale white stripped

                                line_chars.append(character.get_text())
                            else:
                                line_chars.append(character.get_text())
                        filtered_text_chunks.append("".join(line_chars))

        pdfminer_text = "\n".join(filtered_text_chunks).strip()
        if len(pdfminer_text) > 50:
            return pdfminer_text
    except Exception as e:
        logger.debug("pdfminer filtering fallback ignored: %s", e)

    return text_pypdf


def _parse_docx(data: bytes) -> str:
    """
    Extract text from DOCX bytes including paragraphs and table cells.
    """
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        text_parts = []

        # Extract standard paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract text inside tables (resumes frequently format skills & history in tables)
        for table in doc.tables:
            for row in table.rows:
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_texts:
                    # Deduplicate adjacent cells if merged
                    seen_cell = []
                    for c in row_texts:
                        if not seen_cell or seen_cell[-1] != c:
                            seen_cell.append(c)
                    text_parts.append(" | ".join(seen_cell))

        return "\n".join(text_parts)
    except Exception as e:
        logger.error("DOCX parse error: %s", e)
        raise


def _parse_txt(data: bytes) -> str:
    """
    Extract plain text from bytes using utf-8 with fallback encodings.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")
