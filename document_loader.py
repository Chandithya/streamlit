"""Module 1 & 2: PDF validation and text extraction with page metadata."""
import re
from pypdf import PdfReader
from langchain_core.documents import Document

MAX_FILE_MB = 10


def validate_file(uploaded_file):
    """Return (is_valid, message). Checks type and size."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, f"{uploaded_file.name}: only PDF files are allowed."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, f"{uploaded_file.name}: file is {size_mb:.1f} MB (limit {MAX_FILE_MB} MB)."
    return True, "OK"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(file, filename: str):
    """Read every page; keep document name and page number as metadata.
    Empty pages are skipped safely."""
    docs = []
    try:
        reader = PdfReader(file)
        if reader.is_encrypted:
            reader.decrypt("")
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:
                text = ""
            if not text:
                continue  # skip empty / scanned pages
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_number},
                )
            )
    except Exception as e:
        raise ValueError(f"Could not read {filename}: {e}")
    return docs
