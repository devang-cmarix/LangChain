# backend/ingest.py
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain_core.documents import Document

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_page_lines(pdf_path: str, page_number: int) -> List[str]:
    """Return list of lines for a single page (page_number is 0-indexed)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number]
        text = page.get_text("text")  # returns page text with newlines
        lines = text.splitlines()
        # normalize blank lines
        lines = [ln for ln in lines if ln.strip() != ""]
        return lines
    finally:
        doc.close()

def chunk_lines_with_sliding_window(lines: List[str], *, target_chars: int = 1000, overlap_ratio: float = 0.2) -> List[Dict[str, Any]]:
    """
    Create chunks from page lines aiming for ~target_chars per chunk, with overlap_ratio sliding window.
    Returns list of dicts: {"start_line": int, "end_line": int, "text": str}
    """
    if not lines:
        return []
    n = len(lines)
    chunks = []
    i = 0
    while i < n:
        cur_lines = []
        cur_chars = 0
        j = i
        while j < n and (cur_chars < target_chars or len(cur_lines) == 0):
            cur_lines.append(lines[j])
            cur_chars += len(lines[j]) + 1
            j += 1
        start_line = i + 1  # 1-indexed
        end_line = j        # inclusive as lines[j-1] was last added
        text = "\n".join(cur_lines)
        chunks.append({"start_line": start_line, "end_line": end_line, "text": text})
        chunk_line_count = end_line - start_line + 1
        overlap_lines = max(1, int(chunk_line_count * overlap_ratio))
        # advance window
        i = j - overlap_lines
    return chunks

def pdf_to_documents(pdf_path: str, pdf_name: str, *,
                     chunk_size_chars: int = 1000,
                     overlap_ratio: float = 0.2) -> List[Document]:
    """
    Parse pdf, chunk pages with sliding window, and return LangChain Documents with metadata:
      {"pdf_name", "page_num", "start_line", "end_line", "chunk_index"}
    """
    docs = []
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    for p in range(total_pages):
        lines = extract_page_lines(pdf_path, p)
        chunks = chunk_lines_with_sliding_window(lines, target_chars=chunk_size_chars, overlap_ratio=overlap_ratio)
        for idx, chunk in enumerate(chunks):
            metadata = {
                "pdf_name": pdf_name,
                "page_num": p + 1,  # 1-indexed
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_index": idx
            }
            docs.append(Document(page_content=chunk["text"], metadata=metadata))
    return docs
