"""Extract document text and find relevant passages while keeping page references."""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from app.rag.intents import is_document_overview

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 50
MAX_DOCUMENTS = 5
# Keep documents in small chunks; only retrieved excerpts go to the model.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
STOP_WORDS = {"the", "a", "an", "is", "are", "in", "of", "to", "and", "what", "how",
              "does", "do", "it", "this", "that", "from", "my", "pdf", "document", "uploaded"}


def requested_page(text: str) -> int | None:
    """Physical PDF page numbers, not printed chapter/page labels."""
    match = re.search(r"\b(?:page|pg\.?)\s*(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+page\b", text, re.I)
    if match:
        return int(next(value for value in match.groups() if value is not None))
    words = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
             'pehla': 1, 'pehle': 1, 'pahla': 1}
    match = re.search(r"\b(" + '|'.join(words) + r")\s+page\b", text, re.I)
    return words[match.group(1).lower()] if match else None


class DocumentError(ValueError):
    """Safe extraction/retrieval error."""


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOP_WORDS]


@dataclass(frozen=True)
class Chunk:
    document: str
    page: int
    chunk_id: str
    text: str


def extract_pdf(name: str, content: bytes) -> list[Chunk]:
    name = name.replace("\\", "/").split("/")[-1]
    if not name.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise DocumentError("Upload a valid PDF file.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("PDF must be at most 10 MB.")
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise DocumentError("Password-protected PDFs are not supported.")
        if len(reader.pages) > MAX_PAGES:
            raise DocumentError("PDF must contain at most 50 pages.")
        chunks = []
        digest = hashlib.sha256(content).hexdigest()[:12]
        for page_number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            for offset in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                excerpt = text[offset:offset + CHUNK_SIZE]
                if excerpt:
                    chunks.append(Chunk(name, page_number, f"{digest}-p{page_number}-{offset}", excerpt))
                if offset + CHUNK_SIZE >= len(text):
                    break
        if not chunks:
            raise DocumentError("No readable text found. Scanned/image PDFs need OCR, which is not included yet.")
        return chunks
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("Could not read this PDF. It may be damaged or unsupported.") from None


class DocumentIndex:
    """Keep extracted text in the current session for local document search."""

    def __init__(self) -> None:
        self._documents: dict[str, tuple[str, list[Chunk]]] = {}

    def add(self, name: str, content: bytes) -> dict:
        name = name.replace("\\", "/").split("/")[-1]
        digest = hashlib.sha256(content).hexdigest()
        if name in self._documents and self._documents[name][0] == digest:
            return {"document": name, "status": "already_indexed"}
        if len(self._documents) >= MAX_DOCUMENTS and name not in self._documents:
            raise DocumentError("Upload at most five text documents per chat.")
        chunks = extract_pdf(name, content)
        # Replace the old document only after the new one has been read successfully.
        self._documents[name] = (digest, chunks)
        return {"document": name, "status": "indexed", "chunks": len(chunks)}

    def add_sections(self, name: str, sections: list[str]) -> None:
        """Index text sections without treating them as Word page numbers."""
        if len(self._documents) >= MAX_DOCUMENTS and name not in self._documents:
            raise DocumentError("Upload at most five text documents per chat.")
        chunks = []
        digest = hashlib.sha256("\n".join(sections).encode()).hexdigest()
        for section, text in enumerate(sections, 1):
            text = " ".join(text.split())
            for offset in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(Chunk(name, section, f"{digest[:12]}-s{section}-{offset}", text[offset:offset+CHUNK_SIZE]))
                if offset + CHUNK_SIZE >= len(text):
                    break
        if not chunks:
            raise DocumentError("No readable text found in the document.")
        self._documents[name] = (digest, chunks)

    def clear(self) -> None:
        self._documents.clear()

    def documents(self) -> list[dict]:
        return [{"document": name, "text_pages": len({c.page for c in chunks}), "chunks": len(chunks)}
                for name, (_, chunks) in self._documents.items()]

    def search(self, query: str, document: str = "", page: str = "") -> dict:
        query = query.strip()
        if not query or len(query) > 500:
            raise DocumentError("Document query must contain 1–500 characters.")
        if not self._documents:
            return {"status": "no_documents", "passages": [], "message": "Upload and index a PDF first."}
        chunks = [chunk for name, (_, items) in self._documents.items()
                  if not document or name == document for chunk in items]
        if not chunks:
            return {"status": "no_results", "passages": [], "message": "No matching uploaded document."}
        selected_page = int(page) if page.isdecimal() else requested_page(query) if not page else None
        if page and (not page.isdecimal() or int(page) < 1):
            raise DocumentError("Page must be a positive physical page number.")
        if selected_page is not None:
            pdf_names = {c.document for c in chunks if c.document.lower().endswith('.pdf')}
            if len(pdf_names) != 1:
                return {"status":"ambiguous_document", "passages":[],
                        "message":"Choose the exact PDF filename for the requested page."}
            selected = [c for c in chunks if c.page == selected_page and c.document in pdf_names]
            if not selected:
                return {"status":"no_readable_page", "passages":[], "requested_page":selected_page,
                        "message":"Requested page has no readable text or is outside the PDF. No other page was substituted."}
            return {"status":"ok", "method":"exact_page", "requested_page":selected_page,
                    "coverage":"First 12 chunks of the requested physical PDF page only.",
                    "passages":[{"document":c.document,"page":c.page,"chunk_id":c.chunk_id,"text":c.text}
                                for c in selected[:12]]}
        normalized = re.sub(r"[^\w\s]", "", query.lower())
        overview_queries = {
            "summary", "summarize", "summarise", "overview",
            "what is the pdf about", "what is this pdf about",
            "what is the pdf is about", "what the pdf is about",
            "what is this document about", "what is the document about",
            "summarize the pdf", "summarize this pdf", "summary of the pdf",
        }
        if normalized in overview_queries or is_document_overview(query):
            return self.overview(chunks)
        terms = set(tokens(query))
        token_lists = [tokens(chunk.text) for chunk in chunks]
        counts = [Counter(words) for words in token_lists]
        average = sum(map(len, token_lists)) / len(chunks) or 1
        scores = [0.0] * len(chunks)
        for term in terms:
            frequency = sum(term in count for count in counts)
            inverse_frequency = math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for index, count in enumerate(counts):
                occurrences = count[term]
                if occurrences:
                    length_adjustment = 1.5 * (1 - 0.75 + 0.75 * len(token_lists[index]) / average)
                    scores[index] += inverse_frequency * occurrences * 2.5 / (occurrences + length_adjustment)
        ranked = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
        passages = [
            {"document": chunks[i].document, "page": chunks[i].page,
             "chunk_id": chunks[i].chunk_id, "text": chunks[i].text}
            for i in ranked[:3] if scores[i] > 0
        ]
        return {"status": "ok" if passages else "no_results", "method": "BM25", "passages": passages}

    def overview(self, chunks: list[Chunk]) -> dict:
        """Return opening, middle and ending excerpts for a short overview."""
        indices = sorted({0, len(chunks) // 2, len(chunks) - 1})
        return {
            "status": "ok", "method": "sampled_overview",
            "coverage": "Opening, middle and ending excerpts only; not an exhaustive summary.",
            "total_chunks": len(chunks),
            "passages": [
                {"document": chunks[i].document, "page": chunks[i].page,
                 "chunk_id": chunks[i].chunk_id, "text": chunks[i].text}
                for i in indices
            ],
        }