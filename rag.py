import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np

from pdf_importer import extract_text_from_pdf_bytes, io_bytes


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    page: int
    text: str


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def chunk_text_by_pages(pdf_bytes: bytes, max_pages: int = 20) -> list[tuple[int, str]]:
    """
    Returns list of (page_number_1_indexed, page_text).
    """
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(io_bytes(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append((i + 1, txt))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for page_num, text in pages:
        start = 0
        text = " ".join(text.split())
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"p{page_num}_c{len(chunks)}"
                chunks.append(RagChunk(chunk_id=chunk_id, page=page_num, text=chunk_text))
            if end >= len(text):
                break
            start = max(0, end - overlap_chars)
    return chunks


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    embedder_name = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = _get_embedder(embedder_name)
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def index_dir(base_dir: Path) -> Path:
    d = base_dir / "data" / "rag_indexes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_or_load_index(
    pdf_bytes: bytes,
    base_dir: Path,
    max_pages: int = 20,
) -> tuple[str, faiss.Index, list[RagChunk]]:
    """
    Persists FAISS + chunks to disk using a doc_id derived from the PDF bytes.
    """
    doc_id = _sha1_bytes(pdf_bytes)
    d = index_dir(base_dir) / doc_id
    d.mkdir(parents=True, exist_ok=True)
    index_path = d / "index.faiss"
    chunks_path = d / "chunks.json"

    if index_path.exists() and chunks_path.exists():
        index = faiss.read_index(str(index_path))
        chunks_raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [RagChunk(**c) for c in chunks_raw]
        return doc_id, index, chunks

    pages = chunk_text_by_pages(pdf_bytes, max_pages=max_pages)
    if not pages:
        # fallback to raw extraction if page extraction fails
        raw = extract_text_from_pdf_bytes(pdf_bytes, max_pages=max_pages)
        pages = [(1, raw)]

    total_chars = sum(len(txt) for _, txt in pages)
    if total_chars < 5000:
        chunk_chars, overlap_chars = 600, 100
    elif total_chars < 20000:
        chunk_chars, overlap_chars = 1000, 150
    else:
        chunk_chars, overlap_chars = 1500, 200

    chunks = chunk_pages(pages, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        raise RuntimeError("Could not create text chunks from PDF.")

    vecs = embed_texts([c.text for c in chunks])
    vecs = np.asarray(vecs, dtype=np.float32)
    vecs = _normalize(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
    index.add(vecs)

    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")

    return doc_id, index, chunks


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[RagChunk],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if top_k is None:
        num_chunks = len(chunks)
        if num_chunks < 10:
            top_k = 3
        elif num_chunks < 30:
            top_k = 4
        else:
            top_k = 5

    qv = embed_texts([query])
    qv = _normalize(np.asarray(qv, dtype=np.float32))
    scores, idxs = index.search(qv, top_k)
    out: list[dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        out.append(
            {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "score": float(score),
                "text": c.text,
            }
        )
    return out

