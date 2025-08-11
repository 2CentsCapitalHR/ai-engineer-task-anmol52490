# chunker.py
# DOCX → Unstructured → CHUNKS (by_title with sane params)
import io
import hashlib
from typing import List, Dict, Any

from unstructured.partition.docx import partition_docx
from unstructured.chunking.title import chunk_by_title


def _hash_id(*parts: str, length: int = 12) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:length]


def _preview(text: str, limit: int = 240) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[:limit] + "…"


def chunk_docx_bytes(
    file_bytes: bytes,
    filename: str,
    max_characters: int = 1400,
    new_after_n_chars: int = 1000,
    combine_text_under_n_chars: int = 250,
) -> List[Dict[str, Any]]:
    # 1) Partition raw elements
    elements = partition_docx(
        file=io.BytesIO(file_bytes),
        metadata_filename=filename,
        include_page_breaks=True,
    )
    # 2) Chunk by title with the agreed knobs
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
        multipage_sections=True,
    )

    out: List[Dict[str, Any]] = []
    for i, el in enumerate(chunks):
        meta = getattr(el, "metadata", None)
        text = getattr(el, "text", "") or ""
        el_type = getattr(el, "category", el.__class__.__name__)
        out.append(
            {
                "id": _hash_id(filename, str(i), el_type, text[:64]),
                "type": el_type,  # CompositeElement / Table / Title / TableChunk
                "text": text,
                "preview": _preview(text),
                "char_count": len(text),
                "metadata": {
                    "filename": filename,
                    "page_start": getattr(meta, "page_number", None),
                    "page_end": getattr(meta, "last_page_number", None),
                    "section_title": getattr(meta, "section_title", None),
                    "section_path": getattr(meta, "section", None),
                },
            }
        )
    return out
