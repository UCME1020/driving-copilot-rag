"""PDF 파싱 + Semantic Chunking.

차량 매뉴얼은 테이블 / 경고등 / 다단 레이아웃이 많으므로
PyMuPDF로 텍스트+구조를 추출하고, 문단 기준 Semantic Chunking을 적용한다.
청크 경계: 문단. 평균 512 토큰 이하 유지 (대략 문자 1200자 내외).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .schemas import Chunk, ContentType


MAX_CHARS_PER_CHUNK = 1200
MIN_CHARS_PER_CHUNK = 80

_WARNING_KEYWORDS = ("경고", "주의", "위험", "⚠", "warning", "caution", "danger")
_WARNING_LIGHT_KEYWORDS = ("경고등", "표시등", "indicator", "warning light")
_SPEC_KEYWORDS = ("사양", "제원", "규격", "specification", "spec")


def _infer_content_type(text: str) -> ContentType:
    low = text.lower()
    if any(k in low for k in _WARNING_LIGHT_KEYWORDS):
        return "warning_light"
    if any(k in low for k in _WARNING_KEYWORDS):
        return "warning"
    if any(k in low for k in _SPEC_KEYWORDS):
        return "spec"
    return "general"


def _extract_section_heading(text: str) -> str:
    """Best-effort: first non-empty line that looks like a heading."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Heuristic: numbered heading like "3.2 공조 시스템" or short ALL-CAPS.
        if re.match(r"^\d+(\.\d+)*\s+\S", line) or (line.endswith(":") and len(line) < 40):
            return line.rstrip(":").strip()
        return line[:60]
    return ""


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _pack_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    """Greedy pack paragraphs into chunks below MAX_CHARS_PER_CHUNK."""
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        if buf_len + len(p) + 1 > MAX_CHARS_PER_CHUNK and buf:
            out.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p) + 1
    if buf:
        out.append("\n\n".join(buf))
    return [c for c in out if len(c) >= MIN_CHARS_PER_CHUNK]


def parse_pdf(path: str | Path) -> list[Chunk]:
    """Parse a single PDF into chunks."""
    import fitz  # PyMuPDF

    path = Path(path)
    doc = fitz.open(path)
    chunks: list[Chunk] = []
    try:
        for page_idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            if not text.strip():
                continue
            section = _extract_section_heading(text)
            packed = _pack_paragraphs(_paragraphs(text))
            for piece in packed:
                chunk_id = hashlib.sha1(
                    f"{path.name}:{page_idx}:{piece[:64]}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=piece,
                        source=path.name,
                        page_num=page_idx + 1,
                        section=section,
                        content_type=_infer_content_type(piece),
                    )
                )
    finally:
        doc.close()
    return chunks


def parse_directory(pdf_dir: str | Path) -> list[Chunk]:
    pdf_dir = Path(pdf_dir)
    all_chunks: list[Chunk] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        all_chunks.extend(parse_pdf(pdf))
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")


def load_chunks(in_path: str | Path) -> list[Chunk]:
    in_path = Path(in_path)
    chunks: list[Chunk] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate_json(line))
    return chunks


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse car manual PDFs into chunks.")
    ap.add_argument("pdf_dir", help="Directory containing PDF files")
    ap.add_argument("--out", default="rag/data/chunks.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    chunks = parse_directory(args.pdf_dir)
    save_chunks(chunks, args.out)
    print(f"Parsed {len(chunks)} chunks → {args.out}")
