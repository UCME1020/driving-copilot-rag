"""RAG 파이프라인.

책임:
  - PDF → chunk → vectordb 인덱싱 (build_index)
  - query → 검색 → (선택) rerank (search)
  - context 조립 (make_context)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .embeddings import EncoderProtocol, get_encoder
from .pdf_parser import load_chunks, parse_directory, save_chunks
from .reranker import Reranker, get_reranker
from .schemas import Chunk, SearchResult
from .vectordb import QdrantStore


class RAGPipeline:
    def __init__(
        self,
        encoder: EncoderProtocol | None = None,
        store: QdrantStore | None = None,
        reranker: Reranker | None = None,
    ):
        self.encoder = encoder or get_encoder()
        self.store = store or QdrantStore(self.encoder)
        self.reranker = reranker or get_reranker()

    # ---------- Index ----------
    def build_index_from_pdfs(
        self, pdf_dir: str | Path, chunks_out: str | Path = "rag/data/chunks.jsonl"
    ) -> int:
        chunks = parse_directory(pdf_dir)
        save_chunks(chunks, chunks_out)
        return self.store.upsert(chunks)

    def build_index_from_jsonl(self, jsonl_path: str | Path) -> int:
        chunks = load_chunks(jsonl_path)
        return self.store.upsert(chunks)

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        return self.store.upsert(chunks)

    # ---------- Query ----------
    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> list[SearchResult]:
        fetch_k = max(top_k * 4, 20) if rerank else top_k
        results = self.store.search(query, top_k=fetch_k)
        if rerank:
            results = self.reranker.rerank(query, results)
        return results[:top_k]

    def make_context(self, results: list[SearchResult], max_chars: int = 2400) -> str:
        """Concatenate top chunks into an LLM context string."""
        out: list[str] = []
        total = 0
        for r in results:
            block = (
                f"[출처: {r.chunk.source} p.{r.chunk.page_num}"
                f"{' / ' + r.chunk.section if r.chunk.section else ''}]"
                f"\n{r.chunk.content}"
            )
            if total + len(block) > max_chars and out:
                break
            out.append(block)
            total += len(block)
        return "\n\n---\n\n".join(out)


_default_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline
