"""Reranker — 1~3주차에는 stub, 4-5주차에 bge-reranker 같은 실제 모델로 교체.

인터페이스만 고정해두면 업그레이드 시 호출부 변경 없음.
"""

from __future__ import annotations

import os
from typing import Protocol

from .schemas import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...


class IdentityReranker:
    """원본 순서 그대로 반환 (stub)."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return list(results)


class CrossEncoderReranker:
    """bge-reranker / ms-marco-MiniLM 같은 cross-encoder 기반 재랭킹.

    GPU 가능한 환경(4-5주차 PM 서버)에서 활성화.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs)
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        out = []
        for r, s in scored:
            out.append(SearchResult(chunk=r.chunk, score=float(s)))
        return out


def get_reranker() -> Reranker:
    kind = os.environ.get("RAG_RERANKER", "identity").lower()
    if kind == "cross":
        model = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        device = os.environ.get("RAG_RERANKER_DEVICE", "cpu")
        return CrossEncoderReranker(model_name=model, device=device)
    return IdentityReranker()
