"""Qdrant 벡터 스토어 래퍼.

기본: 로컬 파일 스토리지 (`./vectordb_storage`) — 도커 없이 바로 사용.
환경변수 `QDRANT_URL`이 있으면 원격 서버에 붙는다.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..embeddings import EncoderProtocol
from ..schemas import Chunk, SearchResult


DEFAULT_COLLECTION = "car_manuals"
DEFAULT_STORAGE_PATH = "./vectordb_storage"


class QdrantStore:
    def __init__(
        self,
        encoder: EncoderProtocol,
        collection: str = DEFAULT_COLLECTION,
        url: str | None = None,
        storage_path: str | None = None,
    ):
        self.encoder = encoder
        self.collection = collection

        url = url or os.environ.get("QDRANT_URL")
        if url:
            self.client = QdrantClient(url=url)
        else:
            path = storage_path or os.environ.get("QDRANT_PATH", DEFAULT_STORAGE_PATH)
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.encoder.dim, distance=qm.Distance.COSINE),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        vectors = self.encoder.encode(texts, normalize=True)
        points = [
            qm.PointStruct(
                id=self._point_id(c.chunk_id),
                vector=vec.tolist(),
                payload=c.model_dump(),
            )
            for c, vec in zip(chunks, vectors)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[i : i + batch_size])
        return len(points)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vec = self.encoder.encode(query, normalize=True)[0]
        # qdrant-client >=1.10: search() deprecated → query_points().
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = resp.points
        out: list[SearchResult] = []
        for h in hits:
            payload = dict(h.payload or {})
            out.append(SearchResult(chunk=Chunk(**payload), score=float(h.score)))
        return out

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()
