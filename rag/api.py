"""FastAPI 진입점.

엔드포인트 (guide의 통합 API Contract 준수):
  POST /rag/search  - 입력 query, 출력 chunks[] + scores[]
  POST /rag/route   - 입력 query + has_image, 출력 route_type
  GET  /health      - 헬스체크
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .query_router import get_router
from .schemas import (
    RouteRequest,
    RouteResponse,
    SearchRequest,
    SearchResponse,
)


STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm up
    get_router()
    # encoder는 사용 시점에 로드 (첫 요청 지연 허용)
    yield


app = FastAPI(title="Driving Copilot — RAG", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def playground() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/rag/search", response_model=SearchResponse)
def rag_search(req: SearchRequest) -> SearchResponse:
    from .pipeline import get_pipeline  # lazy — heavy deps (qdrant, sentence-transformers)

    pipe = get_pipeline()
    results = pipe.search(req.query, top_k=req.top_k, rerank=req.rerank)
    return SearchResponse(
        chunks=[r.chunk for r in results],
        scores=[r.score for r in results],
    )


@app.post("/rag/route", response_model=RouteResponse)
def rag_route(req: RouteRequest) -> RouteResponse:
    return get_router().route(req.query, has_image=req.has_image)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.api:app", host="0.0.0.0", port=8001, reload=False)
