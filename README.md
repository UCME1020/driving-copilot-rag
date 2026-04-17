# Driving Copilot — PART A (RAG Engineer)

On-Device Multimodal Driving Copilot 프로젝트의 RAG 모듈.
GPU 없이 CPU 환경에서 독립 개발 가능하도록 구성

## 구조

```
rag/
├── data/              # 매뉴얼 PDF 원본 + 처리된 chunks (.jsonl)
├── embeddings/        # 임베딩 모델 래퍼 (MiniLM → bge-m3 swappable)
├── vectordb/          # Qdrant 래퍼 (로컬 파일 스토리지)
├── tests/             # pytest 단위 테스트
├── __init__.py
├── schemas.py         # pydantic 스키마 (API Contract)
├── pdf_parser.py      # PyMuPDF 파싱 + Semantic Chunking
├── query_router.py    # 4방향 라우터 (규칙 1차 + TF-IDF LogReg 2차)
├── reranker.py        # stub / CrossEncoder 교체형
├── pipeline.py        # RAG 파이프라인 (index / search / context)
├── evaluate.py        # Recall@k, MRR, Router accuracy
└── api.py             # FastAPI (POST /rag/search, /rag/route)
eval_data/
├── qa_testset.jsonl
└── router_testset.jsonl
requirements.txt
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 사용

### 1) 매뉴얼 PDF 인덱싱

```bash
# PDF들을 rag/data/pdfs/ 아래에 둔 뒤:
python -m rag.pdf_parser rag/data/pdfs --out rag/data/chunks.jsonl

# 인덱스 빌드 (첫 실행 시 임베딩 모델 다운로드)
python -c "from rag.pipeline import get_pipeline; print(get_pipeline().build_index_from_jsonl('rag/data/chunks.jsonl'))"
```

### 2) API 서버 실행

```bash
uvicorn rag.api:app --host 0.0.0.0 --port 8001
```

엔드포인트:
- `POST /rag/search` → `{ "chunks": [...], "scores": [...] }`
- `POST /rag/route`  → `{ "route_type": "rag" | "tool" | "vision" | "chat", "confidence": 0.0~1.0 }`
- `GET  /health`

### 3) 테스트 & 평가

```bash
pytest rag/tests -q
python -m rag.evaluate --qa eval_data/qa_testset.jsonl --router eval_data/router_testset.jsonl
```

## 주차별 산출물 매핑

| 주차 | 산출물 | 구현 위치 |
|------|--------|-----------|
| 1주 | PDF 파싱 + chunks.jsonl | `pdf_parser.py` |
| 2주 | Qdrant 인덱싱 + 검색 API | `vectordb/`, `pipeline.py`, `api.py` |
| 3주 | Query Router + Reranker stub | `query_router.py`, `reranker.py` |
| 4-5주 | 평가 + bge-m3 교체 + Reranker | `evaluate.py` + env `RAG_EMBED_MODEL`, `RAG_RERANKER` |

## 모델 교체 (4-5주차)

임베딩:
```bash
export RAG_EMBED_MODEL="BAAI/bge-m3"        # 1024-dim
```
리랭커:
```bash
export RAG_RERANKER=cross
export RAG_RERANKER_MODEL="BAAI/bge-reranker-base"
export RAG_RERANKER_DEVICE=cuda             # PM GPU 서버
```
> 임베딩 차원이 바뀌면 기존 Qdrant 컬렉션을 `QdrantStore.reset()`으로 재생성 후 재인덱싱

## 통합 API Contract (guide 준수)

| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /rag/search` | `{ "query", "top_k?", "rerank?" }` | `{ "chunks": [Chunk], "scores": [float] }` |
| `POST /rag/route`  | `{ "query", "has_image" }` | `{ "route_type", "confidence", "matched_rule?" }` |

Chunk 필드: `chunk_id, content, source, page_num, section, content_type`
