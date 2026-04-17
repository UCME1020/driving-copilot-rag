"""RAG 평가 프레임워크.

측정 메트릭:
  Retrieval: Recall@k, MRR
  Answer Quality: 정답 키워드 포함 여부 (간이 버전), + 외부 GPT-4 Judge 훅
  Router: 4방향 분류 정확도

테스트셋 형식 (eval_data/qa_testset.jsonl):
  {"query": "...", "expected_answer": "...", "answer_keywords": ["...", "..."],
   "relevant_chunk_ids": ["abc123", ...]}

라우터 테스트셋 (eval_data/router_testset.jsonl):
  {"query": "...", "route": "rag" | "tool" | "vision" | "chat",
   "has_image": false}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .query_router import QueryRouter, get_router
from .schemas import RouteType

# NOTE: RAGPipeline은 retrieval 평가에서만 필요 → 지연 import (heavy deps)


# ---------- Retrieval metrics ----------
def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    topk = retrieved_ids[:k]
    return len(relevant.intersection(topk)) / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


# ---------- Answer quality (lightweight) ----------
def keyword_coverage(context: str, keywords: Iterable[str]) -> float:
    keywords = [k for k in keywords if k]
    if not keywords:
        return 0.0
    ctx = context.lower()
    hit = sum(1 for k in keywords if k.lower() in ctx)
    return hit / len(keywords)


# ---------- Reports ----------
@dataclass
class RetrievalReport:
    n: int = 0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    keyword_coverage_mean: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RouterReport:
    n: int = 0
    accuracy: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------- Evaluators ----------
def evaluate_retrieval(
    testset_path: str | Path,
    pipeline: "RAGPipeline | None" = None,
    k: int = 5,
) -> RetrievalReport:
    from .pipeline import RAGPipeline, get_pipeline  # lazy — heavy deps

    pipeline = pipeline or get_pipeline()
    rows = _read_jsonl(testset_path)
    if not rows:
        return RetrievalReport()

    recall_sum = 0.0
    mrr_sum = 0.0
    kw_sum = 0.0
    for row in rows:
        query = row["query"]
        relevant_ids = row.get("relevant_chunk_ids", [])
        keywords = row.get("answer_keywords", [])
        results = pipeline.search(query, top_k=k, rerank=True)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        recall_sum += recall_at_k(retrieved_ids, relevant_ids, k)
        mrr_sum += reciprocal_rank(retrieved_ids, relevant_ids)
        if keywords:
            context = pipeline.make_context(results)
            kw_sum += keyword_coverage(context, keywords)

    n = len(rows)
    return RetrievalReport(
        n=n,
        recall_at_5=recall_sum / n,
        mrr=mrr_sum / n,
        keyword_coverage_mean=kw_sum / n,
    )


def evaluate_router(
    testset_path: str | Path,
    router: QueryRouter | None = None,
) -> RouterReport:
    router = router or get_router()
    rows = _read_jsonl(testset_path)
    if not rows:
        return RouterReport()

    labels = [RouteType.RAG.value, RouteType.TOOL.value, RouteType.VISION.value, RouteType.CHAT.value]
    confusion = {a: {b: 0 for b in labels} for a in labels}
    correct = 0
    for row in rows:
        expected = row["route"]
        pred = router.route(row["query"], has_image=row.get("has_image", False)).route_type.value
        confusion[expected][pred] += 1
        if expected == pred:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in labels:
        tp = confusion[cls][cls]
        fn = sum(confusion[cls][o] for o in labels if o != cls)
        fp = sum(confusion[o][cls] for o in labels if o != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    n = len(rows)
    return RouterReport(
        n=n,
        accuracy=correct / n,
        per_class=per_class,
        confusion=confusion,
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 평가 실행")
    ap.add_argument("--qa", default="eval_data/qa_testset.jsonl", help="Retrieval/QA 테스트셋 경로")
    ap.add_argument("--router", default="eval_data/router_testset.jsonl", help="라우터 테스트셋 경로")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="eval_data/report.json")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()

    report: dict = {}
    if not args.skip_retrieval:
        retrieval = evaluate_retrieval(args.qa, k=args.k)
        report["retrieval"] = retrieval.as_dict()
        print("Retrieval:", json.dumps(retrieval.as_dict(), ensure_ascii=False, indent=2))

    router_report = evaluate_router(args.router)
    report["router"] = router_report.as_dict()
    print("Router:", json.dumps(router_report.as_dict(), ensure_ascii=False, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
