"""Query Router — 사용자 입력을 4가지 경로로 분류.

route_type: rag | tool | vision | chat

2단계 라우팅:
  1차) 키워드 규칙 기반 — 명확한 매칭이 있으면 즉시 리턴 (신뢰도 high).
  2차) TF-IDF + LogisticRegression — 애매한 경우 학습된 분류기 사용.

이미지 동반 입력은 기본적으로 vision으로 우선 분기하되,
텍스트에 도구 실행 키워드가 강하면 tool로 오버라이드한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schemas import RouteResponse, RouteType


# ---------- 1차 규칙 ----------
#
# 각 규칙은 (pattern, route, label) 형태. 대소문자 무시.
# 보다 구체적인 규칙을 앞쪽에 둔다.
_TOOL_VERBS = [
    "켜줘", "켜", "꺼줘", "꺼", "틀어줘", "틀어",
    "올려줘", "올려", "내려줘", "내려", "높여줘", "낮춰줘",
    "바꿔줘", "변경", "설정해", "맞춰줘", "맞춰",
    "열어줘", "닫아줘", "재생", "정지", "음량", "볼륨",
]
_VISION_CUES = [
    "보여", "보이", "앞에", "뭐가", "상황", "카메라",
    "지금 밖", "화면", "영상", "보이는",
]
_RAG_CUES = [
    "경고등", "매뉴얼", "의미", "뜻", "무엇", "교환 주기", "주기",
    "어떻게", "방법", "스펙", "사양", "원리", "왜",
]

_RULES: list[tuple[re.Pattern[str], RouteType, str]] = []

for kw in _TOOL_VERBS:
    _RULES.append((re.compile(re.escape(kw)), RouteType.TOOL, f"tool:{kw}"))

for kw in _VISION_CUES:
    _RULES.append((re.compile(re.escape(kw)), RouteType.VISION, f"vision:{kw}"))

for kw in _RAG_CUES:
    _RULES.append((re.compile(re.escape(kw)), RouteType.RAG, f"rag:{kw}"))


# ---------- 2차 학습 데이터 (seed) ----------
#
# 실제 운영 시에는 eval_data/router_train.jsonl을 확장한다.
_SEED_TRAINING: list[tuple[str, RouteType]] = [
    # rag
    ("엔진 경고등이 무슨 뜻이야", RouteType.RAG),
    ("오일 교환 주기 알려줘", RouteType.RAG),
    ("타이어 공기압 기준이 뭐야", RouteType.RAG),
    ("배터리 방전되면 어떻게 해야 해", RouteType.RAG),
    ("ABS 경고등 점등 원인", RouteType.RAG),
    ("냉각수 부족 시 대처법", RouteType.RAG),
    ("에어백 작동 조건이 궁금해", RouteType.RAG),
    ("브레이크 패드 교체 주기", RouteType.RAG),
    # tool
    ("온도 24도로 맞춰줘", RouteType.TOOL),
    ("에어컨 켜줘", RouteType.TOOL),
    ("노래 틀어줘", RouteType.TOOL),
    ("창문 열어줘", RouteType.TOOL),
    ("뒷좌석 열선 켜줘", RouteType.TOOL),
    ("볼륨 좀 낮춰줘", RouteType.TOOL),
    ("주행 모드 스포츠로 변경", RouteType.TOOL),
    ("와이퍼 작동시켜줘", RouteType.TOOL),
    # vision
    ("앞에 뭐가 보여", RouteType.VISION),
    ("지금 상황 설명해줘", RouteType.VISION),
    ("카메라 보이는거 알려줘", RouteType.VISION),
    ("밖에 날씨 어때", RouteType.VISION),
    ("신호등 색깔 뭐야", RouteType.VISION),
    ("전방 장애물 있어", RouteType.VISION),
    # chat
    ("안녕", RouteType.CHAT),
    ("고마워", RouteType.CHAT),
    ("오늘 기분 어때", RouteType.CHAT),
    ("심심하다", RouteType.CHAT),
    ("너 이름이 뭐야", RouteType.CHAT),
    ("잘 자", RouteType.CHAT),
]


@dataclass
class RuleHit:
    route: RouteType
    rule: str


def _rule_match(text: str) -> RuleHit | None:
    for pat, route, label in _RULES:
        if pat.search(text):
            return RuleHit(route=route, rule=label)
    return None


class QueryRouter:
    """2단계 라우터. ML 분류기는 lazy init."""

    def __init__(self, extra_training: Iterable[tuple[str, RouteType]] | None = None):
        self._training: list[tuple[str, RouteType]] = list(_SEED_TRAINING)
        if extra_training:
            self._training.extend(extra_training)
        self._vectorizer = None
        self._clf = None
        self._fit()

    def _fit(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = [t for t, _ in self._training]
        labels = [r.value for _, r in self._training]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1
        )
        X = self._vectorizer.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(X, labels)

    def route(self, query: str, has_image: bool = False) -> RouteResponse:
        text = (query or "").strip()
        if not text:
            return RouteResponse(route_type=RouteType.CHAT, confidence=0.0, matched_rule="empty")

        # 이미지가 있으면 기본은 vision. 단, 명시적 도구 키워드면 tool로 override.
        if has_image:
            hit = _rule_match(text)
            if hit and hit.route == RouteType.TOOL:
                return RouteResponse(route_type=RouteType.TOOL, confidence=0.9, matched_rule=hit.rule)
            return RouteResponse(route_type=RouteType.VISION, confidence=0.85, matched_rule="has_image")

        hit = _rule_match(text)
        if hit:
            return RouteResponse(route_type=hit.route, confidence=0.85, matched_rule=hit.rule)

        # ML 분기
        assert self._vectorizer is not None and self._clf is not None
        X = self._vectorizer.transform([text])
        probs = self._clf.predict_proba(X)[0]
        classes = list(self._clf.classes_)
        best_idx = int(probs.argmax())
        best_label = classes[best_idx]
        return RouteResponse(
            route_type=RouteType(best_label),
            confidence=float(probs[best_idx]),
            matched_rule=None,
        )


_default_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    global _default_router
    if _default_router is None:
        _default_router = QueryRouter()
    return _default_router


def load_training_jsonl(path: str | Path) -> list[tuple[str, RouteType]]:
    import json

    path = Path(path)
    out: list[tuple[str, RouteType]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append((obj["query"], RouteType(obj["route"])))
    return out
