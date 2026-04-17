"""Router unit tests — sklearn/sentence-transformers 없이도 실행 가능.

라우터는 CPU / 의존성만 필요해 설치 직후 바로 검증한다.
"""

from __future__ import annotations

from rag.query_router import QueryRouter
from rag.schemas import RouteType


def test_rule_based_tool():
    r = QueryRouter()
    resp = r.route("에어컨 켜줘")
    assert resp.route_type == RouteType.TOOL
    assert resp.matched_rule is not None


def test_rule_based_rag():
    r = QueryRouter()
    resp = r.route("엔진 경고등이 무슨 뜻이야")
    assert resp.route_type == RouteType.RAG


def test_has_image_defaults_vision():
    r = QueryRouter()
    resp = r.route("이거 뭐야", has_image=True)
    assert resp.route_type == RouteType.VISION


def test_has_image_tool_override():
    r = QueryRouter()
    resp = r.route("에어컨 켜줘", has_image=True)
    assert resp.route_type == RouteType.TOOL


def test_empty_query():
    r = QueryRouter()
    resp = r.route("")
    assert resp.route_type == RouteType.CHAT
    assert resp.matched_rule == "empty"


def test_ml_fallback_chat():
    r = QueryRouter()
    resp = r.route("잘 자")
    assert resp.route_type == RouteType.CHAT
