from rag.schemas import (
    Chunk,
    RouteRequest,
    RouteResponse,
    RouteType,
    SearchRequest,
    SearchResponse,
)


def test_chunk_defaults():
    c = Chunk(chunk_id="abc", content="x", source="m.pdf", page_num=1)
    assert c.section == ""
    assert c.content_type == "general"


def test_search_request_defaults():
    req = SearchRequest(query="test")
    assert req.top_k == 5
    assert req.rerank is True


def test_route_request():
    req = RouteRequest(query="에어컨 켜줘")
    assert req.has_image is False


def test_route_response_validation():
    resp = RouteResponse(route_type=RouteType.TOOL, confidence=0.9)
    assert resp.route_type == RouteType.TOOL
    assert 0 <= resp.confidence <= 1
