"""Data schemas shared across RAG module.

Contracts here match the guide's 통합 API (see docs/api-contract.md).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


ContentType = Literal["general", "warning", "spec", "warning_light"]


class Chunk(BaseModel):
    """A manual chunk stored in the vector DB."""

    chunk_id: str
    content: str
    source: str
    page_num: int
    section: str = ""
    content_type: ContentType = "general"


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True


class SearchResponse(BaseModel):
    chunks: list[Chunk]
    scores: list[float]


class RouteType(str, Enum):
    RAG = "rag"
    TOOL = "tool"
    VISION = "vision"
    CHAT = "chat"


class RouteRequest(BaseModel):
    query: str
    has_image: bool = False


class RouteResponse(BaseModel):
    route_type: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    matched_rule: Optional[str] = None
