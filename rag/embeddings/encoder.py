"""Embedding encoder.

두 가지 백엔드:
  - `transformer`: sentence-transformers (MiniLM → bge-m3 교체 가능). 운영용.
  - `hash`: sklearn HashingVectorizer 기반 결정적 벡터. torch 없이 돌아감.
           Mock-first 개발용 및 ML 의존성 불가 환경에서 파이프라인 검증용.

환경변수:
  RAG_ENCODER_BACKEND  = transformer | hash   (기본 transformer)
  RAG_EMBED_MODEL      = transformer 백엔드 모델 이름
  RAG_EMBED_DIM        = hash 백엔드 차원 (기본 384, MiniLM과 맞춤)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Protocol

import numpy as np


DEFAULT_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_BACKEND = os.environ.get("RAG_ENCODER_BACKEND", "transformer").lower()
DEFAULT_HASH_DIM = int(os.environ.get("RAG_EMBED_DIM", "384"))


class EncoderProtocol(Protocol):
    model_name: str
    @property
    def dim(self) -> int: ...
    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray: ...


class TransformerEncoder:
    """Sentence-transformers wrapper. torch/onnx 필요."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


class HashEncoder:
    """sklearn HashingVectorizer 기반 결정적 dense 인코더 (torch 불필요).

    문자 n-gram을 해싱해 고정 차원 벡터로 만든 뒤 L2 정규화.
    품질은 sentence-transformers보다 훨씬 낮지만, 파이프라인 배선이
    제대로 됐는지 확인하는 데에는 충분하다.
    """

    def __init__(self, dim: int = DEFAULT_HASH_DIM):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.model_name = f"hash-char{dim}"
        self._dim = dim
        self._vec = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=dim,
            alternate_sign=True,
            norm=None,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        X = self._vec.transform(texts).toarray().astype(np.float32)
        if normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return X


# Backward-compatible alias.
Encoder = TransformerEncoder


def _make_encoder(backend: str, model_name: str, device: str) -> EncoderProtocol:
    if backend == "hash":
        return HashEncoder(dim=DEFAULT_HASH_DIM)
    return TransformerEncoder(model_name=model_name, device=device)


@lru_cache(maxsize=4)
def get_encoder(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    backend: str = DEFAULT_BACKEND,
) -> EncoderProtocol:
    """Cached encoder (모델을 매 요청 다시 로드하지 않도록)."""
    return _make_encoder(backend=backend, model_name=model_name, device=device)
