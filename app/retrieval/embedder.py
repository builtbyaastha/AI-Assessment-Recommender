"""Wraps sentence-transformers so the rest of the app doesn't care what
model is loaded underneath. Using BAAI/bge-small-en-v1.5 - small enough to
run fast on CPU which matters since we're inside a 30s request window.

BGE wants a query prefix for asymmetric search (short query vs longer doc),
so that only gets added on the query side, never for catalog docs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: List[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([QUERY_INSTRUCTION + text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype="float32")[0]
