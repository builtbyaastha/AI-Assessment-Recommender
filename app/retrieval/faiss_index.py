"""FAISS index over catalog embeddings. Vectors are already L2-normalized
(see embedder.py) so inner product = cosine similarity, hence IndexFlatIP.
Catalog is only a few hundred rows so exact search is plenty fast - no
need for IVF/HNSW here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class CatalogIndex:
    def __init__(self):
        self.index = None
        self.ids: List[str] = []

    def build(self, ids: List[str], vectors: np.ndarray) -> None:
        import faiss

        self.ids = list(ids)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def save(self, dir_path: str) -> None:
        import faiss

        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "catalog.index"))
        (p / "ids.json").write_text(json.dumps(self.ids))

    def load(self, dir_path: str) -> None:
        import faiss

        p = Path(dir_path)
        self.index = faiss.read_index(str(p / "catalog.index"))
        self.ids = json.loads((p / "ids.json").read_text())

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        scores, idxs = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.ids[idx], float(score)))
        return results
