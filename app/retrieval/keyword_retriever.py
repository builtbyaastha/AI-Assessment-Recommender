"""TF-IDF fallback for when the embedding model isn't available (offline
box, no HF Hub access, whatever). Same interface as the FAISS index so
retriever.py can swap between them without callers noticing. Recall is
worse than dense search but it never crashes for lack of a model, and
it's what the test suite uses so pytest doesn't need network access.
"""
from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KeywordIndex:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.ids: List[str] = []

    def build(self, ids: List[str], texts: List[str]) -> None:
        self.ids = list(ids)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in ranked if sims[i] > 0]
