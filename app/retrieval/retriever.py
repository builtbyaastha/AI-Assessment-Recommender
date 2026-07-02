"""Module 3 - Retriever.

Takes a HiringContext, returns the top ~20 candidate assessments. Dense
embeddings (bge-small + FAISS) is the main path; if that model can't be
loaded for whatever reason we drop back to TF-IDF automatically. Callers
never see the difference, they just get CatalogItems back.

We pull 20 candidates instead of going straight to 5 because the reranker
(module 4) uses structured fields the embedder can't see - test type
preference, duration caps - so it's worth giving it some room to work with
rather than trusting raw embedding rank alone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from app.catalog.catalog_loader import Catalog
from app.models.schemas import CatalogItem, HiringContext
from app.retrieval.keyword_retriever import KeywordIndex

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parent.parent / "catalog" / "index"


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.keyword_index = KeywordIndex()
        self.keyword_index.build(
            ids=[item.id for item in catalog.items],
            texts=[self._doc_text(item) for item in catalog.items],
        )

        self.dense_index = None
        self._try_load_dense_index()

    @staticmethod
    def _doc_text(item: CatalogItem) -> str:
        return f"{item.name}. {item.description}"

    def _try_load_dense_index(self) -> None:
        try:
            from app.retrieval.faiss_index import CatalogIndex

            idx = CatalogIndex()
            if (INDEX_DIR / "catalog.index").exists():
                idx.load(str(INDEX_DIR))
            else:
                from app.retrieval.embedder import embed_documents

                vectors = embed_documents([self._doc_text(item) for item in self.catalog.items])
                idx.build(ids=[item.id for item in self.catalog.items], vectors=vectors)
                idx.save(str(INDEX_DIR))
            self.dense_index = idx
            logger.info("dense (FAISS) index ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("dense retrieval unavailable (%s), using TF-IDF fallback", exc)
            self.dense_index = None

    def search(self, context: HiringContext, top_k: int = 20) -> List[CatalogItem]:
        query = context.to_query_text()
        if not query.strip():
            return []

        if self.dense_index is not None:
            try:
                from app.retrieval.embedder import embed_query

                query_vec = embed_query(query)
                hits = self.dense_index.search(query_vec, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dense search failed (%s), falling back to keyword search", exc)
                hits = self.keyword_index.search(query, top_k)
        else:
            hits = self.keyword_index.search(query, top_k)

        results = []
        for item_id, _score in hits:
            item = self.catalog.get(item_id)
            if item is not None:
                results.append(item)
        return results
