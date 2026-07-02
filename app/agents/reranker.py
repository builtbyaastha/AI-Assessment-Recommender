"""Module 4 - Reranker.

Takes the ~20 candidates from the retriever and narrows to a best 5-10,
using structured fields the embedder never saw (test-type preference,
duration cap) plus straightforward keyword overlap. This is where a
"refine" like "add personality tests" actually shows up in results:
assessment_type_preference wasn't necessarily in the original embedding
query, but it directly boosts P-type items here no matter how retrieval
ranked them.

No LLM call here on purpose - reranking a small, already-relevant set with
known structured fields is a scoring problem, not a reasoning problem, and
keeping it deterministic makes it reproducible.
"""
from __future__ import annotations

from typing import List

from app.models.schemas import CatalogItem, HiringContext

MIN_RESULTS = 1
MAX_RESULTS = 10
DEFAULT_SHORTLIST_SIZE = 5


def _score(item: CatalogItem, context: HiringContext) -> float:
    score = 0.0
    text = f"{item.name} {item.description}".lower()

    for skill in context.technical_skills:
        if skill.lower() in text:
            score += 3.0
    for skill in context.soft_skills:
        if skill.lower() in text:
            score += 2.0
    if context.role and any(word.lower() in text for word in context.role.split()):
        score += 1.5

    item_types = {t.value for t in item.test_type}
    preferred_types = {t.value for t in context.assessment_type_preference}
    if preferred_types:
        score += 4.0 * len(item_types & preferred_types)

    if context.personality_required and "P" in item_types:
        score += 3.0
    if context.cognitive_required and "A" in item_types:
        score += 3.0

    if context.max_duration_minutes and item.duration_minutes:
        if item.duration_minutes <= context.max_duration_minutes:
            score += 1.0
        else:
            score -= 2.0

    return score


def rerank(
    candidates: List[CatalogItem],
    context: HiringContext,
    max_results: int = DEFAULT_SHORTLIST_SIZE,
) -> List[CatalogItem]:
    if not candidates:
        return []

    scored = [(item, _score(item, context)) for item in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    k = max(MIN_RESULTS, min(max_results, MAX_RESULTS))
    return [item for item, _score_val in scored[:k]]
