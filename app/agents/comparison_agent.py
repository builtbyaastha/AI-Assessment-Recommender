"""Handles "compare" turns - e.g. "what's the difference between OPQ and GSA?"

Strictly grounded in catalog data: we look each named assessment up by
name, and if we can't find it we say so instead of answering from the
model's general knowledge about SHL products (which might be wrong or
out of date). This is probably the single spot in the pipeline most at
risk of hallucination if handled sloppily, so it never touches the LLM's
own knowledge at all.
"""
from __future__ import annotations

from typing import List

from app.catalog.catalog_loader import Catalog
from app.models.schemas import CatalogItem, ChatResponse, HiringContext, Recommendation

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement",
    "C": "Competencies", "D": "Development & 360", "E": "Assessment Exercises",
    "K": "Knowledge & Skills", "P": "Personality & Behavior", "S": "Simulations",
}


def _describe(item: CatalogItem) -> str:
    types = ", ".join(TEST_TYPE_LABELS.get(t.value, t.value) for t in item.test_type)
    return f"**{item.name}** ({types}): {item.description}"


def compare(catalog: Catalog, context: HiringContext) -> ChatResponse:
    found: List[CatalogItem] = []
    not_found: List[str] = []

    for name in context.comparison_targets:
        item = catalog.find_by_name(name)
        if item:
            found.append(item)
        else:
            not_found.append(name)

    if len(found) < 2:
        if not_found:
            reply = (
                f"I couldn't find {', '.join(not_found)} in the SHL catalog I have indexed, "
                f"so I can't do a grounded comparison. Could you check the name, or ask about "
                f"a different pair of assessments?"
            )
        else:
            reply = "Which two assessments would you like me to compare?"
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    lines = [_describe(item) for item in found]
    reply = "Comparing based on the SHL catalog:\n\n" + "\n\n".join(lines)
    recs = [
        Recommendation(name=item.name, url=item.url, test_type="/".join(t.value for t in item.test_type))
        for item in found
    ]
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)
