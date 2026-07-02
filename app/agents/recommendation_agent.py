"""Module 5 - Recommendation Agent.

Builds the natural-language reply for a shortlist. Template-based on
purpose, not an LLM call: the shortlist already came straight from the
catalog via retrieval + reranking, so having an LLM paraphrase a known
list into prose is just extra latency and an extra chance to make
something up about an assessment it didn't need to touch. Every fact in
the reply is read directly off the CatalogItem objects.
"""
from __future__ import annotations

from typing import List

from app.models.schemas import CatalogItem, HiringContext


def _describe_criteria(context: HiringContext) -> str:
    bits = []
    if context.role:
        bits.append(context.role)
    if context.seniority:
        bits.append(context.seniority)
    if context.technical_skills:
        bits.append("skills in " + ", ".join(context.technical_skills))
    if context.soft_skills:
        bits.append(", ".join(context.soft_skills))
    return " ".join(bits) if bits else "your requirements"


def generate_reply(items: List[CatalogItem], context: HiringContext, is_refine: bool) -> str:
    if not items:
        return (
            "I couldn't find a strong match in the SHL catalog for that combination - "
            "could you tell me a bit more, or broaden the criteria?"
        )

    criteria = _describe_criteria(context)
    count = len(items)

    if is_refine:
        return f"Updated - here are {count} assessment{'s' if count != 1 else ''} that fit {criteria}."
    return f"Here {'are' if count != 1 else 'is'} {count} assessment{'s' if count != 1 else ''} that fit {criteria}."


def generate_clarification_reply(question: str) -> str:
    return question
