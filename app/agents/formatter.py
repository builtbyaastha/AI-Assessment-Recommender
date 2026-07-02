"""Module 6 - Formatter. Just assembles the final response, no decisions
made here - keeps every other module testable on its own without caring
about the wire format.
"""
from __future__ import annotations

import re
from typing import List

from app.models.schemas import CatalogItem, ChatResponse, Recommendation

CLOSING_PHRASES = re.compile(
    r"\b(that'?s all|thanks?,?\s*(that'?s|this is)?\s*(great|perfect|helpful)?|"
    r"no more questions|nothing else|that'?s everything|sounds good,?\s*thanks)\b",
    re.IGNORECASE,
)


def _catalog_item_to_recommendation(item: CatalogItem) -> Recommendation:
    return Recommendation(
        name=item.name,
        url=item.url,
        test_type="/".join(t.value for t in item.test_type),
    )


def user_signals_done(last_user_message: str) -> bool:
    return bool(CLOSING_PHRASES.search(last_user_message))


def build_response(
    reply: str,
    items: List[CatalogItem],
    last_user_message: str = "",
) -> ChatResponse:
    recommendations = [_catalog_item_to_recommendation(item) for item in items]
    end_of_conversation = bool(items) and user_signals_done(last_user_message)
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
