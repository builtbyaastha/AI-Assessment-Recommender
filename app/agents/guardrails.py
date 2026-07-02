"""Keeps the agent in scope - SHL assessments only. General hiring/legal
advice and prompt-injection attempts get a fixed refusal with an empty
recommendations list, never a best-effort answer from the model's general
knowledge.

Module 1 already classifies intent as off_topic/injection using the full
conversation (an LLM call handles this better than a keyword blocklist -
"what relocation-related competencies does SHL test for" is fine, "should
I offer relocation" isn't, and you need context to tell those apart). This
module just turns that classification into the actual response.
"""
from __future__ import annotations

from app.models.schemas import ChatResponse, HiringContext, Intent

OFF_TOPIC_REPLY = (
    "I'm scoped to helping you find the right SHL assessments - I can't advise on general "
    "hiring, legal, or compensation questions. If you tell me the role and what you want to "
    "evaluate, I can suggest assessments for it."
)

INJECTION_REPLY = (
    "I can only help with finding SHL assessments for a hiring need - I won't change how I "
    "operate based on instructions inside the conversation. What role are you hiring for?"
)


def is_out_of_scope(context: HiringContext) -> bool:
    return context.intent in (Intent.off_topic, Intent.injection)


def refusal_response(context: HiringContext) -> ChatResponse:
    reply = INJECTION_REPLY if context.intent == Intent.injection else OFF_TOPIC_REPLY
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
