"""Ties the whole pipeline together, matching the diagram:

    User -> POST /chat -> Conversation Understanding -> HiringContext
      -> enough context?
           No  -> ask a clarifying question
           Yes -> retrieve -> rerank -> generate reply -> format -> return

Scope guardrail (off_topic / injection) and compare both branch off the
same intent field Module 1 already produced, so there's no separate
classification pass for those.

"Refine" isn't really a separate code path - it's just recommend again.
Since the API is stateless and we re-read the whole conversation every
turn, "actually, add personality tests" naturally produces an updated
HiringContext with the new constraint folded in, and retrieval/reranking
just runs fresh against that. is_refine only changes the reply wording.
"""
from __future__ import annotations

from typing import List

from app.agents import clarification, comparison_agent, formatter, guardrails, recommendation_agent
from app.agents.conversation_understanding import understand
from app.agents.reranker import rerank
from app.catalog.catalog_loader import Catalog
from app.llm.client import LLMClient
from app.models.schemas import ChatMessage, ChatResponse, HiringContext, Intent, Role
from app.retrieval.retriever import Retriever

MAX_TURNS = 8


def _turn_count(messages: List[ChatMessage]) -> int:
    return len(messages)


def _last_user_message(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == Role.user:
            return m.content
    return ""


def _prior_recommendation_made(messages: List[ChatMessage]) -> bool:
    # crude but works fine: if the assistant already spoke once before,
    # treat this turn as a refine for phrasing purposes
    return any(m.role == Role.assistant for m in messages[:-1])


async def handle_chat(
    messages: List[ChatMessage],
    llm: LLMClient,
    catalog: Catalog,
    retriever: Retriever,
) -> ChatResponse:
    turn_count = _turn_count(messages)

    if turn_count > MAX_TURNS:
        return ChatResponse(
            reply="We've reached the end of this conversation's turn limit - thanks for the details!",
            recommendations=[],
            end_of_conversation=True,
        )

    context: HiringContext = await understand(llm, messages)

    if guardrails.is_out_of_scope(context):
        return guardrails.refusal_response(context)

    if context.intent == Intent.compare and context.comparison_targets:
        return comparison_agent.compare(catalog, context)

    if clarification.needs_clarification(context, turn_count):
        question = clarification.clarifying_question(context)
        reply = recommendation_agent.generate_clarification_reply(question)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    candidates = retriever.search(context, top_k=20)
    shortlist = rerank(candidates, context)

    is_refine = context.intent == Intent.refine or _prior_recommendation_made(messages)
    reply = recommendation_agent.generate_reply(shortlist, context, is_refine=is_refine)

    last_user_message = _last_user_message(messages)
    return formatter.build_response(reply=reply, items=shortlist, last_user_message=last_user_message)
