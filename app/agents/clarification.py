"""Module 2 - Clarification Engine.

Mostly just reads context.has_enough_context (Module 1 already decided
that), with one override: once we're close to the 8-turn cap, stop asking
questions and just recommend with whatever we've got. An agent that keeps
clarifying until the turns run out and never actually recommends anything
fails the user just as badly as one that recommends too early.
"""
from __future__ import annotations

from app.models.schemas import HiringContext

FORCE_RECOMMEND_AFTER_TURN = 6


def needs_clarification(context: HiringContext, turn_count: int) -> bool:
    if turn_count >= FORCE_RECOMMEND_AFTER_TURN:
        return False
    return not context.has_enough_context


def clarifying_question(context: HiringContext) -> str:
    if context.missing_info:
        missing = context.missing_info[0]
    else:
        missing = "the role and the key skills or competencies you're assessing for"

    if context.role and not context.technical_skills and not context.soft_skills:
        return (
            f"Got it - hiring for a {context.role}. What skills or competencies matter most "
            f"for this role (technical, soft, or both)? And roughly what seniority level are "
            f"you hiring at?"
        )
    if not context.role:
        return "Happy to help. What role are you hiring for, and what should the assessment focus on?"

    return f"Could you tell me a bit more about {missing}?"
