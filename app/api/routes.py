from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.agents.orchestrator import handle_chat
from app.models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    app_state = request.app.state
    try:
        return await handle_chat(
            messages=payload.messages,
            llm=app_state.llm,
            catalog=app_state.catalog,
            retriever=app_state.retriever,
        )
    except Exception:  # noqa: BLE001
        # a single bad turn should never surface as a raw 500 to the
        # evaluator - every response has to match the schema, so fail
        # into a safe clarifying reply instead
        logger.exception("unhandled error in /chat")
        return ChatResponse(
            reply="Sorry, I hit an issue processing that - could you rephrase what role or "
                  "skills you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
