import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, ChatSession, ChatMessage
from schemas import (
    ChatRequest, ChatResponse, SessionCreate, SessionResponse,
    SessionHistoryResponse, MessageResponse,
)
from rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Session Management ────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse, summary="Create a new chat session")
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=body.user_id,
        style=body.style or "professional",
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse, summary="Get session with message history")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = msgs_result.scalars().all()
    return SessionHistoryResponse(session=session, messages=messages)


@router.delete("/sessions/{session_id}", summary="Delete a chat session")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
    return {"success": True, "session_id": session_id}


# ── Helper: load conversation history ────────────────────────────────────────

async def _get_conversation_history(session_id: str, db: AsyncSession) -> List[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in messages]


async def _ensure_session(session_id: Optional[str], style: str, db: AsyncSession) -> str:
    if session_id:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        if result.scalar_one_or_none():
            return session_id
    new_session = ChatSession(id=str(uuid.uuid4()), style=style)
    db.add(new_session)
    await db.commit()
    return new_session.id


# ── Chat Endpoints ────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse, summary="Send a chat message (non-streaming)")
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session_id = await _ensure_session(body.session_id, body.style, db)
    history = await _get_conversation_history(session_id, db)

    answer, sources, confidence = await rag_service.generate_response(
        user_message=body.message,
        style=body.style,
        use_rag=body.use_rag,
        document_ids=body.document_ids,
        conversation_history=history,
    )

    message_id = str(uuid.uuid4())

    # Persist messages
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=body.message,
    )
    assistant_msg = ChatMessage(
        id=message_id,
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=json.dumps([s.model_dump() for s in sources]),
        confidence=confidence,
    )
    db.add(user_msg)
    db.add(assistant_msg)

    # Update session title if first message
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if session and not session.title:
        session.title = body.message[:80]
    await db.commit()

    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        answer=answer,
        sources=sources,
        confidence=confidence,
        style=body.style,
        created_at=datetime.now(timezone.utc),
    )


@router.post("/stream", summary="Send a chat message (SSE streaming)")
async def chat_stream(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session_id = await _ensure_session(body.session_id, body.style, db)
    history = await _get_conversation_history(session_id, db)
    message_id = str(uuid.uuid4())

    # Persist user message immediately
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    await db.commit()

    async def event_generator():
        full_answer = []
        final_sources = []
        final_confidence = 0.5

        async for chunk in rag_service.generate_stream(
            user_message=body.message,
            session_id=session_id,
            message_id=message_id,
            style=body.style,
            use_rag=body.use_rag,
            document_ids=body.document_ids,
            conversation_history=history,
        ):
            yield chunk
            # Parse to accumulate answer for DB persistence
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data.get("type") == "token":
                        full_answer.append(data.get("content", ""))
                    elif data.get("type") == "sources":
                        final_sources = data.get("sources", [])
                    elif data.get("type") == "done":
                        final_confidence = data.get("confidence", 0.5)
                except Exception:
                    pass

        # Persist assistant message after stream ends
        async with db.begin():
            assistant_msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role="assistant",
                content="".join(full_answer),
                sources=json.dumps(final_sources),
                confidence=final_confidence,
            )
            db.add(assistant_msg)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
