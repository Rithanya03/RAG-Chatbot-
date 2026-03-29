import json
import uuid
import logging
from typing import List, Optional, AsyncGenerator, Tuple

from config import settings
from nvidia_client import nvidia_client
from vector_store import vector_store, ChunkMetadata
from schemas import SourceDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """You are Labeeb, an intelligent business assistant with deep knowledge of procurement, inventory management, and enterprise operations. You help users understand business processes, reports, and workflows.

{style_instruction}

INSTRUCTIONS:
- Answer questions based ONLY on the provided context documents when context is available.
- If the context does not contain enough information, say so clearly and offer general guidance.
- Always be accurate, helpful, and structured in your responses.
- When referencing information from documents, mention which document it came from.
- For business processes, use clear numbered steps or bullet points when appropriate.
- If asked something outside the context, rely on your general business knowledge but clearly indicate this.

CONTEXT DOCUMENTS:
{context}

---
Answer the user's question based on the above context:"""

SYSTEM_PROMPT_NO_RAG = """You are Labeeb, an intelligent business assistant with deep knowledge of procurement, inventory management, and enterprise operations.

{style_instruction}

Answer the user's question with your best knowledge about business processes, procurement, inventory management, and enterprise workflows."""


class RAGService:

    # ── Context Retrieval ─────────────────────────────────────────────────────

    async def retrieve_context(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
    ) -> Tuple[List[SourceDocument], str]:
        """Embed query → search → return source docs and formatted context string."""
        query_embedding = await nvidia_client.embed_query(query)
        results = await vector_store.search(
            query_embedding,
            top_k=top_k or settings.TOP_K_RESULTS,
            document_ids=document_ids,
        )

        if not results:
            return [], ""

        sources = []
        context_parts = []

        for meta, score in results:
            excerpt = meta.text[:300] + "..." if len(meta.text) > 300 else meta.text
            sources.append(
                SourceDocument(
                    document_id=meta.document_id,
                    filename=meta.original_name,
                    chunk_index=meta.chunk_index,
                    relevance_score=round(score, 4),
                    excerpt=excerpt,
                )
            )
            context_parts.append(
                f"[Source: {meta.original_name}, chunk {meta.chunk_index + 1}]\n{meta.text}"
            )

        context_str = "\n\n---\n\n".join(context_parts)
        return sources, context_str

    # ── Message Building ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        context: str,
        style: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> List[dict]:
        style_instruction = settings.CHAT_STYLES.get(style, settings.CHAT_STYLES["professional"])

        if context:
            system_content = SYSTEM_PROMPT_BASE.format(
                style_instruction=style_instruction,
                context=context,
            )
        else:
            system_content = SYSTEM_PROMPT_NO_RAG.format(
                style_instruction=style_instruction,
            )

        messages = [{"role": "system", "content": system_content}]

        # Add conversation history (last 6 turns to stay within token limits)
        if conversation_history:
            messages.extend(conversation_history[-12:])

        messages.append({"role": "user", "content": user_message})
        return messages

    # ── Confidence Score ──────────────────────────────────────────────────────

    def _calculate_confidence(self, sources: List[SourceDocument], use_rag: bool) -> float:
        if not use_rag or not sources:
            return 0.5
        avg_score = sum(s.relevance_score for s in sources) / len(sources)
        # Normalise cosine similarity (0–1) to a user-friendly 0–1 range
        return round(min(avg_score * 1.2, 1.0), 3)

    # ── Non-Streaming Response ─────────────────────────────────────────────────

    async def generate_response(
        self,
        user_message: str,
        style: str = "professional",
        use_rag: bool = True,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> Tuple[str, List[SourceDocument], float]:
        sources, context = [], ""
        if use_rag and vector_store.total_chunks > 0:
            sources, context = await self.retrieve_context(user_message, document_ids)

        messages = self._build_messages(user_message, context, style, conversation_history)
        response = await nvidia_client.chat_completion(messages)
        answer = response["choices"][0]["message"]["content"]
        confidence = self._calculate_confidence(sources, use_rag)
        return answer, sources, confidence

    # ── Streaming Response ────────────────────────────────────────────────────

    async def generate_stream(
        self,
        user_message: str,
        session_id: str,
        message_id: str,
        style: str = "professional",
        use_rag: bool = True,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yields SSE-formatted strings:
          - {"type": "sources", ...}  — sent first
          - {"type": "token", "content": "..."} — streamed tokens
          - {"type": "done", "session_id": ..., "message_id": ...}
          - {"type": "error", "content": "..."}
        """
        try:
            sources, context = [], ""
            if use_rag and vector_store.total_chunks > 0:
                sources, context = await self.retrieve_context(user_message, document_ids)

            # Send sources metadata immediately
            sources_payload = {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "session_id": session_id,
                "message_id": message_id,
            }
            yield f"data: {json.dumps(sources_payload)}\n\n"

            messages = self._build_messages(user_message, context, style, conversation_history)

            async for token in nvidia_client.chat_completion_stream(messages):
                token_payload = {"type": "token", "content": token}
                yield f"data: {json.dumps(token_payload)}\n\n"

            confidence = self._calculate_confidence(sources, use_rag)
            done_payload = {
                "type": "done",
                "session_id": session_id,
                "message_id": message_id,
                "confidence": confidence,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_payload = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"


# Singleton
rag_service = RAGService()
