import json
import uuid
import logging
from typing import List, Optional, AsyncGenerator, Tuple

from config import settings
from nvidia_client import nvidia_client
from vector_store import vector_store, ChunkMetadata
from schemas import SourceDocument

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """You are Labeeb, a document-grounded business assistant.

{style_instruction}

STRICT RULES — follow these without exception:
1. Answer ONLY using information found in the CONTEXT DOCUMENTS below.
2. Do NOT use any outside knowledge, general knowledge, or assumptions.
3. If the context documents do not contain a clear answer, respond with exactly:
   "I could not find relevant information in the uploaded documents to answer your question."
4. Never speculate, infer beyond what is written, or fill gaps with general knowledge.
5. Always cite which document/source your answer came from.
6. If the question is completely unrelated to the documents (e.g. general trivia, coding, weather), respond with:
   "This question is outside the scope of the uploaded documents. Please ask questions related to the available business documents."

CONTEXT DOCUMENTS:
{context}

---
Answer strictly from the context above:"""

# Used when use_rag=False is explicitly passed by the caller
SYSTEM_PROMPT_NO_RAG = """You are Labeeb, a document-grounded business assistant.

{style_instruction}

No documents have been uploaded yet. Inform the user that they need to upload relevant business documents first before you can answer questions. Do not answer from general knowledge."""

# Used when documents exist in the store but nothing was retrieved above the threshold
SYSTEM_PROMPT_NO_MATCH = """You are Labeeb, a document-grounded business assistant.

{style_instruction}

The user asked a question but no relevant content was found in the uploaded documents.

Respond with: "I could not find relevant information in the uploaded documents to answer your question. Please make sure you have uploaded the correct documents, or try rephrasing your question." """


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
        docs_exist: bool = False,
        conversation_history: Optional[List[dict]] = None,
    ) -> List[dict]:
        style_instruction = settings.CHAT_STYLES.get(style, settings.CHAT_STYLES["professional"])

        if context:
            # We have retrieved relevant chunks — answer strictly from them
            system_content = SYSTEM_PROMPT_BASE.format(
                style_instruction=style_instruction,
                context=context,
            )
        elif docs_exist:
            # Documents are uploaded but nothing matched this query
            system_content = SYSTEM_PROMPT_NO_MATCH.format(
                style_instruction=style_instruction,
            )
        else:
            # No documents uploaded at all
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
        docs_exist = vector_store.total_chunks > 0

        if use_rag and docs_exist:
            sources, context = await self.retrieve_context(user_message, document_ids)

        messages = self._build_messages(
            user_message, context, style,
            docs_exist=docs_exist,
            conversation_history=conversation_history,
        )
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
            docs_exist = vector_store.total_chunks > 0

            if use_rag and docs_exist:
                sources, context = await self.retrieve_context(user_message, document_ids)

            # Send sources metadata immediately
            sources_payload = {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "session_id": session_id,
                "message_id": message_id,
            }
            yield f"data: {json.dumps(sources_payload)}\n\n"

            messages = self._build_messages(
                user_message, context, style,
                docs_exist=docs_exist,
                conversation_history=conversation_history,
            )

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