import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

CHROMA_COLLECTION_NAME = "labeeb_chunks"


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    filename: str
    original_name: str
    chunk_index: int
    text: str


class VectorStore:
    """Thread-safe ChromaDB vector store with persistence."""

    def __init__(self):
        self._lock = asyncio.Lock()
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

        # Persistent ChromaDB client — data is saved to VECTOR_STORE_PATH on disk
        self._client = chromadb.PersistentClient(
            path=settings.VECTOR_STORE_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection.
        # ChromaDB normalises embeddings internally when using cosine space.
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready — "
            f"{self._collection.count()} vectors loaded."
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    async def add_chunks(
        self,
        embeddings: List[List[float]],
        chunks_meta: List[ChunkMetadata],
    ) -> None:
        """Add pre-computed embeddings + metadata to ChromaDB."""
        if not embeddings:
            return

        ids = [m.chunk_id for m in chunks_meta]
        documents = [m.text for m in chunks_meta]
        metadatas = [
            {
                "document_id": m.document_id,
                "filename": m.filename,
                "original_name": m.original_name,
                "chunk_index": m.chunk_index,
            }
            for m in chunks_meta
        ]

        async with self._lock:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(
            f"Added {len(embeddings)} chunks. "
            f"Total in collection: {self._collection.count()}"
        )

    async def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document."""
        async with self._lock:
            # Find all chunk IDs for this document
            results = self._collection.get(
                where={"document_id": document_id},
                include=[],  # only IDs needed
            )
            ids_to_delete = results["ids"]

            if not ids_to_delete:
                logger.info(f"No chunks found for document {document_id}")
                return 0

            self._collection.delete(ids=ids_to_delete)

        removed = len(ids_to_delete)
        logger.info(f"Deleted {removed} chunks for document {document_id}")
        return removed

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Return top-k chunks sorted by cosine similarity (descending)."""
        if self._collection.count() == 0:
            return []

        k = min(top_k or settings.TOP_K_RESULTS, self._collection.count())

        # Build optional where-filter to restrict to specific documents
        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        raw = self._collection.query(**query_kwargs)

        results: List[Tuple[ChunkMetadata, float]] = []
        for doc_text, meta, distance in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance = 1 − similarity  →  similarity = 1 − distance
            similarity = 1.0 - distance

            if similarity < settings.SIMILARITY_THRESHOLD:
                continue

            chunk = ChunkMetadata(
                chunk_id="",          # not stored back, not needed downstream
                document_id=meta["document_id"],
                filename=meta["filename"],
                original_name=meta["original_name"],
                chunk_index=int(meta["chunk_index"]),
                text=doc_text,
            )
            results.append((chunk, similarity))

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self._collection.count()

    def get_document_chunk_count(self, document_id: str) -> int:
        results = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        return len(results["ids"])


# Singleton — imported by rag_service.py and documents.py
vector_store = VectorStore()
