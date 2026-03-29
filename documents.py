import os
import uuid
import logging
import asyncio
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database import get_db, Document
from schemas import DocumentResponse, DocumentListResponse, DeleteDocumentResponse
from config import settings
from document_processor import process_document
from nvidia_client import nvidia_client
from vector_store import vector_store, ChunkMetadata

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


# ── Background ingestion task ─────────────────────────────────────────────────

async def ingest_document(document_id: str, file_path: str, original_name: str, filename: str):
    """Run the full RAG ingestion pipeline in the background."""
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return

        try:
            chunks, chunk_metas = await process_document(
                file_path=file_path,
                document_id=document_id,
                original_name=original_name,
                filename=filename,
            )

            # Batch embedding (NVIDIA API accepts up to 96 texts per call)
            BATCH_SIZE = 32
            all_embeddings = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                embeddings = await nvidia_client.embed_texts(batch)
                all_embeddings.extend(embeddings)
                await asyncio.sleep(0.1)  # rate-limit courtesy pause

            chunk_objects = [ChunkMetadata(**m) for m in chunk_metas]
            await vector_store.add_chunks(all_embeddings, chunk_objects)

            doc.status = "ready"
            doc.chunk_count = len(chunks)
            await db.commit()
            logger.info(f"Document '{original_name}' ingested: {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Ingestion failed for {original_name}: {e}", exc_info=True)
            doc.status = "failed"
            doc.error_message = str(e)[:500]
            await db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, summary="Upload a document for RAG indexing")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Supported: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read and check size
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Persist to disk
    document_id = str(uuid.uuid4())
    safe_name = f"{document_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Create DB record
    doc = Document(
        id=document_id,
        filename=safe_name,
        original_name=file.filename,
        file_type=ext,
        file_size=len(contents),
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Kick off background ingestion
    background_tasks.add_task(ingest_document, document_id, file_path, file.filename, safe_name)

    return doc


@router.get("/", response_model=DocumentListResponse, summary="List all uploaded documents")
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get document details")
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", response_model=DeleteDocumentResponse, summary="Delete a document and its vectors")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from vector store
    removed = await vector_store.delete_document(document_id)

    # Remove file from disk
    file_path = os.path.join(settings.UPLOAD_DIR, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    await db.delete(doc)
    await db.commit()

    return DeleteDocumentResponse(
        success=True,
        message=f"Document deleted. {removed} vector chunks removed.",
        document_id=document_id,
    )
