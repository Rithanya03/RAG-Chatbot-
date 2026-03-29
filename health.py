from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database import get_db, Document
from schemas import HealthResponse, FAQResponse, FAQItem
from vector_store import vector_store
from nvidia_client import nvidia_client
from config import settings

router = APIRouter()

DEFAULT_FAQS = [
    FAQItem(question="What is an Inventory Aging Trends Report? Why is it important?", category="Inventory"),
    FAQItem(question="Differentiation between Unreleased PRs and PRs Under Procurement?", category="Procurement"),
    FAQItem(question="What are the Purchase Requisition and how are they used?", category="Procurement"),
    FAQItem(question="What are the challenges of Material Consumption? Why is it important?", category="Inventory"),
    FAQItem(question="What is a Purchase order material type? Why is Material Type important in a PO?", category="Purchase Orders"),
    FAQItem(question="Summary of Purchase Order by Material Type Report? Why is it important?", category="Purchase Orders"),
    FAQItem(question="What are the Purchase Requisition and how are they used?", category="Procurement"),
    FAQItem(question="How do I track open purchase orders?", category="Purchase Orders"),
    FAQItem(question="What is the difference between a PR and a PO?", category="Procurement"),
    FAQItem(question="How does inventory aging affect business decisions?", category="Inventory"),
]


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check(db: AsyncSession = Depends(get_db)):
    # Check documents indexed
    result = await db.execute(select(func.count()).select_from(Document).where(Document.status == "ready"))
    doc_count = result.scalar() or 0

    # Check NVIDIA API
    api_ok = await nvidia_client.check_connectivity()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store=f"{vector_store.total_chunks} chunks indexed",
        documents_indexed=doc_count,
        nvidia_api="connected" if api_ok else "unreachable (check API key)",
    )


@router.get("/faqs", response_model=FAQResponse, summary="Get suggested FAQ questions")
async def get_faqs():
    return FAQResponse(faqs=DEFAULT_FAQS)


@router.get("/styles", summary="Get available chat styles")
async def get_styles():
    return {
        "styles": [
            {"id": k, "label": k.capitalize(), "description": v}
            for k, v in settings.CHAT_STYLES.items()
        ]
    }
