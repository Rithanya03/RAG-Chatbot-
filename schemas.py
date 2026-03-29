from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


# ── Chat Schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    style: Optional[Literal["professional", "friendly", "technical", "concise"]] = Field(
        "professional", description="Chat response style"
    )
    use_rag: bool = Field(True, description="Whether to use RAG context")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to query")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is an Inventory Aging Trends Report?",
                "session_id": None,
                "style": "professional",
                "use_rag": True,
            }
        }


class SourceDocument(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    relevance_score: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    sources: List[SourceDocument] = []
    confidence: float
    style: str
    created_at: datetime


class StreamChunk(BaseModel):
    type: Literal["token", "sources", "done", "error"]
    content: str = ""
    sources: Optional[List[SourceDocument]] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None


# ── Document Schemas ──────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class DeleteDocumentResponse(BaseModel):
    success: bool
    message: str
    document_id: str


# ── Session Schemas ───────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: Optional[str] = None
    style: Optional[str] = "professional"


class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str]
    title: Optional[str]
    style: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: Optional[str]
    confidence: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionHistoryResponse(BaseModel):
    session: SessionResponse
    messages: List[MessageResponse]


# ── Health Schemas ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    documents_indexed: int
    nvidia_api: str


class FAQItem(BaseModel):
    question: str
    category: str


class FAQResponse(BaseModel):
    faqs: List[FAQItem]
