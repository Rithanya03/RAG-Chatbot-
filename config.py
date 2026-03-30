from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # NVIDIA NIM API
    # Set NVIDIA_API_KEY in your local .env file (never commit the real key)
    NVIDIA_API_KEY: str
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_LLM_MODEL: str = "meta/llama-3.1-70b-instruct"
    NVIDIA_EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"

    # Vector Store
    VECTOR_STORE_PATH: str = "./vector_store"
    EMBEDDING_DIMENSION: int = 1024

    # Document Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".md", ".csv"]

    # RAG Settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_CONTEXT_TOKENS: int = 4096

    # LLM Settings
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2
    STREAM: bool = True

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./labeeb.db"
    UPLOAD_DIR: str = "./uploads"

    # Server
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Chat Styles
    CHAT_STYLES: dict = {
        "professional": "You are a professional business assistant. Respond formally and concisely.",
        "friendly": "You are a friendly and approachable assistant. Use warm, conversational language.",
        "technical": "You are a technical expert assistant. Provide detailed, precise, technical responses.",
        "concise": "You are a concise assistant. Give brief, direct answers only.",
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()