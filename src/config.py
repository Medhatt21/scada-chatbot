"""Configuration settings for the RAG chatbot application."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # Database settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "scada_chatbot"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"  # Lightweight model
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"  # Lightweight embedding model
    
    # Application settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    
    # Data paths
    DATA_WAREHOUSE_PATH: Path = Path("data/d_warehouse")
    DATA_MART_PATH: Path = Path("data/d_mart")
    
    # Streamlit settings
    STREAMLIT_PORT: int = 8501
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields
    )
    
    @property
    def database_url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def async_database_url(self) -> str:
        """Get async database URL."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


# Global settings instance
settings = Settings() 