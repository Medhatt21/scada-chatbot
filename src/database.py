"""Database operations for the RAG chatbot application."""

import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import threading

import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from loguru import logger

from config import settings


class DatabaseManager:
    """Database manager for PostgreSQL operations."""
    
    def __init__(self):
        """Initialize database manager."""
        self.engine = create_engine(settings.database_url)
        self.async_engine = create_async_engine(
            settings.async_database_url,
            pool_size=1,  # Reduce to single connection to avoid concurrency issues
            max_overflow=0,  # No overflow connections
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.async_session = sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )
        self._initialization_lock = None
        self._initialized = False
        self._session_lock = threading.Lock()
        self._db_semaphore = None
    
    def _ensure_async_objects(self):
        """Ensure async objects are created in the correct context."""
        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()
        if self._db_semaphore is None:
            self._db_semaphore = asyncio.Semaphore(1)
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session with proper error handling and concurrency control."""
        self._ensure_async_objects()
        async with self._db_semaphore:  # Ensure only one operation at a time
            session = None
            try:
                await asyncio.sleep(0.01)  # Small delay to prevent race conditions
                session = self.async_session()
                yield session
            except Exception as e:
                if session:
                    try:
                        await session.rollback()
                    except Exception as rollback_error:
                        logger.warning(f"Failed to rollback session: {rollback_error}")
                logger.error(f"Database session error: {e}")
                raise
            finally:
                if session:
                    try:
                        await session.close()
                    except Exception as close_error:
                        logger.warning(f"Failed to close session: {close_error}")
                        pass
    
    async def initialize_database(self) -> None:
        """Initialize database with required extensions and tables."""
        self._ensure_async_objects()
        
        # Use lock to prevent concurrent initialization
        async with self._initialization_lock:
            if self._initialized:
                logger.info("Database already initialized, skipping...")
                return
                
            try:
                # Create database if it doesn't exist
                await self._create_database_if_not_exists()
                
                # Initialize extensions and tables in a single transaction
                async with self.async_engine.begin() as conn:
                    # Enable pgvector extension
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    
                    # Create documents table
                    await conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    
                    # Create embeddings table
                    await conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS embeddings (
                            id SERIAL PRIMARY KEY,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            chunk_text TEXT NOT NULL,
                            embedding vector(768),  -- nomic-embed-text dimension
                            chunk_index INTEGER NOT NULL,
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    
                    # Create index for vector similarity search
                    await conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS embeddings_embedding_idx 
                        ON embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """))
                    
                    # Create chat sessions table
                    await conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS chat_sessions (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(255) UNIQUE NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    
                    # Create chat messages table
                    await conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                            role VARCHAR(50) NOT NULL,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    
                    # All operations completed successfully
                    self._initialized = True
                    logger.info("Database initialized successfully")
                    
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
    
    async def _create_database_if_not_exists(self) -> None:
        """Create database if it doesn't exist."""
        try:
            conn = await asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database="postgres"  # Connect to default database
            )
            
            # Check if database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                settings.POSTGRES_DB
            )
            
            if not db_exists:
                await conn.execute(f"CREATE DATABASE {settings.POSTGRES_DB}")
                logger.info(f"Created database: {settings.POSTGRES_DB}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def insert_document(self, filename: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a document into the database."""
        import json
        
        async with self.get_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO documents (filename, content, metadata)
                    VALUES (:filename, :content, :metadata)
                    RETURNING id
                """),
                {
                    "filename": filename, 
                    "content": content, 
                    "metadata": json.dumps(metadata) if metadata else None
                }
            )
            document_id = result.fetchone()[0]
            return document_id
    
    async def insert_embeddings(self, document_id: int, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Insert embeddings into the database."""
        import json
        
        async with self.get_session() as session:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = metadata[i] if metadata else {}
                # Format embedding as PostgreSQL vector string
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                await session.execute(
                    text("""
                        INSERT INTO embeddings (document_id, chunk_text, embedding, chunk_index, metadata)
                        VALUES (:document_id, :chunk_text, CAST(:embedding AS vector), :chunk_index, :metadata)
                    """),
                    {
                        "document_id": document_id,
                        "chunk_text": chunk,
                        "embedding": embedding_str,
                        "chunk_index": i,
                        "metadata": json.dumps(chunk_metadata) if chunk_metadata else None
                    }
                )
    
    async def similarity_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using vector embeddings."""
        async with self.get_session() as session:
            # Format query embedding as PostgreSQL vector string
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            result = await session.execute(
                text("""
                    SELECT 
                        e.chunk_text,
                        e.metadata,
                        d.filename,
                        d.metadata as doc_metadata,
                        (e.embedding <=> CAST(:query_embedding AS vector)) as distance
                    FROM embeddings e
                    JOIN documents d ON e.document_id = d.id
                    ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :limit
                """),
                {"query_embedding": query_embedding_str, "limit": limit}
            )
            # Convert SQLAlchemy rows to dictionaries properly
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
    
    async def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        async with self.get_session() as session:
            result = await session.execute(
                text("""
                    SELECT role, content, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = :session_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"session_id": session_id, "limit": limit}
            )
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
    
    async def save_chat_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a chat message to the database."""
        import json
        
        async with self.get_session() as session:
            # Ensure session exists
            await session.execute(
                text("""
                    INSERT INTO chat_sessions (session_id) 
                    VALUES (:session_id) 
                    ON CONFLICT (session_id) DO NOTHING
                """),
                {"session_id": session_id}
            )
            
            # Insert message
            await session.execute(
                text("""
                    INSERT INTO chat_messages (session_id, role, content, metadata)
                    VALUES (:session_id, :role, :content, :metadata)
                """),
                {
                    "session_id": session_id, 
                    "role": role, 
                    "content": content, 
                    "metadata": json.dumps(metadata) if metadata else None
                }
            )
    
    async def close(self) -> None:
        """Close database connections."""
        await self.async_engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


def get_database_connection():
    """Get a synchronous database connection for testing."""
    import psycopg2
    from psycopg2 import sql
    
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None 