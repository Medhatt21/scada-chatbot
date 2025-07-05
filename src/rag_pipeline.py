"""RAG pipeline for retrieval-augmented generation using LlamaIndex and Ollama."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from loguru import logger

from config import settings
from database import db_manager


class RAGPipeline:
    """RAG pipeline for question answering using retrieved documents."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.llm = Ollama(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE,
            request_timeout=120.0,
        )
        self.embedding_model = OllamaEmbedding(
            model_name=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        
        self.qa_prompt = PromptTemplate(
            """You are a knowledgeable assistant specializing in artificial intelligence, machine learning, and computational research. 
            Use the following context from research papers to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Provide a comprehensive answer based on the context
            2. Cite specific papers or sections when relevant
            3. If the context is insufficient, acknowledge this limitation
            4. Keep your response accurate and evidence-based
            5. Format your response clearly with proper structure

            Answer:"""
        )
        
        self.system_prompt = """You are an AI research assistant with expertise in machine learning, artificial intelligence, and computational methods. You help users understand complex research concepts by providing clear, accurate explanations based on academic literature."""
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the user query."""
        try:
            embedding = await self.embedding_model.aget_text_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query."""
        try:
            # Generate query embedding
            query_embedding = await self.generate_query_embedding(query)
            
            # Perform similarity search
            results = await db_manager.similarity_search(query_embedding, limit=top_k)
            
            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            filename = doc.get('filename', 'Unknown')
            chunk_text = doc.get('chunk_text', '')
            distance = doc.get('distance', 0)
            
            context_parts.append(f"Document {i} (from {filename}, relevance: {1-distance:.3f}):\n{chunk_text}")
        
        return "\n\n".join(context_parts)
    
    async def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Format prompt with context and query
            prompt = self.qa_prompt.format(context=context, question=query)
            
            # Generate response
            response = await self.llm.acomplete(prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise
    
    async def query(self, question: str, session_id: str = None, top_k: int = 5) -> Dict[str, Any]:
        """Main query method that performs RAG pipeline."""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant documents
            documents = await self.retrieve_relevant_documents(question, top_k=top_k)
            
            # Format context
            context = self.format_context(documents)
            
            # Generate answer
            answer = await self.generate_answer(question, context)
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer,
                "context": context,
                "sources": [
                    {
                        "filename": doc.get('filename', 'Unknown'),
                        "relevance": 1 - doc.get('distance', 0),
                        "snippet": doc.get('chunk_text', '')[:200] + "..." if len(doc.get('chunk_text', '')) > 200 else doc.get('chunk_text', '')
                    }
                    for doc in documents
                ],
                "metadata": {
                    "num_retrieved_docs": len(documents),
                    "model_used": settings.OLLAMA_LLM_MODEL,
                    "embedding_model": settings.OLLAMA_EMBEDDING_MODEL
                }
            }
            
            # Save to chat history if session_id provided
            if session_id:
                await db_manager.save_chat_message(
                    session_id=session_id,
                    role="user",
                    content=question
                )
                await db_manager.save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "sources": [doc.get('filename', 'Unknown') for doc in documents],
                        "num_sources": len(documents)
                    }
                )
            
            logger.info(f"Query processed successfully, {len(documents)} sources used")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            history = await db_manager.get_chat_history(session_id, limit)
            return history
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the RAG pipeline."""
        try:
            # Test LLM
            llm_response = await self.llm.acomplete("Test query")
            llm_healthy = bool(llm_response.text)
            
            # Test embedding model
            test_embedding = await self.embedding_model.aget_text_embedding("test")
            embedding_healthy = len(test_embedding) > 0
            
            # Test database connection
            try:
                await db_manager.similarity_search([0.1] * 768, limit=1)
                db_healthy = True
            except Exception:
                db_healthy = False
            
            return {
                "llm_healthy": llm_healthy,
                "embedding_healthy": embedding_healthy,
                "database_healthy": db_healthy,
                "overall_healthy": llm_healthy and embedding_healthy and db_healthy
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "llm_healthy": False,
                "embedding_healthy": False,
                "database_healthy": False,
                "overall_healthy": False,
                "error": str(e)
            }
    
    async def get_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of available documents in the database."""
        try:
            async with db_manager.async_session() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text("""
                        SELECT 
                            d.filename,
                            d.created_at,
                            d.metadata,
                            COUNT(e.id) as chunk_count
                        FROM documents d
                        LEFT JOIN embeddings e ON d.id = e.document_id
                        GROUP BY d.id, d.filename, d.created_at, d.metadata
                        ORDER BY d.created_at DESC
                    """)
                )
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get available documents: {e}")
            raise
    
    def suggest_questions_sync(self, topic: str = None) -> List[str]:
        """Synchronously suggest questions based on available documents."""
        try:
            # Create basic suggestions without async DB calls to avoid conflicts
            suggestions = [
                "What are the main topics covered in the research papers?",
                "How does reinforcement learning apply to game development?",
                "What are the key findings about large language models?", 
                "Can you summarize the optimization algorithms discussed?",
                "What evolutionary computation methods are mentioned?",
                "How do neural networks perform in game playing?",
                "What are the applications of machine learning in games?",
                "What research methodologies are used in these papers?"
            ]
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return [
                "What are the main topics in the research papers?",
                "How does reinforcement learning work?",
                "What is mentioned about machine learning?",
                "Summarize the key findings",
                "What methodologies are discussed?"
            ]

    async def suggest_questions(self, topic: str = None) -> List[str]:
        """Suggest questions based on available documents."""
        return self.suggest_questions_sync(topic)


# Global RAG pipeline instance
rag_pipeline = RAGPipeline() 