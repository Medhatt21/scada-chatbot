"""Main Streamlit application for the RAG chatbot."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
from loguru import logger

from config import settings
from database import db_manager
from document_processor import document_processor
from rag_pipeline import rag_pipeline


def run_async_safe(coro):
    """Safely run async coroutine in Streamlit."""
    try:
        # Ensure we're not in an existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in a loop, we need to handle this differently
            logger.warning("Already in an event loop, using alternative approach")
            return None
        except RuntimeError:
            # No loop running, safe to use asyncio.run()
            pass
        
        return asyncio.run(coro)
    except Exception as e:
        logger.error(f"Async operation failed: {e}")
        return None


# Configure Streamlit page
st.set_page_config(
    page_title="SCADA RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    
    .source-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    
    .status-good {
        color: #28a745;
    }
    
    .status-bad {
        color: #dc3545;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


async def initialize_app():
    """Initialize the application components."""
    try:
        # Initialize database first
        await db_manager.initialize_database()
        
        # Check Ollama models (with timeout to avoid hanging)
        import asyncio
        try:
            model_status = await asyncio.wait_for(
                document_processor.check_ollama_models(), 
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Ollama model check timed out")
            model_status = {
                "llm_model": False,
                "embedding_model": False,
                "available_models": [],
                "error": "Timeout checking models"
            }
        
        return {
            "database_initialized": True,
            "models_available": model_status
        }
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        return {
            "database_initialized": False,
            "models_available": {"error": str(e)}
        }


def display_chat_message(message: Dict[str, Any], is_user: bool = True):
    """Display a chat message with styling."""
    message_class = "user-message" if is_user else "assistant-message"
    role = "👤 You" if is_user else "🤖 Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{role}</strong><br>
        {message['content']}
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources for assistant messages
    if not is_user and 'sources' in message:
        sources = message['sources']
        if sources:
            st.markdown("**Sources:**")
            for i, source in enumerate(sources, 1):
                with st.expander(f"📄 {source['filename']} (Relevance: {source['relevance']:.3f})"):
                    st.write(source['snippet'])


def sidebar_status():
    """Display system status in sidebar."""
    st.sidebar.markdown("## 🔧 System Status")
    
    # Check if initialization status is available
    if 'init_status' in st.session_state:
        init_status = st.session_state.init_status
        
        # Database status
        db_status = "✅ Connected" if init_status.get('database_initialized', False) else "❌ Disconnected"
        st.sidebar.markdown(f"**Database:** {db_status}")
        
        # Model status
        models = init_status.get('models_available', {})
        if 'error' in models:
            st.sidebar.markdown(f"**Models:** ❌ Error: {models['error']}")
        else:
            llm_status = "✅" if models.get('llm_model', False) else "❌"
            embed_status = "✅" if models.get('embedding_model', False) else "❌"
            st.sidebar.markdown(f"**LLM:** {llm_status} {settings.OLLAMA_LLM_MODEL}")
            st.sidebar.markdown(f"**Embeddings:** {embed_status} {settings.OLLAMA_EMBEDDING_MODEL}")
    else:
        st.sidebar.markdown("**Status:** Initializing...")


def sidebar_document_management():
    """Document management section in sidebar."""
    st.sidebar.markdown("## 📚 Document Management")
    
    # Process documents button
    if st.sidebar.button("🔄 Process Documents"):
        with st.spinner("Processing documents..."):
            results = run_async_safe(document_processor.process_all_pdfs())
            
            if results:
                st.sidebar.success(f"Processed {results['processed_files']} files, {results['total_chunks']} chunks")
                st.session_state.documents_processed = True
            else:
                st.sidebar.error("Processing failed - check logs for details")
    
    # Show available documents
    if st.sidebar.button("📋 Show Available Documents"):
        docs = run_async_safe(rag_pipeline.get_available_documents())
        
        if docs:
            st.sidebar.markdown("### Available Documents:")
            for doc in docs:
                st.sidebar.markdown(f"- **{doc['filename']}** ({doc['chunk_count']} chunks)")
        else:
            st.sidebar.markdown("No documents available or failed to fetch")


def sidebar_suggestions():
    """Display question suggestions in sidebar."""
    st.sidebar.markdown("## 💡 Suggested Questions")
    
    # Use synchronous version to avoid async conflicts
    try:
        suggestions = rag_pipeline.suggest_questions_sync()
        
        for suggestion in suggestions:
            if st.sidebar.button(f"❓ {suggestion}", key=f"suggest_{hash(suggestion)}"):
                st.session_state.suggested_question = suggestion
                st.rerun()
    except Exception as e:
        # Use a simple fallback for suggestions
        fallback_suggestions = [
            "What are the main topics in the research papers?",
            "How does reinforcement learning work?",
            "What is mentioned about machine learning?",
            "Summarize the key findings",
            "What methodologies are discussed?"
        ]
        for suggestion in fallback_suggestions:
            if st.sidebar.button(f"❓ {suggestion}", key=f"suggest_{hash(suggestion)}"):
                st.session_state.suggested_question = suggestion
                st.rerun()


def main_chat_interface():
    """Main chat interface."""
    st.markdown('<div class="main-header">🤖 SCADA RAG Chatbot</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Display welcome message
    if not st.session_state.messages:
        st.info("👋 Welcome! I'm your AI research assistant. I can help you understand the research papers in your knowledge base. Ask me questions about artificial intelligence, machine learning, or any topics covered in your documents.")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, is_user=message['role'] == 'user')
    
    # Chat input
    query = st.chat_input("Ask me anything about your research documents...")
    
    # Handle suggested question
    if 'suggested_question' in st.session_state:
        query = st.session_state.suggested_question
        del st.session_state.suggested_question
    
    # Process query
    if query:
        # Add user message
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        
        # Display user message
        display_chat_message(user_message, is_user=True)
        
        # Generate response
        with st.spinner("Thinking..."):
            response = run_async_safe(
                rag_pipeline.query(query, session_id=st.session_state.session_id)
            )
            
            if response:
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources'],
                    "metadata": response['metadata']
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Display assistant message
                display_chat_message(assistant_message, is_user=False)
            else:
                error_message = {
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error processing your query. Please make sure Ollama is running and the required models are available.",
                    "sources": []
                }
                st.session_state.messages.append(error_message)
                display_chat_message(error_message, is_user=False)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def admin_panel():
    """Admin panel for system management."""
    st.markdown("## ⚙️ Admin Panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Health Check")
        if st.button("🔍 Run Health Check"):
            with st.spinner("Checking system health..."):
                health = run_async_safe(rag_pipeline.health_check())
                
                if health:
                    st.json(health)
                else:
                    st.error("Health check failed - check logs for details")
    
    with col2:
        st.markdown("### Model Management")
        if st.button("📥 Pull Required Models"):
            with st.spinner("Pulling models from Ollama..."):
                results = run_async_safe(document_processor.pull_required_models())
                
                if results:
                    st.json(results)
                else:
                    st.error("Failed to pull models - check logs for details")


def main():
    """Main application function."""
    # Initialize app if not already done
    if 'init_status' not in st.session_state:
        with st.spinner("Initializing application..."):
            init_result = run_async_safe(initialize_app())
            st.session_state.init_status = init_result if init_result else {"database_initialized": False, "models_available": {"error": "Initialization failed"}}
    
    # Sidebar components
    sidebar_status()
    sidebar_document_management()
    sidebar_suggestions()
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Admin"])
    
    with tab1:
        main_chat_interface()
    
    with tab2:
        admin_panel()


if __name__ == "__main__":
    # Run the app
    try:
        main()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        st.error(f"Application failed to start: {e}")
        st.info("Please ensure PostgreSQL and Ollama are running and properly configured.") 