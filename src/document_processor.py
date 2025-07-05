"""Document processing module for PDF parsing and embedding generation."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import ollama
from pypdf import PdfReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from loguru import logger

from config import settings
from database import db_manager


class DocumentProcessor:
    """Document processor for handling PDF files and generating embeddings."""
    
    def __init__(self):
        """Initialize document processor."""
        self.embedding_model = OllamaEmbedding(
            model_name=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks using LlamaIndex."""
        try:
            document = Document(text=text, metadata=metadata or {})
            nodes = self.text_splitter.get_nodes_from_documents([document])
            return nodes
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.embedding_model.aget_text_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def process_pdf_file(self, pdf_path: Path) -> Tuple[int, int]:
        """Process a single PDF file and store embeddings."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_path}")
                return 0, 0
            
            # Store document
            document_id = await db_manager.insert_document(
                filename=pdf_path.name,
                content=text,
                metadata={"file_path": str(pdf_path), "file_size": pdf_path.stat().st_size}
            )
            
            # Chunk text
            chunks = self.chunk_text(text, metadata={"filename": pdf_path.name})
            chunk_texts = [chunk.text for chunk in chunks]
            
            if not chunk_texts:
                logger.warning(f"No chunks generated from {pdf_path}")
                return document_id, 0
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(chunk_texts)
            
            # Store embeddings
            chunk_metadata = [
                {
                    "filename": pdf_path.name,
                    "chunk_index": i,
                    "start_char_idx": chunk.start_char_idx,
                    "end_char_idx": chunk.end_char_idx,
                }
                for i, chunk in enumerate(chunks)
            ]
            
            await db_manager.insert_embeddings(
                document_id=document_id,
                chunks=chunk_texts,
                embeddings=embeddings,
                metadata=chunk_metadata
            )
            
            logger.info(f"Processed {pdf_path}: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return document_id, len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
    
    async def process_all_pdfs(self, input_dir: Path = None) -> Dict[str, Any]:
        """Process all PDF files in the input directory."""
        if input_dir is None:
            input_dir = settings.DATA_WAREHOUSE_PATH
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return {"processed_files": 0, "total_chunks": 0, "files": []}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            "processed_files": 0,
            "total_chunks": 0,
            "files": []
        }
        
        for pdf_file in pdf_files:
            try:
                document_id, chunk_count = await self.process_pdf_file(pdf_file)
                results["processed_files"] += 1
                results["total_chunks"] += chunk_count
                results["files"].append({
                    "filename": pdf_file.name,
                    "document_id": document_id,
                    "chunks": chunk_count
                })
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                results["files"].append({
                    "filename": pdf_file.name,
                    "error": str(e)
                })
        
        # Save processing results
        output_file = settings.DATA_MART_PATH / "processing_results.json"
        settings.DATA_MART_PATH.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processing complete: {results['processed_files']} files, {results['total_chunks']} chunks")
        return results
    
    async def check_ollama_models(self) -> Dict[str, bool]:
        """Check if required Ollama models are available."""
        try:
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            models_response = client.list()
            
            # Handle different response formats
            available_models = []
            
            # First, try to extract models from the response
            if hasattr(models_response, 'models'):
                # This is likely an ollama._types.ListResponse object
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Traditional dict response
                models_list = models_response['models']
            elif isinstance(models_response, list):
                # Direct list response
                models_list = models_response
            else:
                # Try to convert to dict if it has __dict__
                if hasattr(models_response, '__dict__'):
                    response_dict = models_response.__dict__
                    models_list = response_dict.get('models', [])
                else:
                    logger.warning(f"Unexpected models response format: {type(models_response)}")
                    models_list = []
            
            # Extract model names from the list
            for model in models_list:
                if hasattr(model, 'name') and model.name:
                    available_models.append(model.name)
                elif hasattr(model, 'model') and model.model:  # Sometimes it's stored as 'model'
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    # Handle dictionary format
                    name = model.get('name') or model.get('model')
                    if name:
                        available_models.append(name)
                elif isinstance(model, str):
                    # Handle string format
                    available_models.append(model)
            
            # Filter out empty names and clean up
            available_models = [name.strip() for name in available_models if name and name.strip()]
            logger.info(f"Available Ollama models: {available_models}")
            
            # Check if our required models are available
            llm_available = any(settings.OLLAMA_LLM_MODEL in model for model in available_models)
            embedding_available = any(settings.OLLAMA_EMBEDDING_MODEL in model for model in available_models)
            
            return {
                "llm_model": llm_available,
                "embedding_model": embedding_available,
                "available_models": available_models
            }
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            return {
                "llm_model": False,
                "embedding_model": False,
                "available_models": [],
                "error": str(e)
            }
    
    async def pull_required_models(self) -> Dict[str, bool]:
        """Pull required Ollama models if they don't exist."""
        try:
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            
            models_to_pull = [settings.OLLAMA_LLM_MODEL, settings.OLLAMA_EMBEDDING_MODEL]
            results = {}
            
            for model in models_to_pull:
                try:
                    logger.info(f"Pulling model: {model}")
                    client.pull(model)
                    results[model] = True
                    logger.info(f"Successfully pulled model: {model}")
                except Exception as e:
                    logger.error(f"Failed to pull model {model}: {e}")
                    results[model] = False
            
            return results
        except Exception as e:
            logger.error(f"Failed to pull models: {e}")
            return {model: False for model in [settings.OLLAMA_LLM_MODEL, settings.OLLAMA_EMBEDDING_MODEL]}


# Global document processor instance
document_processor = DocumentProcessor() 