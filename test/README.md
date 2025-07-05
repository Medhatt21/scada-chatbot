# Test Scripts for Ollama RAG Chatbot

This directory contains test scripts to help debug and verify the functionality of the Ollama RAG chatbot system.

## Test Scripts

### 1. `test_ollama_models.py`
Tests Ollama model connectivity and functionality:
- Connection to Ollama server
- Model availability and listing
- LLM text generation
- Embedding generation
- Model pulling (if needed)

### 2. `test_rag_pipeline.py`
Tests the complete RAG pipeline:
- Database connectivity (PostgreSQL + pgvector)
- Document processing
- Embedding generation and storage
- Vector similarity search
- RAG query functionality
- Document ingestion from PDF files

### 3. `run_tests.py`
Test runner script to execute all tests or specific test suites.

## Usage

### Running All Tests
```bash
# Run all tests
python test/run_tests.py

# Or using the runner directly
python test/run_tests.py --test all
```

### Running Specific Test Suites
```bash
# Test only Ollama models
python test/run_tests.py --test ollama

# Test only RAG pipeline
python test/run_tests.py --test rag
```

### Running Individual Tests
```bash
# Test Ollama models directly
python test/test_ollama_models.py

# Test RAG pipeline directly
python test/test_rag_pipeline.py
```

## Prerequisites

Before running the tests, ensure:

1. **Docker services are running:**
   ```bash
   make up
   ```

2. **Ollama is accessible:**
   - The Ollama service should be running on `http://localhost:11434`
   - Required models should be available or will be pulled automatically

3. **Database is accessible:**
   - PostgreSQL should be running on `localhost:5432`
   - pgvector extension should be installed

## Troubleshooting Common Issues

### ❌ LLM Model Not Available
**Problem:** Frontend shows "❌ llama3.1:8b"

**Solutions:**
1. Run the Ollama test to check model availability:
   ```bash
   python test/test_ollama_models.py
   ```

2. Pull the model manually:
   ```bash
   docker exec -it scada_chatbot-ollama-1 ollama pull llama3.1:8b
   ```

3. Check if Ollama service is running:
   ```bash
   docker ps | grep ollama
   ```

### ❌ Embedding Model Not Available
**Problem:** Frontend shows "❌ nomic-embed-text"

**Solutions:**
1. Run the embedding test:
   ```bash
   python test/test_rag_pipeline.py
   ```

2. Pull the embedding model:
   ```bash
   docker exec -it scada_chatbot-ollama-1 ollama pull nomic-embed-text
   ```

3. Check model compatibility:
   ```bash
   docker exec -it scada_chatbot-ollama-1 ollama list
   ```

### ❌ Database Connection Issues
**Problem:** Database tests fail

**Solutions:**
1. Check if PostgreSQL is running:
   ```bash
   docker ps | grep postgres
   ```

2. Check database logs:
   ```bash
   make logs
   ```

3. Verify database connection:
   ```bash
   docker exec -it scada_chatbot-db-1 psql -U scada_user -d scada_chatbot -c "SELECT version();"
   ```

### ❌ Vector Extension Issues
**Problem:** pgvector extension not available

**Solutions:**
1. Check if extension is installed:
   ```bash
   docker exec -it scada_chatbot-db-1 psql -U scada_user -d scada_chatbot -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

2. Install the extension:
   ```bash
   docker exec -it scada_chatbot-db-1 psql -U scada_user -d scada_chatbot -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

## Test Output Interpretation

### ✅ Success Indicators
- `✅ Connected` - Service is accessible
- `✅ Model available` - Model is loaded and ready
- `✅ Generation successful` - Model can generate responses
- `✅ Embedding generation successful` - Embedding model works

### ❌ Failure Indicators
- `❌ Failed to connect` - Service is not accessible
- `❌ Model not found` - Model needs to be pulled
- `❌ Generation failed` - Model has issues
- `❌ No embedding generated` - Embedding model problems

### ⚠️ Warning Indicators
- `⚠️ Attempting to pull` - Model is being downloaded
- `⚠️ No source nodes found` - RAG pipeline works but no relevant documents

## Model Requirements

### Required Models
- **LLM Model:** `llama3.1:8b` (~4.7GB)
- **Embedding Model:** `nomic-embed-text` (~274MB)

### Pulling Models Manually
```bash
# Pull LLM model
docker exec -it scada_chatbot-ollama-1 ollama pull llama3.1:8b

# Pull embedding model
docker exec -it scada_chatbot-ollama-1 ollama pull nomic-embed-text

# List all models
docker exec -it scada_chatbot-ollama-1 ollama list
```

## Performance Notes

- **Model Loading:** First query may take 30-60 seconds as models load into memory
- **M3 Max Optimization:** Models are optimized for Apple Silicon
- **Memory Usage:** Ensure sufficient system RAM (16GB+ recommended)
- **Disk Space:** Models require ~5GB total disk space

## Getting Help

If tests continue to fail after troubleshooting:

1. Check the main application logs:
   ```bash
   make logs
   ```

2. Restart the services:
   ```bash
   make down
   make up
   ```

3. Check system resources:
   ```bash
   docker stats
   ```

4. Verify Docker health:
   ```bash
   docker system df
   docker system prune  # If needed
   ``` 