# SCADA RAG Chatbot Makefile
# Usage: make [command]

.PHONY: help init up down logs clean status models restart rebuild

# Default target
help:
	@echo "SCADA RAG Chatbot - Available Commands:"
	@echo ""
	@echo "  init     - Initialize the application (setup models and dependencies)"
	@echo "  up       - Start all services in detached mode"
	@echo "  down     - Stop all services and remove containers"
	@echo "  logs     - Show logs from all services"
	@echo "  status   - Show status of all services"
	@echo "  models   - Pull required Ollama models"
	@echo "  restart  - Restart all services"
	@echo "  rebuild  - Rebuild and restart services"
	@echo "  clean    - Clean up containers, images, and volumes"
	@echo ""
	@echo "Application will be available at:"
	@echo "  - Streamlit App: http://localhost:8501"
	@echo "  - PostgreSQL:    localhost:5432"
	@echo "  - Ollama:        http://localhost:11434"

# Initialize the application
init:
	@echo "🚀 Initializing SCADA RAG Chatbot..."
	@echo "📦 Creating necessary directories..."
	@mkdir -p data/d_mart logs
	@echo "🐳 Building Docker images..."
	@docker-compose build
	@echo "📥 Starting services..."
	@docker-compose up -d postgres ollama
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "🤖 Pulling Ollama models..."
	@$(MAKE) models
	@echo "✅ Initialization complete!"
	@echo "🎯 Run 'make up' to start the application"

# Start all services
up:
	@echo "🚀 Starting SCADA RAG Chatbot..."
	@docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 15
	@echo "✅ All services are running!"
	@echo "🌐 Application available at: http://localhost:8501"

# Stop all services
down:
	@echo "🛑 Stopping SCADA RAG Chatbot..."
	@docker-compose down
	@echo "✅ All services stopped!"

# Show logs from all services
logs:
	@echo "📋 Showing logs from all services..."
	@docker-compose logs -f

# Show specific service logs
logs-app:
	@docker-compose logs -f app

logs-postgres:
	@docker-compose logs -f postgres

logs-ollama:
	@docker-compose logs -f ollama

# Show status of all services
status:
	@echo "📊 Service Status:"
	@docker-compose ps

# Pull required Ollama models
models:
	@echo "📥 Pulling required Ollama models..."
	@echo "🤖 Pulling LLM model: llama3.2:3b"
	@docker exec -it scada_chatbot_ollama ollama pull llama3.2:3b || true
	@echo "🧠 Pulling embedding model: nomic-embed-text"
	@docker exec -it scada_chatbot_ollama ollama pull nomic-embed-text || true
	@echo "✅ Models pulled successfully!"

# Restart all services
restart:
	@echo "🔄 Restarting SCADA RAG Chatbot..."
	@docker-compose restart
	@echo "✅ All services restarted!"

# Rebuild and restart services
rebuild:
	@echo "🔨 Rebuilding and restarting services..."
	@docker-compose down
	@docker-compose build --no-cache
	@docker-compose up -d
	@echo "✅ Services rebuilt and restarted!"

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v --rmi all --remove-orphans
	@docker system prune -f
	@echo "✅ Cleanup complete!"

# Development commands
dev-up:
	@echo "🔧 Starting development environment..."
	@docker-compose -f docker-compose.yml up -d postgres ollama
	@echo "💻 Run the app locally with: streamlit run src/streamlit_app.py"

dev-down:
	@echo "🔧 Stopping development environment..."
	@docker-compose down

# Database management
db-shell:
	@echo "🐘 Connecting to PostgreSQL..."
	@docker exec -it scada_chatbot_postgres psql -U postgres -d scada_chatbot

db-backup:
	@echo "💾 Creating database backup..."
	@docker exec scada_chatbot_postgres pg_dump -U postgres -d scada_chatbot > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created!"

db-restore:
	@echo "🔄 Restoring database from backup..."
	@echo "Usage: make db-restore BACKUP_FILE=backup_file.sql"
	@docker exec -i scada_chatbot_postgres psql -U postgres -d scada_chatbot < $(BACKUP_FILE)

# Health checks
health:
	@echo "🏥 Checking application health..."
	@echo "📊 Service Status:"
	@docker-compose ps
	@echo ""
	@echo "🐘 PostgreSQL Health:"
	@docker exec scada_chatbot_postgres pg_isready -U postgres -d scada_chatbot
	@echo ""
	@echo "🤖 Ollama Health:"
	@curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | head -5
	@echo ""
	@echo "🌐 Streamlit Health:"
	@curl -s http://localhost:8501/_stcore/health || echo "❌ Streamlit not healthy"

# Show application information
info:
	@echo "📋 SCADA RAG Chatbot Information:"
	@echo ""
	@echo "🌐 Application URLs:"
	@echo "  - Streamlit App: http://localhost:8501"
	@echo "  - Ollama API:    http://localhost:11434"
	@echo ""
	@echo "🐘 Database Info:"
	@echo "  - Host: localhost"
	@echo "  - Port: 5432"
	@echo "  - Database: scada_chatbot"
	@echo "  - User: postgres"
	@echo ""
	@echo "🤖 Ollama Models:"
	@echo "  - LLM: llama3.2:3b"
	@echo "  - Embeddings: nomic-embed-text"
	@echo ""
	@echo "📁 Data Directories:"
	@echo "  - Source: data/d_warehouse/"
	@echo "  - Processed: data/d_mart/"
	@echo "  - Logs: logs/" 