#!/bin/bash
# Weaviate setup and initialization script

set -e

echo "🚀 Starting Weaviate with Docker Compose..."

# Start Weaviate
docker-compose up -d

echo "⏳ Waiting for Weaviate to be ready..."

# Wait for Weaviate to be healthy
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
        echo "✓ Weaviate is ready!"
        break
    fi
    echo "  Waiting... ($((attempt+1))/$max_attempts)"
    sleep 1
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "✗ Weaviate did not become ready in time"
    exit 1
fi

echo ""
echo "📊 Weaviate Information:"
echo "  URL: http://localhost:8080"
echo "  API Key: demo-key"
echo "  REST Endpoint: http://localhost:8080/v1"
echo "  gRPC: localhost:50051"
echo ""
echo "✓ Setup complete! You can now index repositories and search code."
echo ""
echo "Quick start:"
echo "  1. uv sync                                    # Install dependencies"
echo "  2. uv run python -m src.code_chat_agent.cli index /path/to/repo"
echo "  3. uv run python -m src.code_chat_agent.cli search 'OAuth'"
