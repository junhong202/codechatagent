# Code Chat Agent

A multi-repository chat agent that answers questions across codebases in multiple languages without hallucinating. Uses **Weaviate** for hybrid semantic and keyword search, ensuring grounded answers based on actual code.

## Features

- **Weaviate-Powered Search**: Uses Weaviate for semantic embeddings + BM25 keyword search
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search with configurable weights
- **Multi-Repository Search**: Index and search across multiple code repositories  
- **Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java, C++, and more
- **Code-Grounded Answers**: All responses are based on actual code from repositories
- **No Hallucination**: Avoids making up code or APIs that don't exist
- **Context-Aware**: Provides repository, file path, and line numbers with results
- **LLM Integration**: Optional OpenAI integration for natural language answers
- **Docker Ready**: Run Weaviate locally with Docker Compose

## Architecture

```
Git Repos
  ↓
Repo Ingestion Pipeline    (RepositoryIndexer)
  ↓
Code Parsing & Chunking    (CodeParser)
  ↓
Weaviate Ingestion         (WeaviateStore)
  ↓
Vector DB + Keyword Index  (Weaviate text2vec-openai + BM25)
  ↓
Hybrid Retrieval           (Semantic + Keyword)
  ↓
LLM (Answer + Code)        (CodeChatAgent)
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- `uv` for package management
- OpenAI API key (optional, for embeddings and answers)

### 1. Start Weaviate

```bash
# Start Weaviate in Docker
docker-compose up -d

# Verify it's running
curl http://localhost:8080/v1/.well-known/ready
```

Weaviate will be available at `http://localhost:8080`

### 2. Setup Python Environment

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```

### 3. Index Repositories

```bash
# Index one or more repositories
uv run python -m src.code_chat_agent.cli index /path/to/repo1 /path/to/repo2

# The data is persisted in Weaviate (not in local files)
```

### 4. Search and Ask Questions

```bash
# Hybrid search (semantic + keyword)
uv run python -m src.code_chat_agent.cli search "OAuth implementation"

# Semantic-only search
uv run python -m src.code_chat_agent.cli search "authentication" --type semantic

# Keyword-only search
uv run python -m src.code_chat_agent.cli search "def login" --type keyword

# Ask questions (with LLM)
uv run python -m src.code_chat_agent.cli ask "How do I implement OAuth 2.0?" --use-llm

# Filter by language
uv run python -m src.code_chat_agent.cli search "async function" --language typescript
```

### Python API

```python
from src.code_chat_agent.agent import CodeChatAgent
from pathlib import Path

# Create agent connected to Weaviate
agent = CodeChatAgent(
    weaviate_url="http://localhost:8080",
    weaviate_key="demo-key"
)

# Index repositories
agent.index_repositories([Path("/path/to/repo1"), Path("/path/to/repo2")])

# Hybrid search (semantic + keyword)
results = agent.search_code("show me OAuth code", search_type="hybrid")

# Semantic-only search
results = agent.search_code("authentication patterns", search_type="semantic")

# Keyword search
results = agent.search_code("def login", search_type="keyword")

# Display results
for result in results:
    print(f"{result.repository}/{result.file_path}:{result.start_line}")
    print(f"Symbol: {result.symbol_name}")
    print(f"Score: {result.score}")
    print(result.content)

# Answer questions
answer = agent.answer_question("How to implement OAuth?", results)
print(answer)

agent.close()
```

## Search Types

### Hybrid Search (Default)
Combines semantic embeddings with BM25 keyword search.
- **Best for**: Most queries, natural language questions
- **Weight**: 50% semantic + 50% keyword (configurable in code)

### Semantic Search
Uses only vector embeddings.
- **Best for**: Understanding code intent and patterns
- **Better for**: Finding similar implementations

### Keyword Search
Uses BM25 algorithm for exact term matching.
- **Best for**: Finding specific functions, variable names, patterns
- **Better for**: Precise syntax-based queries

## Weaviate Configuration

### Environment Variables

```bash
# Connection settings
WEAVIATE_URL=http://localhost:8080
WEAVIATE_KEY=demo-key

# Embeddings model (in Weaviate schema)
# Currently configured for: text-embedding-3-small
```

### Persistence

Weaviate data is persisted in Docker volume `weaviate_data:` - data survives container restarts.

### Scaling

For production:
- Use managed Weaviate Cloud
- Configure replication and sharding
- Update docker-compose.yml with production settings

## Project Structure

```
.
├── src/code_chat_agent/
│   ├── __init__.py
│   ├── agent.py           # Main chat agent with Weaviate
│   ├── cli.py             # Command-line interface
│   ├── indexer.py         # Repository ingestion
│   ├── parser.py          # Code parsing & chunking
│   ├── weaviate_store.py  # Weaviate integration
│   └── retriever.py       # Hybrid retrieval (legacy)
├── tests/
│   ├── test_agent.py
│   ├── test_parser.py
│   ├── test_weaviate_store.py
│   └── ...
├── docker-compose.yml     # Weaviate Docker setup
├── pyproject.toml         # Project config
├── README.md
└── .github/copilot-instructions.md
```

## Development

### Code Quality

```bash
# Format code
uv run ruff format src tests

# Lint
uv run ruff check src tests

# Type checking
uv run mypy src

# Tests with coverage
uv run pytest

# Tests with Weaviate integration
uv run pytest tests/test_weaviate_store.py -v
```

### Dependencies

- **Core**: pydantic, httpx, python-dotenv, click, weaviate-client
- **LLM**: openai (optional)
- **Dev**: pytest, mypy, ruff, black

## Troubleshooting

### Weaviate Connection Issues

```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# View Weaviate logs
docker-compose logs weaviate

# Restart Weaviate
docker-compose restart weaviate
```

### Clear Indexed Data

```bash
# Via CLI
uv run python -m src.code_chat_agent.cli clear

# Via Python API
agent = CodeChatAgent()
agent.vector_store.clear()
agent.close()
```

### Embeddings Issues

If using OpenAI embeddings:
- Set `OPENAI_API_KEY` environment variable
- Ensure API key has embedding permissions

## Next Steps

- [ ] Add support for Weaviate Cloud
- [ ] Implement caching layer
- [ ] Add web UI
- [ ] Support for private repositories
- [ ] Fine-tuning for domain-specific knowledge
- [ ] Add code summarization
- [ ] Implement conversation history