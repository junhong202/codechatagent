# Weaviate Integration Summary

## What's Been Implemented

### 1. **Weaviate Store** (`weaviate_store.py`)
- Full Weaviate integration with `weaviate-client` library
- Automatic schema creation for `CodeChunk` collection
- Three search modes:
  - **Hybrid Search**: Combines vector embeddings + BM25 keyword search (50/50 by default)
  - **Semantic Search**: Vector embeddings only (conceptual similarity)
  - **Keyword Search**: BM25 algorithm (exact term matching)
- Symbol search (find functions/classes by name)
- File-based search (get all chunks from a file)
- Language filtering across all search types

### 2. **Updated Agent** (`agent.py`)
- Replaced in-memory vector store with Weaviate
- Removed local embeddings provider dependency
- Simplified indexing to directly add chunks to Weaviate
- Three search modes via `search_code(..., search_type='hybrid'|'semantic'|'keyword')`
- Weaviate handles embeddings via `text2vec-openai` module

### 3. **Docker Setup**
- `docker-compose.yml`: Production-ready Weaviate setup
  - Latest Weaviate image
  - API authentication enabled
  - Health checks configured
  - Persistent data volume
  - Both HTTP (8080) and gRPC (50051) ports

### 4. **CLI Enhancements** (`cli.py`)
- Added Weaviate connection options (`--weaviate-url`, `--weaviate-key`)
- Added search type selector (`--type hybrid|semantic|keyword`)
- Added language filter (`--language python|javascript|...`)
- Added `clear` command to delete all indexed data
- Better status messages and error handling

### 5. **Comprehensive Tests** (`test_weaviate_store.py`)
- Hybrid search tests
- Semantic search tests
- Keyword search tests
- Symbol search tests
- File search tests

## Key Advantages of Weaviate

✅ **Hybrid Search**: Best of both worlds - semantic + keyword matching
✅ **Built-in Embeddings**: text2vec-openai module handles vectorization
✅ **BM25 Indexing**: Fast, proven keyword search algorithm
✅ **Scalability**: Handles large codebases efficiently
✅ **Persistence**: Data survives container restarts
✅ **Production Ready**: Weaviate is enterprise-grade

## Search Type Comparison

| Feature | Hybrid | Semantic | Keyword |
|---------|--------|----------|---------|
| Concept Match | ✓ | ✓ | ✗ |
| Exact Terms | ✓ | ✗ | ✓ |
| Speed | Medium | Medium | Fast |
| Best For | General queries | Intent-based | Syntax queries |

## Usage Examples

### CLI

```bash
# Start Weaviate
docker-compose up -d

# Index repos
uv run python -m src.code_chat_agent.cli index /path/to/repo

# Hybrid search (recommended)
uv run python -m src.code_chat_agent.cli search "OAuth token verification"

# Semantic search (find similar code)
uv run python -m src.code_chat_agent.cli search "authentication" --type semantic

# Keyword search (exact terms)
uv run python -m src.code_chat_agent.cli search "def login" --type keyword

# Filter by language
uv run python -m src.code_chat_agent.cli search "async" --type hybrid --language typescript
```

### Python API

```python
from src.code_chat_agent.agent import CodeChatAgent
from pathlib import Path

agent = CodeChatAgent()
agent.index_repositories([Path("/path/to/repo")])

# Hybrid search
results = agent.search_code("OAuth implementation", search_type="hybrid")

# Process results
for result in results:
    print(f"{result.repository}/{result.file_path}:{result.start_line}")
    print(f"Score: {result.score}")
    print(result.content)

agent.close()
```

## Configuration

### Connection Settings

```python
# Default (local Docker)
agent = CodeChatAgent()

# Custom Weaviate
agent = CodeChatAgent(
    weaviate_url="http://custom-host:8080",
    weaviate_key="your-api-key"
)

# Weaviate Cloud
agent = CodeChatAgent(
    weaviate_url="https://your-cluster.weaviate.network",
    weaviate_key="your-cloud-key"
)
```

## Architecture Flow

```
Code Repository
    ↓
Parser (extract chunks)
    ↓
Weaviate Store (add_chunk)
    ↓
Weaviate Backend:
  - Vectorizes with text2vec-openai
  - Creates BM25 index
  - Stores in persistent volume
    ↓
Search Query
    ↓
Weaviate Hybrid Search:
  - Vector similarity search
  - BM25 keyword search
  - Merge and rank results
    ↓
CodeSearchResult objects
    ↓
LLM (optional) → Answer
```

## Next Steps

1. ✅ Weaviate integration complete
2. Test with real repositories
3. Optional: Set up Weaviate Cloud for production
4. Add conversation history
5. Implement caching layer
6. Build web UI

## Troubleshooting

**Weaviate won't start:**
```bash
docker-compose down -v  # Remove volume
docker-compose up -d    # Start fresh
```

**Connection refused:**
```bash
curl http://localhost:8080/v1/.well-known/ready
docker-compose ps
```

**Clear all data:**
```bash
uv run python -m src.code_chat_agent.cli clear
```

## Performance Notes

- Weaviate automatically batches operations for efficiency
- BM25 search is significantly faster than semantic
- Hybrid search balances speed and accuracy
- Indexing time depends on repository size
- Search queries typically complete in <1 second
