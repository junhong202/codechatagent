"""Weaviate vector store integration."""

from dataclasses import dataclass
from typing import Optional
import logging

import weaviate
from weaviate.classes.config import Property, DataType, Vectorizers
from weaviate.util import generate_uuid5

from src.code_chat_agent.embeddings import EmbeddingsProvider

logger = logging.getLogger(__name__)


@dataclass
class WeaviateConfig:
    """Configuration for Weaviate connection."""

    url: str = "http://localhost:8080"


class WeaviateStore:
    """Weaviate vector store for code indexing and retrieval."""

    def __init__(self, config: Optional[WeaviateConfig] = None) -> None:
        """Initialize Weaviate store."""
        self.config = config or WeaviateConfig()
        logger.info(f"Connecting to Weaviate at {self.config.url}")
        
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            skip_init_checks=False,
        )
        
        # Initialize embeddings provider with OpenAI
        self.embeddings = EmbeddingsProvider()
        
        # Verify connection
        if not self.client.is_ready():
            logger.error(f"Weaviate not ready at {self.config.url}")
            raise ConnectionError(f"Cannot connect to Weaviate at {self.config.url}")
        
        logger.info("✓ Connected to Weaviate")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure CodeChunk collection exists."""
        try:
            self.client.collections.get("CodeChunk")
            logger.info("✓ CodeChunk collection already exists")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            # Collection doesn't exist, create it with no vectorizer
            # We'll manually add embeddings
            logger.info("Creating CodeChunk collection...")
            self.client.collections.create(
                name="CodeChunk",
                description="Code chunks from repositories with CodeBERT embeddings",
                properties=[
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="start_line", data_type=DataType.INT),
                    Property(name="end_line", data_type=DataType.INT),
                    Property(name="language", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="symbol_name", data_type=DataType.TEXT),
                    Property(name="symbol_type", data_type=DataType.TEXT),
                    Property(name="repository", data_type=DataType.TEXT),
                ],
                vectorizer_config=Vectorizers.none(),
            )
            logger.info("✓ CodeChunk collection created")

    def add_chunk(
        self,
        file_path: str,
        content: str,
        language: str,
        start_line: int,
        end_line: int,
        symbol_name: Optional[str] = None,
        symbol_type: Optional[str] = None,
        repository: str = "unknown",
    ) -> str:
        """Add a code chunk to the store with CodeBERT embedding."""
        chunks = self.client.collections.get("CodeChunk")
        
        chunk_id = generate_uuid5(
            f"{file_path}:{start_line}:{end_line}"
        )
        
        logger.debug(f"Adding chunk: {file_path}:{start_line}-{end_line} ({symbol_name or 'no-symbol'})")
        
        # Generate embedding for content (truncate if too large for embedding model)
        # Try embedding with progressively smaller slices to avoid token limits
        candidates = [12000, 8000]
        embedding = None
        for max_chars in candidates:
            text_for_embedding = content if len(content) <= max_chars else content[:max_chars]
            if len(content) > max_chars:
                logger.debug(f"Chunk too large ({len(content)} chars), truncating to {max_chars} for embedding")
            try:
                embedding = self.embeddings.embed_text(text_for_embedding)
                break
            except Exception as e:
                logger.warning(f"Embedding failed for chunk {chunk_id} with max_chars={max_chars}: {e}")
                embedding = None

        if embedding is None:
            raise RuntimeError(f"Embedding failed for chunk {chunk_id} after truncation attempts")

        # If an object with the same UUID already exists, skip insertion
        try:
            exists = self.client.data_object.exists(chunk_id, class_name="CodeChunk")
        except Exception:
            exists = False

        if exists:
            logger.debug(f"Chunk {chunk_id} already exists — skipping insert")
            return chunk_id

        # Insert the object with vector
        try:
            chunks.data.insert(
                uuid=chunk_id,
                vector=embedding,
                properties={
                    "file_path": file_path,
                    "content": content,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_name": symbol_name or "",
                    "symbol_type": symbol_type or "",
                    "repository": repository,
                },
            )
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            msg = str(e)
            if "already exists" in msg or "already exists" in getattr(e, 'args', [''])[0]:
                logger.debug(f"Insert conflict for {chunk_id} — object already exists, skipping")
                return chunk_id
            raise

        return chunk_id

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        alpha: float = 0.5,
        language_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search combining semantic (vector) and keyword search.
        
        Args:
            query: Search query
            limit: Maximum results
            alpha: Balance between keyword (0.0) and vector (1.0) search
            language_filter: Optional language filter (currently unused)
            
        Returns:
            List of results with metadata
        """
        chunks = self.client.collections.get("CodeChunk")
        
        # Embed query with CodeBERT
        query_embedding = self.embeddings.embed_text(query)
        
        # Hybrid search with embedded query
        results = chunks.query.hybrid(
            query=query,
            vector=query_embedding,
            limit=limit,
            alpha=alpha,
            return_properties=[
                "file_path",
                "content",
                "language",
                "start_line",
                "end_line",
                "symbol_name",
                "symbol_type",
                "repository",
            ],
        )
        
        return self._format_results(results)

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        language_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search using CodeBERT vector embeddings only.
        
        Args:
            query: Search query
            limit: Maximum results
            language_filter: Optional language filter (currently unused)
            
        Returns:
            List of results with metadata
        """
        chunks = self.client.collections.get("CodeChunk")
        
        # Embed query with CodeBERT
        query_embedding = self.embeddings.embed_text(query)
        
        # Semantic search using vector
        results = chunks.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_properties=[
                "file_path",
                "content",
                "language",
                "start_line",
                "end_line",
                "symbol_name",
                "symbol_type",
                "repository",
            ],
        )
        
        return self._format_results(results)

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        language_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Keyword (BM25) search on code content.
        
        Args:
            query: Search query
            limit: Maximum results
            language_filter: Optional language filter (currently unused)
            
        Returns:
            List of results with metadata
        """
        chunks = self.client.collections.get("CodeChunk")
        
        results = chunks.query.bm25(
            query=query,
            limit=limit,
            return_properties=[
                "file_path",
                "content",
                "language",
                "start_line",
                "end_line",
                "symbol_name",
                "symbol_type",
                "repository",
            ],
        )
        
        return self._format_results(results)

    def search_by_symbol(
        self,
        symbol_name: str,
        language_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for specific symbols by name.
        
        Args:
            symbol_name: Name of symbol to find
            language_filter: Optional language filter
            
        Returns:
            List of matching symbols
        """
        chunks = self.client.collections.get("CodeChunk")
        
        where = {
            "operator": "And",
            "operands": [
                {
                    "path": ["symbol_name"],
                    "operator": "Equal",
                    "valueText": symbol_name,
                },
            ]
        }
        
        if language_filter:
            where["operands"].append({
                "path": ["language"],
                "operator": "Equal",
                "valueText": language_filter,
            })
        
        results = chunks.query.fetch_objects(
            where=where,
            limit=10,
            return_properties=[
                "file_path",
                "content",
                "language",
                "start_line",
                "end_line",
                "symbol_name",
                "symbol_type",
                "repository",
            ],
        )
        
        return self._format_results(results)

    def search_by_file(self, file_path: str) -> list[dict]:
        """Get all chunks for a file."""
        chunks = self.client.collections.get("CodeChunk")
        
        results = chunks.query.fetch_objects(
            where={
                "path": ["file_path"],
                "operator": "Equal",
                "valueText": file_path,
            },
            limit=100,
            return_properties=[
                "file_path",
                "content",
                "language",
                "start_line",
                "end_line",
                "symbol_name",
                "symbol_type",
                "repository",
            ],
        )
        
        return self._format_results(results)

    def clear(self) -> None:
        """Delete all chunks from the store."""
        try:
            self.client.collections.delete("CodeChunk")
            self._ensure_schema()
        except Exception:
            pass

    @staticmethod
    def _format_results(weaviate_results) -> list[dict]:
        """Format Weaviate results."""
        formatted = []
        
        for result in weaviate_results.objects:
            formatted.append({
                "chunk_id": str(result.uuid),
                "file_path": result.properties.get("file_path", ""),
                "content": result.properties.get("content", ""),
                "language": result.properties.get("language", ""),
                "start_line": result.properties.get("start_line", 0),
                "end_line": result.properties.get("end_line", 0),
                "symbol_name": result.properties.get("symbol_name", ""),
                "symbol_type": result.properties.get("symbol_type", ""),
                "repository": result.properties.get("repository", ""),
                "score": result.metadata.score if result.metadata else 1.0,
            })
        
        return formatted

    def close(self) -> None:
        """Close the Weaviate connection."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
