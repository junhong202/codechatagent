"""MongoDB 8 vector store with hybrid search support.

This store mirrors the WeaviateStore interface to enable switching backends.

Requirements:
- MongoDB 8.0+ for `$vectorSearch` (local) OR Atlas Vector Search
- A vector index on the `embedding` field (created outside of this code)
- A text index on relevant text fields for keyword search
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging
import hashlib
import time

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from code_chat_agent.embeddings import EmbeddingsProvider

logger = logging.getLogger(__name__)


@dataclass
class MongoConfig:
    """Configuration for MongoDB connection."""

    uri: str = "mongodb://localhost:27017"
    database: str = "codechat"
    collection: str = "chunks"
    vector_index_name: str = "default"  # Name of the vector index for `$vectorSearch`
    use_atlas_search: bool = False  # If True, use `$search` (Atlas); else fallback to `$text`


class MongoDBStore:
    """MongoDB vector store for code indexing and retrieval with hybrid search."""

    def __init__(self, config: Optional[MongoConfig] = None) -> None:
        self.config = config or MongoConfig()
        logger.info(f"Connecting to MongoDB at {self.config.uri}")

        try:
            self.client = MongoClient(self.config.uri)
            self.db = self.client[self.config.database]
            self.collection: Collection = self.db[self.config.collection]
        except PyMongoError as e:
            raise ConnectionError(f"Cannot connect to MongoDB: {e}")

        # Initialize embeddings provider
        self.embeddings = EmbeddingsProvider()

        # Ensure indexes for keyword search; vector index must be created externally
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Ensure text indexes exist. Vector index is managed externally."""
        try:
            # Text index for keyword search
            self.collection.create_index(
                [
                    ("content", "text"),
                    ("symbol_name", "text"),
                    ("file_path", "text"),
                    ("repository", "text"),
                ],
                name="text_idx",
            )
            logger.info("✓ MongoDB text index ensured")
        except PyMongoError as e:
            logger.warning(f"Failed to create text index: {e}")

    def create_search_indexes(self, wait_until_ready: bool = True) -> None:
        """Create both text search and vector search indexes.
        
        Call this once after setting up your collection to enable hybrid search.
        
        Args:
            wait_until_ready: If True, blocks until indexes are ready for querying
        """
        # Create text search index for keyword search
        text_search_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": True,
                },
            },
            name="hybrid-text-search",
        )
        
        # Create vector search index for semantic search
        vector_search_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,  # OpenAI text-embedding-3-small
                        "similarity": "cosine",
                    }
                ]
            },
            name=self.config.vector_index_name,
            type="vectorSearch",
        )
        
        try:
            # Create both indexes
            text_result = self.collection.create_search_index(model=text_search_model)
            logger.info(f"Text search index '{text_result}' is building...")
            
            vector_result = self.collection.create_search_index(model=vector_search_model)
            logger.info(f"Vector search index '{vector_result}' is building...")
            
            if wait_until_ready:
                logger.info("Waiting for indexes to be ready (this may take up to a minute)...")
                self._wait_for_index(text_result)
                self._wait_for_index(vector_result)
                logger.info("✓ All search indexes are ready for querying")
        except PyMongoError as e:
            logger.error(f"Failed to create search indexes: {e}")
            raise
    
    def _wait_for_index(self, index_name: str, timeout: int = 60) -> None:
        """Wait for a search index to become queryable."""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Index {index_name} did not become ready within {timeout}s")
            
            indices = list(self.collection.list_search_indexes(index_name))
            if indices and indices[0].get("queryable") is True:
                logger.info(f"✓ Index '{index_name}' is ready")
                break
            
            time.sleep(5)

    @staticmethod
    def _make_chunk_id(file_path: str, start_line: int, end_line: int) -> str:
        """Deterministic chunk id from file path and line range."""
        raw = f"{file_path}:{start_line}:{end_line}".encode()
        return hashlib.sha1(raw).hexdigest()

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
        """Add a code chunk document with an embedding."""
        chunk_id = self._make_chunk_id(file_path, start_line, end_line)

        # Generate embedding with truncation to avoid token limits
        embedding = None
        for max_chars in (12000, 8000):
            text_for_embedding = content if len(content) <= max_chars else content[:max_chars]
            try:
                embedding = self.embeddings.embed_text(text_for_embedding)
                break
            except Exception as e:
                logger.warning(f"Embedding failed for {chunk_id} (max_chars={max_chars}): {e}")
                embedding = None

        if embedding is None:
            raise RuntimeError(f"Embedding failed for chunk {chunk_id}")

        # Upsert document
        # Note: renamed "language" to "prog_language" to avoid MongoDB text index language override conflict
        doc = {
            "_id": chunk_id,
            "file_path": file_path,
            "content": content,
            "prog_language": language,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name or "",
            "symbol_type": symbol_type or "",
            "repository": repository,
            "embedding": embedding,
        }

        try:
            self.collection.replace_one({"_id": chunk_id}, doc, upsert=True)
        except PyMongoError as e:
            raise RuntimeError(f"Failed to insert chunk {chunk_id}: {e}")

        return chunk_id

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        alpha: float = 0.5,
        language_filter: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search combining vector and keyword search.

        Tries MongoDB $rankFusion if available, falls back to manual merging.
        `alpha` balances vector (alpha) vs keyword (1-alpha).
        """
        query_embedding = self.embeddings.embed_text(query)
        try:
            return self._native_hybrid_search(query_embedding, query, limit, alpha, language_filter)
        except PyMongoError as e:
            if "rankFusion" in str(e) or "QueryFeatureNotAllowed" in str(e):
                logger.info("$rankFusion not available, using manual hybrid search")
                return self._manual_hybrid_search(query_embedding, query, limit, alpha, language_filter)
            raise

    def semantic_search(
        self, query: str, limit: int = 5, language_filter: Optional[str] = None
    ) -> list[dict]:
        """Semantic (vector-only) search."""
        query_embedding = self.embeddings.embed_text(query)
        results = self._vector_search(query_embedding, limit, language_filter)
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:limit]

    def keyword_search(
        self, query: str, limit: int = 5, language_filter: Optional[str] = None
    ) -> list[dict]:
        """Keyword-only search using Atlas `$search` or `$text` index."""
        results = self._keyword_search(query, limit, language_filter)
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:limit]

    def search_by_symbol(
        self, symbol_name: str, language_filter: Optional[str] = None
    ) -> list[dict]:
        """Search for specific symbols by name."""
        q: dict[str, Any] = {"symbol_name": symbol_name}
        if language_filter:
            q["language"] = language_filter

        docs = list(self.collection.find(q, limit=100))
        return [self._format_doc(d, score=1.0) for d in docs]

    def search_by_file(self, file_path: str) -> list[dict]:
        """Get all chunks for a file."""
        docs = list(self.collection.find({"file_path": file_path}))
        return [self._format_doc(d, score=1.0) for d in docs]

    def clear(self) -> None:
        """Delete all chunks from the store."""
        try:
            self.collection.drop()
            # Recreate text indexes
            self._ensure_indexes()
        except PyMongoError:
            pass

    def close(self) -> None:
        """Close MongoDB connection."""
        try:
            self.client.close()
        except Exception:
            pass

    # Internal helpers
    def _manual_hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        alpha: float,
        language_filter: Optional[str],
    ) -> list[dict]:
        """Manual hybrid search by merging vector and keyword results."""
        # Get more results than needed for better merging
        fetch_limit = limit * 3
        
        # Run both searches
        vector_results = self._vector_search(query_embedding, fetch_limit, language_filter)
        keyword_results = self._keyword_search(query_text, fetch_limit, language_filter)
        
        # Normalize scores to 0-1 range
        def normalize_scores(results: list[dict]) -> None:
            if not results:
                return
            max_score = max((r.get("score", 0.0) for r in results), default=1.0)
            if max_score > 0:
                for r in results:
                    r["score"] = r.get("score", 0.0) / max_score
        
        normalize_scores(vector_results)
        normalize_scores(keyword_results)
        
        # Merge by chunk_id with weighted scores
        merged: dict[str, dict] = {}
        
        for result in vector_results:
            chunk_id = result["chunk_id"]
            merged[chunk_id] = result.copy()
            merged[chunk_id]["score"] = alpha * result.get("score", 0.0)
        
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id in merged:
                merged[chunk_id]["score"] += (1.0 - alpha) * result.get("score", 0.0)
            else:
                merged[chunk_id] = result.copy()
                merged[chunk_id]["score"] = (1.0 - alpha) * result.get("score", 0.0)
        
        # Sort by combined score and return top results
        results = list(merged.values())
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:limit]
    
    def _native_hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int,
        alpha: float,
        language_filter: Optional[str],
    ) -> list[dict]:
        """MongoDB $rankFusion for automatic hybrid search with reranking."""
        pipeline: list[dict[str, Any]] = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorPipeline": [
                                {
                                    "$vectorSearch": {
                                        "index": self.config.vector_index_name,
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": max(limit * 20, 100),
                                        "limit": limit,
                                    }
                                }
                            ],
                            "fullTextPipeline": [
                                {
                                    "$search": {
                                        "index": "hybrid-text-search",
                                        "compound": {
                                            "should": [
                                                {
                                                    "text": {
                                                        "query": query_text,
                                                        "path": ["content", "symbol_name", "file_path"],
                                                    }
                                                }
                                            ]
                                        },
                                    }
                                },
                                {"$limit": limit},
                            ],
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorPipeline": alpha,
                            "fullTextPipeline": 1.0 - alpha,
                        }
                    },
                }
            },
            {
                "$project": {
                    "file_path": 1,
                    "content": 1,
                    "language": 1,
                    "start_line": 1,
                    "end_line": 1,
                    "symbol_name": 1,
                    "symbol_type": 1,
                    "repository": 1,
                    "score": {"$meta": "score"},
                }
            },
            {"$limit": limit},
        ]

        if language_filter:
            # Add language filter to both pipelines
            pipeline[0]["$rankFusion"]["input"]["pipelines"]["vectorPipeline"].insert(
                1, {"$match": {"language": language_filter}}
            )
            pipeline[0]["$rankFusion"]["input"]["pipelines"]["fullTextPipeline"].insert(
                1, {"$match": {"language": language_filter}}
            )

        docs = list(self.collection.aggregate(pipeline))
        return [self._format_doc(d, score=d.get("score", 0.0)) for d in docs]

    def _vector_search(
        self,
        query_embedding: list[float],
        limit: int,
        language_filter: Optional[str],
    ) -> list[dict]:
        """Run `$vectorSearch` aggregation. Requires a vector index on `embedding`."""
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.config.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(limit * 20, 100),
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "file_path": 1,
                    "content": 1,
                    "language": 1,
                    "start_line": 1,
                    "end_line": 1,
                    "symbol_name": 1,
                    "symbol_type": 1,
                    "repository": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        if language_filter:
            pipeline.insert(1, {"$match": {"language": language_filter}})

        try:
            docs = list(self.collection.aggregate(pipeline))
        except PyMongoError as e:
            logger.warning(f"Vector search failed (is vector index configured?): {e}")
            docs = []

        return [self._format_doc(d, score=d.get("score", 0.0)) for d in docs]

    def _keyword_search(
        self, query_text: str, limit: int, language_filter: Optional[str]
    ) -> list[dict]:
        """Keyword search via Atlas `$search` or `$text` lookup."""
        docs: list[dict] = []
        try:
            if self.config.use_atlas_search:
                pipeline: list[dict[str, Any]] = [
                    {
                        "$search": {
                            "compound": {
                                "should": [
                                    {"text": {"query": query_text, "path": ["content", "symbol_name", "file_path", "repository"]}},
                                ]
                            }
                        }
                    },
                    {"$set": {"score": {"$meta": "searchScore"}}},
                    {"$limit": limit},
                    {"$project": {
                        "file_path": 1,
                        "content": 1,
                        "language": 1,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol_name": 1,
                        "symbol_type": 1,
                        "repository": 1,
                        "score": 1,
                    }},
                ]
                if language_filter:
                    pipeline.insert(1, {"$match": {"language": language_filter}})
                docs = list(self.collection.aggregate(pipeline))
            else:
                # Fallback to classic text index
                query: dict[str, Any] = {"$text": {"$search": query_text}}
                projection = {"score": {"$meta": "textScore"}}
                if language_filter:
                    query["language"] = language_filter
                docs = list(self.collection.find(query, projection).sort("score", -1).limit(limit))
        except PyMongoError as e:
            logger.warning(f"Keyword search failed: {e}")
            docs = []

        # Normalize docs to a common shape
        return [self._format_doc(d, score=d.get("score", 0.0)) for d in docs]

    @staticmethod
    def _format_doc(doc: dict, score: float) -> dict:
        """Format a MongoDB doc to the common result structure."""
        return {
            "chunk_id": str(doc.get("_id", "")),
            "file_path": doc.get("file_path", ""),
            "content": doc.get("content", ""),
            "language": doc.get("language", ""),
            "start_line": int(doc.get("start_line", 0)),
            "end_line": int(doc.get("end_line", 0)),
            "symbol_name": doc.get("symbol_name") or "",
            "symbol_type": doc.get("symbol_type") or "",
            "repository": doc.get("repository") or "",
            "score": float(score or 0.0),
        }
