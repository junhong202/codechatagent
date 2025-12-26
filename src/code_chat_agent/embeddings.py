"""Embeddings generation using OpenAI API."""

from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a chunk."""

    chunk_id: str
    embedding: list[float]
    metadata: dict


class EmbeddingsProvider:
    """Generate embeddings for code using OpenAI API."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize embeddings provider with OpenAI."""
        try:
            from openai import OpenAI
            
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not provided. Set it as parameter or environment variable."
                )
            
            self.client = OpenAI(api_key=self.api_key)
            logger.info("✓ OpenAI embeddings provider initialized")
        except ImportError:
            raise ImportError("openai package required for embeddings. Install with: pip install openai")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for code text using OpenAI."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        # Sort by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

        hash_int = int(hash_obj.hexdigest(), 16)
        vector = []
        for i in range(384):  # Common embedding dimension
            vector.append(float((hash_int >> i) & 1) * 2 - 1)
        return vector

    @staticmethod
    def generate_chunk_id(file_path: str, start_line: int, end_line: int) -> str:
        """Generate unique ID for a chunk."""
        content = f"{file_path}:{start_line}:{end_line}"
        return hashlib.md5(content.encode()).hexdigest()
