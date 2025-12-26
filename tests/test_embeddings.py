"""Tests for embeddings generation."""

import pytest

from src.code_chat_agent.embeddings import EmbeddingsProvider


class TestEmbeddingsProvider:
    """Test embeddings generation."""

    def test_local_embeddings(self) -> None:
        """Test local embeddings generation."""
        provider = EmbeddingsProvider(model="local")
        
        text1 = "def hello(): pass"
        text2 = "def hello(): pass"
        text3 = "class MyClass: pass"
        
        embedding1 = provider.embed_text(text1)
        embedding2 = provider.embed_text(text2)
        embedding3 = provider.embed_text(text3)
        
        assert len(embedding1) == 384
        assert len(embedding2) == 384
        assert len(embedding3) == 384
        
        # Same text should produce same embedding
        assert embedding1 == embedding2
        
        # Different text should produce different embedding
        assert embedding1 != embedding3

    def test_chunk_id_generation(self) -> None:
        """Test chunk ID generation."""
        id1 = EmbeddingsProvider.generate_chunk_id("file.py", 1, 10)
        id2 = EmbeddingsProvider.generate_chunk_id("file.py", 1, 10)
        id3 = EmbeddingsProvider.generate_chunk_id("file.py", 1, 20)
        
        assert id1 == id2  # Same parameters produce same ID
        assert id1 != id3  # Different parameters produce different IDs
