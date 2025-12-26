"""Tests for hybrid retriever."""

import pytest

from src.code_chat_agent.retriever import HybridRetriever
from src.code_chat_agent.vector_store import VectorStore, VectorStoreEntry, Symbol
from src.code_chat_agent.embeddings import EmbeddingsProvider


class TestHybridRetriever:
    """Test hybrid retriever functionality."""

    def test_keyword_extraction(self) -> None:
        """Test keyword extraction from query."""
        queries = [
            "show me OAuth code",
            "how to implement authentication",
            "find password hashing functions",
        ]
        
        for query in queries:
            keywords = HybridRetriever._extract_keywords(query)
            assert len(keywords) > 0
            assert all(len(kw) > 2 for kw in keywords)
            assert "show" not in keywords  # Stop word
            assert "the" not in keywords  # Stop word

    def test_keyword_matching(self) -> None:
        """Test keyword matching score calculation."""
        keywords = ["oauth", "authenticate"]
        
        high_score_text = "def oauth_authenticate(): pass"
        low_score_text = "def hash_password(): pass"
        
        high_score = HybridRetriever._keyword_match_score(keywords, high_score_text)
        low_score = HybridRetriever._keyword_match_score(keywords, low_score_text)
        
        assert high_score > low_score

    def test_hybrid_retrieval(self) -> None:
        """Test hybrid retrieval combining semantic and keyword search."""
        # Setup
        store = VectorStore()
        embeddings = EmbeddingsProvider(model="local")
        retriever = HybridRetriever(store, embeddings)
        
        # Add test code chunks
        test_chunks = [
            ("def oauth_login(token):\n    return verify_token(token)", "oauth login"),
            ("def basic_auth(user, pass):\n    return user == db[user].password", "basic auth"),
            ("def hash_password(pwd):\n    return bcrypt.hash(pwd)", "password hashing"),
        ]
        
        for content, description in test_chunks:
            embedding = embeddings.embed_text(content)
            entry = VectorStoreEntry(
                chunk_id=f"test-{description}",
                embedding=embedding,
                content=content,
                file_path=f"auth_{description.replace(' ', '_')}.py",
                start_line=1,
                end_line=2,
                language="python",
            )
            store.add(entry)
        
        # Search for OAuth
        results = retriever.retrieve("show me oauth", top_k=2)
        
        assert len(results) > 0
        # OAuth-related code should be first
        assert "oauth" in results[0].content.lower()
