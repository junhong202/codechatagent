"""Tests for Weaviate store."""

import pytest

from src.code_chat_agent.weaviate_store import WeaviateStore, WeaviateConfig


@pytest.fixture
def weaviate_store():
    """Create a Weaviate store for testing."""
    try:
        store = WeaviateStore()
        store.clear()  # Clear before test
        yield store
        store.clear()  # Clear after test
        store.close()
    except Exception as e:
        pytest.skip(f"Weaviate not available: {e}")


class TestWeaviateStore:
    """Test Weaviate store functionality."""

    def test_add_chunk(self, weaviate_store) -> None:
        """Test adding code chunks."""
        chunk_id = weaviate_store.add_chunk(
            file_path="auth.py",
            content="def oauth_login(token):\n    return verify_token(token)",
            language="python",
            start_line=10,
            end_line=12,
            symbol_name="oauth_login",
            symbol_type="function",
            repository="auth-lib",
        )
        
        assert chunk_id is not None

    def test_hybrid_search(self, weaviate_store) -> None:
        """Test hybrid search."""
        # Add test chunks
        weaviate_store.add_chunk(
            file_path="auth.py",
            content="def oauth_login(token): return verify_token(token)",
            language="python",
            start_line=1,
            end_line=1,
            symbol_name="oauth_login",
            symbol_type="function",
            repository="auth-lib",
        )
        
        weaviate_store.add_chunk(
            file_path="utils.py",
            content="def hash_password(pwd): return bcrypt.hash(pwd)",
            language="python",
            start_line=5,
            end_line=5,
            symbol_name="hash_password",
            symbol_type="function",
            repository="auth-lib",
        )
        
        # Search
        results = weaviate_store.hybrid_search("oauth", limit=5)
        assert len(results) > 0

    def test_semantic_search(self, weaviate_store) -> None:
        """Test semantic search."""
        weaviate_store.add_chunk(
            file_path="auth.py",
            content="OAuth 2.0 authentication implementation",
            language="python",
            start_line=1,
            end_line=1,
            repository="auth-lib",
        )
        
        results = weaviate_store.semantic_search("authentication", limit=5)
        assert isinstance(results, list)

    def test_keyword_search(self, weaviate_store) -> None:
        """Test keyword (BM25) search."""
        weaviate_store.add_chunk(
            file_path="test.py",
            content="def test_oauth(): pass",
            language="python",
            start_line=1,
            end_line=1,
            repository="test-repo",
        )
        
        results = weaviate_store.keyword_search("oauth", limit=5)
        assert isinstance(results, list)

    def test_search_by_symbol(self, weaviate_store) -> None:
        """Test symbol search."""
        weaviate_store.add_chunk(
            file_path="utils.py",
            content="def helper_func(): pass",
            language="python",
            start_line=1,
            end_line=1,
            symbol_name="helper_func",
            symbol_type="function",
            repository="utils",
        )
        
        results = weaviate_store.search_by_symbol("helper_func")
        assert isinstance(results, list)

    def test_search_by_file(self, weaviate_store) -> None:
        """Test file-based search."""
        weaviate_store.add_chunk(
            file_path="main.py",
            content="code chunk 1",
            language="python",
            start_line=1,
            end_line=5,
            repository="main",
        )
        
        weaviate_store.add_chunk(
            file_path="main.py",
            content="code chunk 2",
            language="python",
            start_line=10,
            end_line=15,
            repository="main",
        )
        
        results = weaviate_store.search_by_file("main.py")
        assert len(results) == 2
