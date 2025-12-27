"""Tests for MongoDB store (skips if MongoDB not available)."""

import os
import pytest

from code_chat_agent.mongo_store import MongoDBStore, MongoConfig


@pytest.fixture
def mongo_store():
    """Create a MongoDB store for testing, skip if connection fails."""
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    cfg = MongoConfig(uri=uri, database="codechat_test", collection="chunks_test", use_atlas_search=False)
    try:
        store = MongoDBStore(cfg)
        store.clear()
        yield store
        store.clear()
        store.close()
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")


class TestMongoStore:
    def test_add_chunk(self, mongo_store) -> None:
        chunk_id = mongo_store.add_chunk(
            file_path="auth.py",
            content="def oauth_login(token):\n    return verify_token(token)",
            language="python",
            start_line=10,
            end_line=12,
            symbol_name="oauth_login",
            symbol_type="function",
            repository="auth-lib",
        )
        assert chunk_id

    def test_keyword_search(self, mongo_store) -> None:
        mongo_store.add_chunk(
            file_path="test.py",
            content="def test_oauth(): pass",
            language="python",
            start_line=1,
            end_line=1,
            repository="test-repo",
        )
        results = mongo_store.keyword_search("oauth", limit=5)
        assert isinstance(results, list)

    def test_search_by_file(self, mongo_store) -> None:
        mongo_store.add_chunk(
            file_path="main.py",
            content="code chunk 1",
            language="python",
            start_line=1,
            end_line=5,
            repository="main",
        )
        mongo_store.add_chunk(
            file_path="main.py",
            content="code chunk 2",
            language="python",
            start_line=10,
            end_line=15,
            repository="main",
        )
        results = mongo_store.search_by_file("main.py")
        assert len(results) == 2
