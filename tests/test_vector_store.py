"""Tests for vector store."""

import pytest

from src.code_chat_agent.vector_store import VectorStore, VectorStoreEntry, Symbol


class TestVectorStore:
    """Test vector store functionality."""

    def test_add_entry(self) -> None:
        """Test adding entries to vector store."""
        store = VectorStore()
        
        symbol = Symbol(
            name="hello",
            type="function",
            file_path="main.py",
            line_number=1,
            language="python",
            content="def hello(): pass",
        )
        
        entry = VectorStoreEntry(
            chunk_id="test-1",
            embedding=[0.1, 0.2, 0.3],
            content="def hello(): pass",
            file_path="main.py",
            start_line=1,
            end_line=1,
            language="python",
            symbol=symbol,
        )
        
        store.add(entry)
        
        assert store.get_by_id("test-1") is not None
        assert "hello" in store.symbol_index

    def test_symbol_search(self) -> None:
        """Test symbol search."""
        store = VectorStore()
        
        symbol = Symbol(
            name="greet",
            type="function",
            file_path="utils.py",
            line_number=5,
            language="python",
            content="def greet(name): return f'Hello {name}'",
        )
        
        entry = VectorStoreEntry(
            chunk_id="test-2",
            embedding=[0.1, 0.2],
            content=symbol.content,
            file_path="utils.py",
            start_line=5,
            end_line=7,
            language="python",
            symbol=symbol,
        )
        
        store.add(entry)
        
        results = store.search_by_symbol("greet")
        assert len(results) == 1
        assert results[0].name == "greet"

    def test_file_search(self) -> None:
        """Test file-based search."""
        store = VectorStore()
        
        for i in range(3):
            entry = VectorStoreEntry(
                chunk_id=f"test-{i}",
                embedding=[float(i)] * 2,
                content=f"code {i}",
                file_path="test.py",
                start_line=i,
                end_line=i + 1,
                language="python",
            )
            store.add(entry)
        
        results = store.search_by_file("test.py")
        assert len(results) == 3

    def test_similarity_search(self) -> None:
        """Test similarity search."""
        store = VectorStore()
        
        # Add test entries with different embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]
        
        for i, emb in enumerate(embeddings):
            entry = VectorStoreEntry(
                chunk_id=f"test-{i}",
                embedding=emb,
                content=f"code {i}",
                file_path=f"file{i}.py",
                start_line=1,
                end_line=2,
                language="python",
            )
            store.add(entry)
        
        # Search with query similar to first embedding
        query = [0.95, 0.05, 0.0]
        results = store.similarity_search(query, top_k=2)
        
        assert len(results) == 2
        # First result should be most similar
        assert results[0][1] >= results[1][1]  # Similarity scores are sorted

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        sim_same = VectorStore._cosine_similarity(vec1, vec3)
        sim_orthogonal = VectorStore._cosine_similarity(vec1, vec2)
        
        assert abs(sim_same - 1.0) < 0.01  # Same vectors
        assert abs(sim_orthogonal) < 0.01  # Orthogonal vectors
