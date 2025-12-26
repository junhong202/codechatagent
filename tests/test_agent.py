"""Tests for code chat agent."""

import pytest

from src.code_chat_agent.agent import CodeChatAgent


class TestCodeChatAgent:
    """Test the main chat agent."""

    def test_agent_initialization(self) -> None:
        """Test agent initialization."""
        agent = CodeChatAgent(embedding_model="local")
        
        assert agent.parser is not None
        assert agent.embeddings is not None
        assert agent.vector_store is not None
        assert agent.retriever is not None

    def test_build_context(self) -> None:
        """Test context building from search results."""
        from src.code_chat_agent.retriever import RetrievalResult
        
        agent = CodeChatAgent()
        
        results = [
            RetrievalResult(
                chunk_id="test-1",
                file_path="auth.py",
                start_line=10,
                end_line=20,
                language="python",
                content="def oauth_login(): pass",
                symbol_name="oauth_login",
                relevance_score=0.95,
            ),
            RetrievalResult(
                chunk_id="test-2",
                file_path="utils.py",
                start_line=5,
                end_line=8,
                language="python",
                content="def verify_token(): pass",
                symbol_name="verify_token",
                relevance_score=0.87,
            ),
        ]
        
        context = agent._build_context(results)
        
        assert "auth.py" in context
        assert "oauth_login" in context
        assert "0.95" in context
        assert "utils.py" in context
        assert "verify_token" in context

    def test_answer_without_llm(self) -> None:
        """Test fallback answer without LLM."""
        agent = CodeChatAgent()
        
        question = "How to implement OAuth?"
        context = "def oauth_login(): pass"
        
        answer = agent._answer_without_llm(question, context)
        
        assert question in answer
        assert context in answer
