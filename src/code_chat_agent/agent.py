"""Updated core chat agent logic with Weaviate backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import logging

from src.code_chat_agent.indexer import RepositoryIndexer
from src.code_chat_agent.parser import CodeParser
from src.code_chat_agent.weaviate_store import WeaviateStore, WeaviateConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """Result from a code search query."""

    chunk_id: str
    repository: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    score: float = 1.0


class CodeChatAgent:
    """Chat agent for answering questions about code across repositories."""

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        openai_model: Optional[str] = None,
    ) -> None:
        """Initialize the chat agent with Weaviate backend."""
        self.indexer = RepositoryIndexer([])
        self.parser = CodeParser()
        
        # Initialize Weaviate store
        config = WeaviateConfig(url=weaviate_url)
        self.vector_store = WeaviateStore(config)
        # OpenAI model selection: explicit param -> env var -> default
        self.openai_model = (
            openai_model
            or os.getenv("OPENAI_API_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-3.5-turbo"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def index_repositories(self, repo_paths: list[str | Path]) -> None:
        """Index multiple repositories in Weaviate."""
        self.indexer = RepositoryIndexer(repo_paths)
        logger.info(f"Indexing {len(repo_paths)} repositories...")
        indexed = self.indexer.index_repositories()
        
        total_chunks = 0
        # Parse all files and add to Weaviate
        for repo_name, repo_data in indexed.items():
            repo_path = Path(repo_data["path"])
            logger.info(f"Processing repository: {repo_name}")
            
            for file_info in repo_data.get("files", []):
                file_path = repo_path / file_info["path"]

                # Log which file we're indexing to show progress
                logger.info(f"Indexing file: {file_path}")

                # Parse file into chunks
                chunks = self.parser.parse_file(file_path)
                logger.info(f"  → Parsed {len(chunks)} chunks from {file_path.name}")
                total_chunks += len(chunks)

                for chunk in chunks:
                    # Add to Weaviate
                    self.vector_store.add_chunk(
                        file_path=chunk.file_path,
                        content=chunk.content,
                        language=chunk.language,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        symbol_name=chunk.symbol_name,
                        symbol_type=chunk.symbol_type,
                        repository=repo_name,
                    )
        
        logger.info(f"✓ Indexed {total_chunks} code chunks")

    def search_code(
        self,
        query: str,
        top_k: int = 5,
        language_filter: Optional[str] = None,
        search_type: str = "hybrid",
    ) -> list[CodeSearchResult]:
        """
        Search for code matching the query.

        Args:
            query: Search query (e.g., "OAuth implementation")
            top_k: Number of results to return
            language_filter: Optional language filter
            search_type: "hybrid" (semantic + keyword), "semantic", or "keyword"

        Returns:
            List of matching code results
        """
        logger.info(f"Searching: {query} (type={search_type}, top_k={top_k})")
        
        if search_type == "hybrid":
            results = self.vector_store.hybrid_search(
                query, limit=top_k, alpha=0.5, language_filter=language_filter
            )
        elif search_type == "semantic":
            results = self.vector_store.semantic_search(
                query, limit=top_k, language_filter=language_filter
            )
        elif search_type == "keyword":
            results = self.vector_store.keyword_search(
                query, limit=top_k, language_filter=language_filter
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        logger.info(f"✓ Found {len(results)} results")
        return [self._format_result(r) for r in results]

    def answer_question(self, question: str, search_results: list[CodeSearchResult]) -> str:
        """
        Answer a question based on search results.

        Args:
            question: User question
            search_results: Code search results to base answer on

        Returns:
            Answer grounded in actual code
        """
        if not search_results:
            return "No relevant code found to answer this question."
        
        # Build context from search results
        context = self._build_context(search_results)
        
        try:
            return self._answer_with_llm(question, context)
        except Exception:
            # Fallback to simple context-based answer
            return self._answer_without_llm(question, context)

    def _build_context(self, results: list[CodeSearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"--- Result {i} ---")
            context_parts.append(f"Repository: {result.repository}")
            context_parts.append(f"File: {result.file_path}")
            context_parts.append(f"Lines: {int(result.start_line)}-{int(result.end_line)}")
            context_parts.append(f"Language: {result.language}")
            if result.symbol_name:
                context_parts.append(f"Symbol: {result.symbol_name} ({result.symbol_type})")
            if result.score:
                context_parts.append(f"Score: {result.score:.2f}")
            context_parts.append("")
            context_parts.append(result.content)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _answer_with_llm(self, question: str, context: str) -> str:
        """Answer question using LLM."""
        try:
            from openai import OpenAI
            
            client = OpenAI()
            
            system_prompt = (
                "You are a code assistant that answers questions based on actual code. "
                "Always ground your answers in the provided code examples. "
                "Never hallucinate or suggest code that isn't shown in the context."
            )
            
            user_message = f"Question: {question}\n\nCode Context:\n{context}"
            
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,  # Lower temperature for more factual responses
            )
            
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai package required for LLM integration")

    @staticmethod
    def _answer_without_llm(question: str, context: str) -> str:
        """Answer question without LLM (fallback)."""
        return f"Based on the code found:\n\n{context}\n\nThis code appears relevant to your question about: {question}"

    @staticmethod
    def _format_result(weaviate_result: dict) -> CodeSearchResult:
        """Format Weaviate result to CodeSearchResult."""
        return CodeSearchResult(
            chunk_id=weaviate_result["chunk_id"],
            repository=weaviate_result.get("repository", "unknown"),
            file_path=weaviate_result["file_path"],
            language=weaviate_result["language"],
            content=weaviate_result["content"],
            start_line=weaviate_result["start_line"],
            end_line=weaviate_result["end_line"],
            symbol_name=weaviate_result.get("symbol_name"),
            symbol_type=weaviate_result.get("symbol_type"),
            score=weaviate_result.get("score", 1.0),
        )

    def close(self) -> None:
        """Close connection to Weaviate."""
        if self.vector_store:
            self.vector_store.close()
