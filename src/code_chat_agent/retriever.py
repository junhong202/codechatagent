"""Hybrid retriever for code search."""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    content: str
    symbol_name: Optional[str] = None
    relevance_score: float = 0.0


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""

    def __init__(self, vector_store, embeddings_provider) -> None:
        """Initialize retriever."""
        self.vector_store = vector_store
        self.embeddings = embeddings_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[RetrievalResult]:
        """Retrieve relevant code chunks using hybrid search."""
        
        # Semantic search
        semantic_results = self._semantic_search(query, top_k)
        
        # Keyword search
        keyword_results = self._keyword_search(query, top_k)
        
        # Combine and rank results
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight,
        )
        
        return combined[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search using semantic embeddings."""
        try:
            query_embedding = self.embeddings.embed_text(query)
        except Exception:
            # Fallback to keyword search if embedding fails
            return []
        
        results = []
        for entry, similarity in self.vector_store.similarity_search(query_embedding, top_k):
            results.append(RetrievalResult(
                chunk_id=entry.chunk_id,
                file_path=entry.file_path,
                start_line=entry.start_line,
                end_line=entry.end_line,
                language=entry.language,
                content=entry.content,
                symbol_name=entry.symbol.name if entry.symbol else None,
                relevance_score=similarity,
            ))
        
        return results

    def _keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search using keyword matching."""
        results = []
        keywords = self._extract_keywords(query)
        
        for entry in self.vector_store.entries.values():
            score = self._keyword_match_score(keywords, entry.content)
            if score > 0:
                results.append(RetrievalResult(
                    chunk_id=entry.chunk_id,
                    file_path=entry.file_path,
                    start_line=entry.start_line,
                    end_line=entry.end_line,
                    language=entry.language,
                    content=entry.content,
                    symbol_name=entry.symbol.name if entry.symbol else None,
                    relevance_score=score,
                ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def _combine_results(
        self,
        semantic: list[RetrievalResult],
        keyword: list[RetrievalResult],
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[RetrievalResult]:
        """Combine semantic and keyword results."""
        # Create dict to merge results
        merged = {}
        
        # Add semantic results
        for result in semantic:
            merged[result.chunk_id] = result
            merged[result.chunk_id].relevance_score = result.relevance_score * semantic_weight
        
        # Add/merge keyword results
        for result in keyword:
            if result.chunk_id in merged:
                # Combine scores
                merged[result.chunk_id].relevance_score += (
                    result.relevance_score * keyword_weight
                )
            else:
                merged[result.chunk_id] = result
                merged[result.chunk_id].relevance_score = result.relevance_score * keyword_weight
        
        # Sort and return
        results = list(merged.values())
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Extract keywords from query."""
        # Remove common words and split
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "is", "show", "me"}
        keywords = [
            w.lower() for w in re.findall(r"\w+", query)
            if w.lower() not in stop_words and len(w) > 2
        ]
        return keywords

    @staticmethod
    def _keyword_match_score(keywords: list[str], text: str) -> float:
        """Score text based on keyword matches."""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords) if keywords else 0.0
