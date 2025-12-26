"""Example usage of Code Chat Agent with Weaviate."""

from pathlib import Path
from src.code_chat_agent.agent import CodeChatAgent

# Example 1: Basic hybrid search
def example_hybrid_search() -> None:
    """Hybrid search example (semantic + keyword)."""
    print("=== Hybrid Search Example ===\n")
    
    agent = CodeChatAgent()
    
    try:
        # Index repositories
        agent.index_repositories([
            Path("/path/to/oauth-lib"),
            Path("/path/to/auth-service"),
        ])
        
        # Hybrid search - combines semantic embeddings with BM25
        results = agent.search_code(
            "OAuth 2.0 token verification",
            top_k=5,
            search_type="hybrid"
        )
        
        for result in results:
            print(f"\n{result.repository}/{result.file_path}:{result.start_line}")
            print(f"Symbol: {result.symbol_name or 'N/A'}")
            print(f"Score: {result.score:.4f}")
            print(result.content[:200])
        
        agent.close()
    except Exception as e:
        print(f"Error: {e}")


# Example 2: Semantic-only search
def example_semantic_search() -> None:
    """Semantic search using vector embeddings only."""
    print("=== Semantic Search Example ===\n")
    
    agent = CodeChatAgent()
    
    try:
        agent.index_repositories([Path("/path/to/repos")])
        
        # Semantic search - finds conceptually similar code
        results = agent.search_code(
            "authentication patterns",
            search_type="semantic"
        )
        
        print(f"Found {len(results)} semantically similar code chunks")
        
        agent.close()
    except Exception as e:
        print(f"Error: {e}")


# Example 3: Keyword-only search
def example_keyword_search() -> None:
    """Keyword search for specific terms and syntax."""
    print("=== Keyword Search Example ===\n")
    
    agent = CodeChatAgent()
    
    try:
        agent.index_repositories([Path("/path/to/repos")])
        
        # Keyword search - exact term matching with BM25
        results = agent.search_code(
            "def login verify_token",
            search_type="keyword"
        )
        
        print(f"Found {len(results)} keyword matches")
        
        agent.close()
    except Exception as e:
        print(f"Error: {e}")


# Example 4: Ask question with LLM
def example_ask_with_llm() -> None:
    """Ask a question with LLM integration."""
    print("=== Ask Question Example ===\n")
    
    agent = CodeChatAgent()
    
    try:
        agent.index_repositories([Path("/path/to/repos")])
        
        question = "How do I implement OAuth 2.0 with refresh tokens?"
        
        # Search for relevant code
        search_results = agent.search_code(question, search_type="hybrid")
        
        if search_results:
            # Get LLM-powered answer
            answer = agent.answer_question(question, search_results)
            print(f"Q: {question}")
            print(f"\nA: {answer}")
            
            print("\n--- Code Context ---")
            for i, result in enumerate(search_results[:3], 1):
                print(f"\n[{i}] {result.file_path}:{result.start_line}")
                print(result.content[:200])
        
        agent.close()
    except Exception as e:
        print(f"Error: {e}")


# Example 5: Language-filtered search
def example_filtered_search() -> None:
    """Search with language filter."""
    print("=== Language-Filtered Search Example ===\n")
    
    agent = CodeChatAgent()
    
    try:
        agent.index_repositories([Path("/path/to/repos")])
        
        # Search only in Python files
        python_results = agent.search_code(
            "async function",
            language_filter="python",
            search_type="hybrid"
        )
        
        print(f"Found {len(python_results)} Python results")
        
        # Search only in TypeScript files
        ts_results = agent.search_code(
            "async function",
            language_filter="typescript",
            search_type="hybrid"
        )
        
        print(f"Found {len(ts_results)} TypeScript results")
        
        agent.close()
    except Exception as e:
        print(f"Error: {e}")


# Example 6: Direct Weaviate store operations
def example_weaviate_operations() -> None:
    """Direct Weaviate store operations."""
    print("=== Weaviate Store Operations ===\n")
    
    from src.code_chat_agent.weaviate_store import WeaviateStore
    
    try:
        store = WeaviateStore()
        
        # Add a code chunk
        chunk_id = store.add_chunk(
            file_path="auth/oauth.py",
            content="def verify_oauth_token(token: str) -> bool:\n    return token in valid_tokens",
            language="python",
            start_line=10,
            end_line=12,
            symbol_name="verify_oauth_token",
            symbol_type="function",
            repository="auth-lib"
        )
        print(f"Added chunk: {chunk_id}")
        
        # Search by symbol
        results = store.search_by_symbol("verify_oauth_token")
        print(f"Found {len(results)} symbols named 'verify_oauth_token'")
        
        # Search by file
        results = store.search_by_file("auth/oauth.py")
        print(f"Found {len(results)} chunks in 'auth/oauth.py'")
        
        store.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Uncomment examples to run
    # example_hybrid_search()
    # example_semantic_search()
    # example_keyword_search()
    # example_ask_with_llm()
    # example_filtered_search()
    # example_weaviate_operations()
    
    print("Uncomment examples in main() to run them")
    print("\nMake sure Weaviate is running: docker-compose up -d")
