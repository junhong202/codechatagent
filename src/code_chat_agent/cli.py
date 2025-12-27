"""CLI interface for Code Chat Agent with selectable backend.

MongoDB is the default local backend; no Mongo-specific CLI flags are required.
"""

from pathlib import Path
from typing import Optional
import logging
import warnings

import click

from code_chat_agent.agent import CodeChatAgent

# Suppress ResourceWarnings from gRPC connections
warnings.filterwarnings("ignore", category=ResourceWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.option("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
@click.option("--backend", default="mongodb", type=click.Choice(["mongodb", "weaviate"]), help="Vector backend")
@click.option("--openai-model", default=None, help="OpenAI embedding model to use (overrides OPENAI_MODEL env)")
@click.pass_context
def cli(
    ctx,
    weaviate_url: str,
    backend: str,
    openai_model: Optional[str] = None,
) -> None:
    """Code Chat Agent - Search and answer questions across code repositories."""
    ctx.ensure_object(dict)
    ctx.obj["weaviate_url"] = weaviate_url
    ctx.obj["backend"] = backend
    ctx.obj["openai_model"] = openai_model


@cli.command()
@click.argument("repo_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.pass_context
def index(ctx, repo_paths: tuple[str, ...]) -> None:
    """Index one or more repositories."""
    click.echo(f"Indexing {len(repo_paths)} repositories...")
    if ctx.obj["backend"] == "weaviate":
        click.echo(f"Connecting to Weaviate at {ctx.obj['weaviate_url']}...\n")
    else:
        click.echo(f"Connecting to MongoDB at mongodb://localhost:27017...\n")
    
    try:
        with CodeChatAgent(
            weaviate_url=ctx.obj["weaviate_url"],
            openai_model=ctx.obj.get("openai_model"),
            backend=ctx.obj["backend"],
        ) as agent:
            agent.index_repositories([Path(p) for p in repo_paths])
            
            click.echo(f"✓ Successfully indexed all repositories")
            click.echo(f"  Data stored in Weaviate")
    except Exception as e:
        click.echo(f"✗ Error indexing repositories: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--language", default=None, help="Filter by language")
@click.option("--type", "search_type", default="hybrid", 
              type=click.Choice(["hybrid", "semantic", "keyword"]),
              help="Search type")
@click.pass_context
def search(ctx, query: str, top_k: int, language: str, search_type: str) -> None:
    """Search for code matching a query."""
    click.echo(f"Searching for: {query}")
    click.echo(f"Search type: {search_type}\n")
    
    try:
        with CodeChatAgent(
            weaviate_url=ctx.obj["weaviate_url"],
            openai_model=ctx.obj.get("openai_model"),
            backend=ctx.obj["backend"],
        ) as agent:
            results = agent.search_code(
                query,
                top_k=top_k,
                language_filter=language,
                search_type=search_type,
            )
            
            if not results:
                click.echo("No results found.")
                return
            
            for i, result in enumerate(results, 1):
                click.echo(f"--- Result {i} ---")
                click.echo(f"Repository: {result.repository}")
                click.echo(f"File: {result.file_path}")
                click.echo(f"Lines: {int(result.start_line)}-{int(result.end_line)}")
                click.echo(f"Language: {result.language}")
                if result.symbol_name:
                    click.echo(f"Symbol: {result.symbol_name} ({result.symbol_type})")
                if result.score:
                    click.echo(f"Score: {result.score:.4f}")
                click.echo("")
                click.echo(result.content[:500])
                if len(result.content) > 500:
                    click.echo("...")
                click.echo("\n")
    except Exception as e:
        click.echo(f"✗ Error searching: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, help="Number of code results to use for context")
@click.option("--use-llm", is_flag=True,default=True, help="Use OpenAI API for answer generation")
@click.option("--language", default=None, help="Filter by language")
@click.pass_context
def ask(ctx, question: str, top_k: int, use_llm: bool, language: str) -> None:
    """Ask a question about the code."""
    click.echo(f"Question: {question}\n")
    
    try:
        with CodeChatAgent(
            weaviate_url=ctx.obj["weaviate_url"],
            openai_model=ctx.obj.get("openai_model"),
            backend=ctx.obj["backend"],
        ) as agent:
            # Search for relevant code
            click.echo("Searching for relevant code...\n")
            search_results = agent.search_code(
                question,
                top_k=top_k,
                language_filter=language,
                search_type="hybrid",
            )
            
            if not search_results:
                click.echo("No relevant code found to answer this question.")
                return
            
            # Get answer
            if use_llm:
                click.echo("Generating answer with LLM...\n")
            
            answer = agent.answer_question(question, search_results)
            
            click.echo(answer)
            click.echo("\n--- Code Context ---")
            for i, result in enumerate(search_results, 1):
                click.echo(f"\n[{i}] {result.repository}/{result.file_path}:{int(result.start_line)}")
                click.echo(result.content[:300])
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.pass_context
def clear(ctx) -> None:
    """Clear all indexed data from the configured backend."""
    if not click.confirm("Are you sure you want to delete all indexed data?"):
        return
    
    try:
        with CodeChatAgent(
            weaviate_url=ctx.obj["weaviate_url"],
            openai_model=ctx.obj.get("openai_model"),
            backend=ctx.obj["backend"],
        ) as agent:
            agent.vector_store.clear()
            click.echo("✓ Cleared all indexed data")
    except Exception as e:
        click.echo(f"✗ Error clearing data: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
