# Copilot Instructions for Python Project

This file contains custom instructions for GitHub Copilot to follow when working in this Python repository.

## Project Goal

Build a chat agent that can answer questions across many repositories and multiple programming languages without hallucinating. The agent should be able to handle queries like "show me OAuth code" by searching actual code in indexed repositories and grounding responses in real implementations rather than generating plausible-sounding but incorrect code.

Git Repos
  ↓
Repo Ingestion Pipeline
  ↓
Code Parsing & Chunking
  ↓
Embeddings + Metadata
  ↓
Vector DB + Symbol Index
  ↓
Retriever (Hybrid)
  ↓
LLM (Answer + Code)


## Python Style & Standards

- Follow [PEP 8](https://pep8.org/) guidelines and modern Python conventions (3.10+)
- Use type hints for all function arguments and return types (`def func(x: int) -> str:`)
- Use clear, descriptive variable and function names (snake_case for functions/variables)
- Write self-documenting code with docstrings (Google or NumPy style)
- Keep functions small and focused on a single responsibility

## Modern Python Tools

- **Package Management**: Use `uv` for fast, reliable dependency management
  - Define dependencies in `pyproject.toml`
  - Use `uv sync` to install dependencies
  - Use `uv run` to execute scripts
- **Linting**: Use `ruff` for code quality (replaces flake8, isort, etc.)
- **Formatting**: Use `ruff format` or `black` for consistent code formatting
- **Type Checking**: Use `mypy` for static type analysis
- **Testing**: Use `pytest` with meaningful test coverage
- **Virtual Environments**: Use `uv venv` for isolated environments

## Best Practices

- Prioritize readability and maintainability over clever solutions
- Add comprehensive error handling and input validation
- Include type hints for all public APIs
- Use dataclasses or Pydantic for data models
- Avoid mutable default arguments
- Use context managers (`with` statements) for resource management
- Follow functional programming patterns where applicable

## Documentation

- Update README.md when adding major features
- Include docstrings for all public functions and classes
- Provide usage examples for complex features
- Document environment setup and development workflow

## Git & Version Control

- Write clear, descriptive commit messages
- Reference issue numbers in commits when applicable
- Keep commits focused and logical
- Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`, etc.

## Performance & Security

- Avoid hardcoding secrets or sensitive information (use environment variables)
- Consider performance implications of suggested code
- Use security best practices: input validation, safe dependencies
- Run `uv pip check` to identify security issues
- Never use `eval()` or `exec()`

## Testing

- Write tests for all new functionality using `pytest`
- Use fixtures for common test setup
- Aim for meaningful test coverage (target 80%+)
- Write tests that verify behavior, not just line coverage
- Run tests locally with `uv run pytest` before committing

## Project Structure

```
project/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project documentation
├── src/                    # Source code
│   └── module/
│       ├── __init__.py
│       └── *.py
├── tests/                  # Test files (mirror src structure)
│   └── test_*.py
├── .github/                # GitHub configuration
│   └── copilot-instructions.md
└── .gitignore
```

---

**Note:** Customize the sections above to match your project's specific requirements.
