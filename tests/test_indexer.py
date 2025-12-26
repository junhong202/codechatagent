"""Tests for repository indexer."""

import pytest
from pathlib import Path

from src.code_chat_agent.indexer import RepositoryIndexer


class TestRepositoryIndexer:
    """Test repository indexing functionality."""

    def test_language_detection(self) -> None:
        """Test language detection from file extensions."""
        test_cases = [
            (Path("main.py"), "python"),
            (Path("app.js"), "javascript"),
            (Path("types.ts"), "typescript"),
            (Path("config.yaml"), "unknown"),
            (Path("script.sh"), "unknown"),
        ]
        
        for file_path, expected_lang in test_cases:
            assert RepositoryIndexer._detect_language(file_path) == expected_lang

    def test_ignored_files(self) -> None:
        """Test that ignored files are filtered."""
        ignored_cases = [
            Path(".git/config"),
            Path("__pycache__/file.pyc"),
            Path("node_modules/package/index.js"),
            Path(".venv/lib/python.py"),
        ]
        
        for file_path in ignored_cases:
            assert RepositoryIndexer._is_ignored(file_path)

    def test_non_ignored_files(self) -> None:
        """Test that non-ignored files pass through."""
        allowed_cases = [
            Path("src/main.py"),
            Path("tests/test_file.py"),
            Path("README.md"),
        ]
        
        for file_path in allowed_cases:
            assert not RepositoryIndexer._is_ignored(file_path)
