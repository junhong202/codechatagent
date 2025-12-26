"""Tests for code parser and chunking."""

import tempfile
from pathlib import Path

import pytest

from src.code_chat_agent.parser import CodeParser, CodeChunk


class TestCodeParser:
    """Test code parsing and chunking."""

    def test_detect_language(self) -> None:
        """Test language detection from file extensions."""
        test_cases = [
            (Path("main.py"), "python"),
            (Path("app.js"), "javascript"),
            (Path("script.ts"), "typescript"),
            (Path("config.yaml"), "default"),
        ]
        
        parser = CodeParser()
        for file_path, expected_lang in test_cases:
            assert parser._detect_language(file_path) == expected_lang

    def test_parse_python_file(self) -> None:
        """Test parsing a Python file."""
        python_code = """
def hello(name: str) -> str:
    return f"Hello, {name}"

class MyClass:
    def method(self) -> None:
        pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            f.flush()
            
            parser = CodeParser()
            chunks = parser.parse_file(f.name)
            
            assert len(chunks) > 0
            assert any(c.symbol_type == "function" for c in chunks)
            assert any(c.symbol_type == "class" for c in chunks)
            
            Path(f.name).unlink()

    def test_parse_javascript_file(self) -> None:
        """Test parsing a JavaScript file."""
        js_code = """
function greet(name) {
    return `Hello, ${name}`;
}

const arrow = (x) => x * 2;
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            f.flush()
            
            parser = CodeParser()
            chunks = parser.parse_file(f.name)
            
            assert len(chunks) > 0
            
            Path(f.name).unlink()
