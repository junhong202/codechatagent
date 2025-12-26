"""Code parsing and chunking utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import ast
import re


@dataclass
class CodeChunk:
    """A semantically meaningful chunk of code."""

    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None  # "function", "class", "method", etc.


class CodeParser:
    """Parse code and extract semantic chunks."""

    def __init__(self) -> None:
        """Initialize code parser."""
        self.parsers: dict[str, callable] = {
            "python": self._parse_python,
            "javascript": self._parse_javascript,
            "typescript": self._parse_javascript,
            "default": self._parse_generic,
        }

    def parse_file(self, file_path: str | Path) -> list[CodeChunk]:
        """Parse a code file and extract chunks."""
        file_path = Path(file_path)
        
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            return []

        language = self._detect_language(file_path)
        parser = self.parsers.get(language, self.parsers["default"])
        
        return parser(file_path, content, language)

    def _parse_python(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Parse Python files using AST."""
        chunks = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to generic chunking
            return self._parse_generic(file_path, content, language)

        lines = content.split("\n")
        
        # Extract top-level definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not self._is_nested(node, tree):
                    chunk = CodeChunk(
                        file_path=str(file_path),
                        language=language,
                        content=self._extract_node_content(node, lines),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        symbol_name=node.name,
                        symbol_type="class" if isinstance(node, ast.ClassDef) else "function",
                    )
                    chunks.append(chunk)
        
        return chunks or self._parse_generic(file_path, content, language)

    def _parse_javascript(
        self, file_path: Path, content: str, language: str
    ) -> list[CodeChunk]:
        """Parse JavaScript/TypeScript using regex patterns."""
        chunks = []
        lines = content.split("\n")
        
        # Pattern for function declarations
        func_pattern = r"(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*(?:=|\()"
        # Pattern for class declarations
        class_pattern = r"class\s+(\w+)"
        
        for match in re.finditer(func_pattern, content):
            symbol_name = match.group(1)
            start_line = content[:match.start()].count("\n") + 1
            chunks.append(CodeChunk(
                file_path=str(file_path),
                language=language,
                content=match.group(0),
                start_line=start_line,
                end_line=start_line,
                symbol_name=symbol_name,
                symbol_type="function",
            ))
        
        return chunks or self._parse_generic(file_path, content, language)

    def _parse_generic(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Generic chunking for unsupported languages."""
        chunks = []
        lines = content.split("\n")
        
        # Split by empty lines (paragraphs)
        current_chunk = []
        start_line = 1
        
        for i, line in enumerate(lines):
            if line.strip():
                current_chunk.append(line)
            else:
                if current_chunk and len("\n".join(current_chunk)) > 50:
                    chunk = CodeChunk(
                        file_path=str(file_path),
                        language=language,
                        content="\n".join(current_chunk),
                        start_line=start_line,
                        end_line=i,
                    )
                    chunks.append(chunk)
                current_chunk = []
                start_line = i + 1
        
        # Add final chunk
        if current_chunk and len("\n".join(current_chunk)) > 50:
            chunks.append(CodeChunk(
                file_path=str(file_path),
                language=language,
                content="\n".join(current_chunk),
                start_line=start_line,
                end_line=len(lines),
            ))
        
        return chunks

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        """Detect language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }
        return extension_map.get(file_path.suffix, "default")

    @staticmethod
    def _extract_node_content(node: ast.AST, lines: list[str]) -> str:
        """Extract source code for an AST node."""
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno)
        return "\n".join(lines[start:end])

    @staticmethod
    def _is_nested(node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is nested inside another function/class."""
        for parent in ast.walk(tree):
            if parent is node:
                continue
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for child in ast.walk(parent):
                    if child is node:
                        return True
        return False
