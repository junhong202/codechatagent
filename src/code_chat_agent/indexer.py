"""Repository indexing and search functionality."""

from pathlib import Path
from typing import Optional
import fnmatch
import os

MAX_FILE_BYTES = 1_000_000  # skip files larger than 1MB
DEFAULT_IGNORES = [
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".vscode",
    "dist",
    "build",
    "target",
    "vendor",
    ".idea",
    "internal",
    "assets",
    "snap",
    "test",
]
BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".class",
    ".jar",
    ".so",
    ".exe",
    ".dll",
    ".zip",
    ".tar",
    ".gz",
}


class RepositoryIndexer:
    """Index and search code across multiple repositories."""

    def __init__(self, repo_paths: list[str | Path]) -> None:
        """Initialize indexer with repository paths."""
        self.repo_paths = [Path(p) for p in repo_paths]

    def index_repositories(self) -> dict[str, dict]:
        """Index all repositories and extract metadata."""
        indexed_repos = {}
        for repo_path in self.repo_paths:
            files = self._extract_files(repo_path)
            indexed_repos[repo_path.name] = {
                "path": str(repo_path),
                "files": files,
            }
        return indexed_repos

    def _extract_files(self, repo_path: Path) -> list[dict]:
        """Extract file information from repository."""
        # Load .gitignore patterns if present
        gitignore_file = repo_path / ".gitignore"
        gitignore_patterns = []
        if gitignore_file.exists():
            try:
                with gitignore_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        gitignore_patterns.append(line)
            except Exception:
                gitignore_patterns = []

        files = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # skip large files
            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except Exception:
                pass

            if self._is_ignored(file_path, repo_path, gitignore_patterns):
                continue

            files.append({
                "path": str(file_path.relative_to(repo_path)),
                "language": self._detect_language(file_path),
            })

        return files

    @staticmethod
    def _is_ignored(file_path: Path, repo_root: Path, gitignore_patterns: list[str]) -> bool:
        """Check if file should be ignored based on defaults and .gitignore patterns.

        Args:
            file_path: absolute path to file
            repo_root: repository root path
            gitignore_patterns: list of patterns from .gitignore
        """
        parts = set(file_path.parts)
        if any(p in parts for p in DEFAULT_IGNORES):
            return True

        # Skip binary extensions
        if file_path.suffix.lower() in BINARY_EXTS:
            return True

        # Check gitignore patterns relative to repo root
        try:
            rel = str(file_path.relative_to(repo_root))
        except Exception:
            rel = str(file_path.name)

        for pat in gitignore_patterns:
            # Normalize simple directory patterns
            if pat.endswith("/"):
                pat = pat.rstrip("/")
                if rel.startswith(pat + "/") or rel == pat:
                    return True

            # fnmatch for glob patterns
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(file_path.name, pat):
                return True

        return False

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".sh": "bash",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".json": "json",
        }
        return extension_map.get(file_path.suffix, "unknown")
