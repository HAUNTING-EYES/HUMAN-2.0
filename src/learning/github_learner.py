"""
HUMAN 2.0 - GitHub Learner
Actually clones repositories and extracts real code patterns.

This replaces the fake learning that only stored repo descriptions.

Features:
- Shallow clone repos (--depth 1 for speed)
- Parse Python files with AST
- Extract function/class patterns
- Store in ChromaDB for semantic search
- Cleanup after learning
"""

import ast
import os
import shutil
import subprocess
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class PatternType(Enum):
    """Types of code patterns."""
    FUNCTION = "function"
    CLASS = "class"
    DECORATOR = "decorator"
    ASYNC_PATTERN = "async_pattern"
    ERROR_HANDLING = "error_handling"
    DATA_CLASS = "dataclass"
    CONTEXT_MANAGER = "context_manager"


@dataclass
class CodePattern:
    """A learned code pattern."""
    pattern_id: str
    pattern_type: PatternType
    name: str
    code: str
    docstring: Optional[str]
    source_repo: str
    source_file: str
    line_number: int
    complexity: int  # Estimated complexity
    dependencies: List[str]  # Import dependencies
    tags: List[str] = field(default_factory=list)
    learned_at: datetime = field(default_factory=datetime.now)

    def to_document(self) -> str:
        """Convert to document for ChromaDB storage."""
        doc = f"# {self.pattern_type.value}: {self.name}\n\n"
        if self.docstring:
            doc += f"Description: {self.docstring}\n\n"
        doc += f"Source: {self.source_repo}\n"
        doc += f"File: {self.source_file}:{self.line_number}\n\n"
        doc += f"```python\n{self.code}\n```\n"
        if self.tags:
            doc += f"\nTags: {', '.join(self.tags)}"
        return doc


@dataclass
class LearningResult:
    """Result of learning from a repository."""
    repo_url: str
    repo_name: str
    patterns_learned: int
    files_analyzed: int
    total_lines: int
    patterns: List[CodePattern]
    errors: List[str]
    duration_seconds: float
    success: bool


class GitHubLearner:
    """
    Learn real code patterns from GitHub repositories.

    Actually clones repos, parses Python files, extracts patterns,
    and stores them in the knowledge base.
    """

    # Repos to skip (too large or not useful)
    SKIP_PATTERNS = [
        'test', 'tests', 'examples', 'docs', 'documentation',
        '__pycache__', '.git', 'node_modules', 'venv', 'env',
        'build', 'dist', '.tox', '.mypy_cache'
    ]

    # File size limits
    MAX_FILE_SIZE = 100 * 1024  # 100KB
    MAX_FILES_PER_REPO = 50
    MAX_PATTERNS_PER_FILE = 20

    def __init__(
        self,
        temp_dir: str = "temp/repos",
        chromadb_client=None,
        collection_name: str = "learned_patterns"
    ):
        """
        Initialize GitHub learner.

        Args:
            temp_dir: Directory for temporary clones
            chromadb_client: ChromaDB client for storage
            collection_name: Name of ChromaDB collection
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.chromadb_client = chromadb_client
        self.collection_name = collection_name

        # Stats
        self.total_repos_learned = 0
        self.total_patterns_learned = 0

        self.logger.info(f"GitHubLearner initialized. Temp dir: {self.temp_dir}")

    async def learn_from_repo(
        self,
        repo_url: str,
        topics: List[str] = None,
        max_files: int = None
    ) -> LearningResult:
        """
        Learn from a GitHub repository.

        Args:
            repo_url: GitHub repository URL
            topics: Topics/tags to associate with patterns
            max_files: Maximum files to analyze

        Returns:
            LearningResult with extracted patterns
        """
        import time
        start_time = time.time()

        topics = topics or []
        max_files = max_files or self.MAX_FILES_PER_REPO

        # Extract repo name
        repo_name = self._extract_repo_name(repo_url)
        repo_path = self.temp_dir / repo_name

        patterns = []
        errors = []
        files_analyzed = 0
        total_lines = 0

        try:
            # 1. Clone repository (shallow)
            self.logger.info(f"Cloning {repo_url}...")
            success = await self._clone_repo(repo_url, repo_path)

            if not success:
                return LearningResult(
                    repo_url=repo_url,
                    repo_name=repo_name,
                    patterns_learned=0,
                    files_analyzed=0,
                    total_lines=0,
                    patterns=[],
                    errors=["Failed to clone repository"],
                    duration_seconds=time.time() - start_time,
                    success=False
                )

            # 2. Find Python files
            py_files = self._find_python_files(repo_path, max_files)
            self.logger.info(f"Found {len(py_files)} Python files")

            # 3. Extract patterns from each file
            for py_file in py_files:
                try:
                    file_patterns, file_lines, file_errors = await self._extract_patterns_from_file(
                        py_file,
                        repo_url,
                        topics
                    )
                    patterns.extend(file_patterns)
                    total_lines += file_lines
                    errors.extend(file_errors)
                    files_analyzed += 1

                except Exception as e:
                    errors.append(f"Error processing {py_file}: {e}")

            # 4. Store patterns in ChromaDB
            if patterns and self.chromadb_client:
                await self._store_patterns(patterns)

            # 5. Update stats
            self.total_repos_learned += 1
            self.total_patterns_learned += len(patterns)

            duration = time.time() - start_time
            self.logger.info(
                f"Learned {len(patterns)} patterns from {repo_name} "
                f"({files_analyzed} files, {duration:.1f}s)"
            )

            return LearningResult(
                repo_url=repo_url,
                repo_name=repo_name,
                patterns_learned=len(patterns),
                files_analyzed=files_analyzed,
                total_lines=total_lines,
                patterns=patterns,
                errors=errors,
                duration_seconds=duration,
                success=True
            )

        finally:
            # 6. Cleanup
            self._cleanup(repo_path)

    async def _clone_repo(self, repo_url: str, repo_path: Path) -> bool:
        """Clone a repository (shallow clone)."""
        try:
            # Remove if exists
            if repo_path.exists():
                shutil.rmtree(repo_path)

            # Clone with depth 1 (shallow)
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', '--single-branch', repo_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode != 0:
                self.logger.error(f"Git clone failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"Clone timed out for {repo_url}")
            return False
        except Exception as e:
            self.logger.error(f"Clone error: {e}")
            return False

    def _find_python_files(self, repo_path: Path, max_files: int) -> List[Path]:
        """Find Python files in repo, excluding unwanted directories."""
        py_files = []

        for py_file in repo_path.rglob('*.py'):
            # Skip unwanted directories
            parts = py_file.relative_to(repo_path).parts
            if any(skip in parts for skip in self.SKIP_PATTERNS):
                continue

            # Skip large files
            try:
                if py_file.stat().st_size > self.MAX_FILE_SIZE:
                    continue
            except:
                continue

            py_files.append(py_file)

            if len(py_files) >= max_files:
                break

        return py_files

    async def _extract_patterns_from_file(
        self,
        file_path: Path,
        repo_url: str,
        topics: List[str]
    ) -> Tuple[List[CodePattern], int, List[str]]:
        """Extract code patterns from a Python file."""
        patterns = []
        errors = []

        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = len(code.split('\n'))

            # Parse AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return [], lines, [f"Syntax error in {file_path}: {e}"]

            # Extract functions and classes
            for node in ast.walk(tree):
                pattern = None

                if isinstance(node, ast.FunctionDef):
                    pattern = self._extract_function_pattern(
                        node, code, file_path, repo_url, topics
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    pattern = self._extract_function_pattern(
                        node, code, file_path, repo_url, topics, is_async=True
                    )
                elif isinstance(node, ast.ClassDef):
                    pattern = self._extract_class_pattern(
                        node, code, file_path, repo_url, topics
                    )

                if pattern:
                    patterns.append(pattern)

                    if len(patterns) >= self.MAX_PATTERNS_PER_FILE:
                        break

            return patterns, lines, errors

        except Exception as e:
            return [], 0, [f"Error reading {file_path}: {e}"]

    def _extract_function_pattern(
        self,
        node: ast.FunctionDef,
        code: str,
        file_path: Path,
        repo_url: str,
        topics: List[str],
        is_async: bool = False
    ) -> Optional[CodePattern]:
        """Extract a function pattern."""
        # Skip private functions (except __init__)
        if node.name.startswith('_') and node.name != '__init__':
            return None

        # Skip very short functions
        if len(node.body) < 2:
            return None

        # Extract code segment
        try:
            lines = code.split('\n')
            func_code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        except Exception as e:
            self.logger.debug(f"Failed to extract code segment for {node.name}: {e}")
            return None

        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity (simple heuristic)
        complexity = self._estimate_complexity(node)

        # Extract dependencies (imports used in function)
        dependencies = self._extract_dependencies(node, code)

        # Determine pattern type
        pattern_type = PatternType.ASYNC_PATTERN if is_async else PatternType.FUNCTION

        # Check for decorators
        if node.decorator_list:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    if dec.id == 'dataclass':
                        pattern_type = PatternType.DATA_CLASS
                    elif dec.id in ('contextmanager', 'asynccontextmanager'):
                        pattern_type = PatternType.CONTEXT_MANAGER

        # Generate unique ID
        pattern_id = self._generate_pattern_id(repo_url, str(file_path), node.name)

        # Build tags
        tags = list(topics)
        if is_async:
            tags.append('async')
        if docstring:
            tags.append('documented')

        return CodePattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            name=node.name,
            code=func_code,
            docstring=docstring,
            source_repo=repo_url,
            source_file=str(file_path.name),
            line_number=node.lineno,
            complexity=complexity,
            dependencies=dependencies,
            tags=tags
        )

    def _extract_class_pattern(
        self,
        node: ast.ClassDef,
        code: str,
        file_path: Path,
        repo_url: str,
        topics: List[str]
    ) -> Optional[CodePattern]:
        """Extract a class pattern."""
        # Skip private classes
        if node.name.startswith('_'):
            return None

        # Skip very simple classes
        if len(node.body) < 2:
            return None

        # Extract code segment
        try:
            lines = code.split('\n')
            class_code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        except:
            return None

        # Limit class size
        if len(class_code) > 5000:
            # Extract just the class signature and docstring
            class_code = self._extract_class_skeleton(node, code)

        docstring = ast.get_docstring(node)
        complexity = self._estimate_complexity(node)
        dependencies = self._extract_dependencies(node, code)

        pattern_id = self._generate_pattern_id(repo_url, str(file_path), node.name)

        tags = list(topics)
        tags.append('class')

        # Check for dataclass decorator
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'dataclass':
                tags.append('dataclass')

        return CodePattern(
            pattern_id=pattern_id,
            pattern_type=PatternType.CLASS,
            name=node.name,
            code=class_code,
            docstring=docstring,
            source_repo=repo_url,
            source_file=str(file_path.name),
            line_number=node.lineno,
            complexity=complexity,
            dependencies=dependencies,
            tags=tags
        )

    def _estimate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity of a node."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _extract_dependencies(self, node: ast.AST, code: str) -> List[str]:
        """Extract import dependencies used in a node."""
        # This is a simplified version - could be more sophisticated
        names_used = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names_used.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    names_used.add(child.value.id)

        # Filter to likely imports
        common_builtins = {
            'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'True', 'False', 'None', 'print', 'len', 'range', 'enumerate',
            'self', 'cls', 'super'
        }

        return [name for name in names_used if name not in common_builtins][:10]

    def _extract_class_skeleton(self, node: ast.ClassDef, code: str) -> str:
        """Extract class skeleton (signature + docstring + method signatures)."""
        lines = code.split('\n')
        skeleton_lines = []

        # Class definition line
        skeleton_lines.append(lines[node.lineno - 1])

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            skeleton_lines.append(f'    """{docstring}"""')

        # Method signatures
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                skeleton_lines.append(f"    {lines[item.lineno - 1].strip()}")
                skeleton_lines.append("        ...")

        return '\n'.join(skeleton_lines)

    def _generate_pattern_id(self, repo_url: str, file_path: str, name: str) -> str:
        """Generate a unique pattern ID."""
        content = f"{repo_url}:{file_path}:{name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        # Handle various URL formats
        name = repo_url.rstrip('/').split('/')[-1]
        if name.endswith('.git'):
            name = name[:-4]
        return name

    async def _store_patterns(self, patterns: List[CodePattern]):
        """Store patterns in ChromaDB."""
        if not self.chromadb_client:
            self.logger.warning("No ChromaDB client - patterns not stored")
            return

        try:
            collection = self.chromadb_client.get_or_create_collection(
                name=self.collection_name
            )

            documents = []
            metadatas = []
            ids = []

            for pattern in patterns:
                documents.append(pattern.to_document())
                metadatas.append({
                    'pattern_type': pattern.pattern_type.value,
                    'name': pattern.name,
                    'source_repo': pattern.source_repo,
                    'complexity': pattern.complexity,
                    'tags': ','.join(pattern.tags),
                    'learned_at': pattern.learned_at.isoformat()
                })
                ids.append(pattern.pattern_id)

            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            self.logger.info(f"Stored {len(patterns)} patterns in ChromaDB")

        except Exception as e:
            self.logger.error(f"Failed to store patterns: {e}")

    def _cleanup(self, repo_path: Path):
        """Cleanup cloned repository."""
        try:
            if repo_path.exists():
                shutil.rmtree(repo_path)
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'total_repos_learned': self.total_repos_learned,
            'total_patterns_learned': self.total_patterns_learned,
            'temp_dir': str(self.temp_dir)
        }


# Convenience function
async def learn_from_repo(repo_url: str, **kwargs) -> LearningResult:
    """Quick function to learn from a repository."""
    learner = GitHubLearner()
    return await learner.learn_from_repo(repo_url, **kwargs)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing GitHub Learner...")
        learner = GitHubLearner()

        # Test with a small repo
        result = await learner.learn_from_repo(
            "https://github.com/pallets/click",
            topics=["cli", "python"],
            max_files=10
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Patterns learned: {result.patterns_learned}")
        print(f"  Files analyzed: {result.files_analyzed}")
        print(f"  Duration: {result.duration_seconds:.1f}s")

        if result.patterns:
            print(f"\nSample patterns:")
            for pattern in result.patterns[:3]:
                print(f"  - {pattern.pattern_type.value}: {pattern.name}")
                print(f"    Complexity: {pattern.complexity}")
                print(f"    Docstring: {pattern.docstring[:50] if pattern.docstring else 'None'}...")

        if result.errors:
            print(f"\nErrors: {len(result.errors)}")
            for err in result.errors[:3]:
                print(f"  - {err}")

        print(f"\nStats: {learner.get_stats()}")

    asyncio.run(test())
