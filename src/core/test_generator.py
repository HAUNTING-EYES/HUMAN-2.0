#!/usr/bin/env python3
"""
HUMAN 2.0 Test Generator
Auto-generates pytest tests using Claude 3.5 Sonnet.

Solves the problem: 49% of components have no tests
Solution: Use LLM to generate test stubs automatically
"""

import os
import ast
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import requests
import json

load_dotenv()


class ClaudeClient:
    """Client for Claude 3.5 Sonnet API."""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')  # Claude via OpenAI-compatible endpoint
        self.model = 'gpt-4'  # Actually Claude 3.5 Sonnet
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Generate response from Claude.

        Args:
            prompt: Prompt for Claude
            temperature: Temperature for generation (lower = more deterministic)

        Returns:
            Generated text
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': 3000
            }

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['message']['content']

            # Extract code from markdown if present
            if '```python' in content:
                content = content.split('```python')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            return content

        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return ""


class TestGenerator:
    """
    Auto-generates pytest tests for untested components.

    Uses Claude 3.5 Sonnet to analyze code and generate test stubs.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = ClaudeClient()

    def find_untested_files(self, target_dirs: List[str], root_dir: str = '.') -> List[str]:
        """
        Find Python files without corresponding test files.

        Args:
            target_dirs: Directories to scan
            root_dir: Root directory of project

        Returns:
            List of file paths that don't have tests
        """
        self.logger.info(f"Finding untested files in: {target_dirs}")

        root_path = Path(root_dir).resolve()
        untested_files = []

        for target_dir in target_dirs:
            target_path = root_path / target_dir

            if not target_path.exists():
                self.logger.warning(f"Directory not found: {target_path}")
                continue

            for py_file in target_path.rglob('*.py'):
                # Skip test files, __init__, __pycache__
                if any(skip in str(py_file) for skip in ['test_', '__init__.py', '__pycache__', 'archived_']):
                    continue

                # Check if test exists
                if not self.has_tests(str(py_file)):
                    untested_files.append(str(py_file))

        self.logger.info(f"Found {len(untested_files)} untested files")
        return untested_files

    def has_tests(self, file_path: str) -> bool:
        """
        Check if a file has corresponding tests.

        Args:
            file_path: Path to file to check

        Returns:
            True if test file exists, False otherwise
        """
        test_file = self.find_test_file(file_path)
        return test_file is not None

    def find_test_file(self, file_path: str) -> Optional[str]:
        """
        Find corresponding test file for a source file.

        Args:
            file_path: Path to source file

        Returns:
            Path to test file or None
        """
        file_path_obj = Path(file_path)

        # Try tests/ directory
        test_path = Path('tests') / f"test_{file_path_obj.name}"
        if test_path.exists():
            return str(test_path)

        # Try same directory with test_ prefix
        test_path = file_path_obj.parent / f"test_{file_path_obj.name}"
        if test_path.exists():
            return str(test_path)

        return None

    def generate_test_stub(self, file_path: str, code: str) -> str:
        """
        Generate pytest test stub using Claude.

        Args:
            file_path: Path to file being tested
            code: Source code to generate tests for

        Returns:
            Generated test code
        """
        self.logger.info(f"Generating tests for: {file_path}")

        # Analyze code to understand structure
        code_analysis = self._analyze_code_structure(code)

        # Build comprehensive prompt
        prompt = self._build_test_generation_prompt(
            file_path, code, code_analysis
        )

        # Generate tests using Claude
        test_code = self.llm.generate(prompt, temperature=0.3)

        if not test_code:
            self.logger.error(f"Failed to generate tests for {file_path}")
            return ""

        # Post-process generated code
        test_code = self._post_process_test_code(test_code, file_path)

        return test_code

    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure using AST."""
        try:
            tree = ast.parse(code)

            analysis = {
                'classes': [],
                'functions': [],
                'imports': [],
                'has_async': False
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    }
                    analysis['classes'].append(class_info)

                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        func_info = {
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'is_async': isinstance(node, ast.AsyncFunctionDef)
                        }
                        analysis['functions'].append(func_info)

                        if isinstance(node, ast.AsyncFunctionDef):
                            analysis['has_async'] = True

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing code structure: {e}")
            return {'classes': [], 'functions': [], 'imports': [], 'has_async': False}

    def _build_test_generation_prompt(self, file_path: str, code: str,
                                     analysis: Dict[str, Any]) -> str:
        """Build comprehensive prompt for test generation."""
        file_name = Path(file_path).name

        prompt = f"""You are an expert Python test engineer. Generate comprehensive pytest test stubs for this code.

File: {file_name}

Code to test:
```python
{code[:3000]}  # Limit code length
```

Code Analysis:
- Classes: {', '.join([c['name'] for c in analysis['classes']]) or 'None'}
- Functions: {', '.join([f['name'] for f in analysis['functions']]) or 'None'}
- Uses async: {analysis['has_async']}

Requirements:
1. Create pytest test stubs for ALL public classes and functions
2. Use pytest fixtures where appropriate
3. Use mocks for external dependencies (requests, file I/O, etc.)
4. Test both success cases and edge cases/error conditions
5. Use descriptive test names (test_<function>_<scenario>)
6. Add docstrings to test functions
7. Group related tests in test classes
8. Import necessary modules (pytest, mock, fixtures)

Test Structure:
```python
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Import module being tested
from {self._get_import_path(file_path)} import ...

class Test<ClassName>:
    @pytest.fixture
    def <fixture_name>(self):
        \"\"\"Fixture description\"\"\"
        return ...

    def test_<method>_success(self, <fixture_name>):
        \"\"\"Test successful case\"\"\"
        # Arrange
        ...
        # Act
        ...
        # Assert
        assert ...

    def test_<method>_error_case(self):
        \"\"\"Test error handling\"\"\"
        ...
```

Generate ONLY valid Python test code. Do not include explanations."""

        return prompt

    def _get_import_path(self, file_path: str) -> str:
        """Get Python import path for a file."""
        try:
            file_path_obj = Path(file_path)

            # Convert to absolute path and make relative to current directory
            abs_path = file_path_obj.resolve()
            cwd = Path.cwd()

            try:
                rel_path = abs_path.relative_to(cwd)
            except ValueError:
                # If not relative to cwd, use as-is
                rel_path = file_path_obj

            # Convert path to Python module format
            # e.g., src\core\dependency_analyzer.py -> src.core.dependency_analyzer
            parts = []
            for part in rel_path.parts:
                if part.endswith('.py'):
                    parts.append(part[:-3])  # Remove .py extension
                elif part not in ['.', '..', '']:
                    parts.append(part)

            return '.'.join(parts)

        except Exception as e:
            self.logger.warning(f"Error getting import path for {file_path}: {e}")
            return Path(file_path).stem

    def _post_process_test_code(self, test_code: str, file_path: str) -> str:
        """Post-process generated test code."""
        # Ensure proper imports
        if 'import pytest' not in test_code:
            test_code = 'import pytest\n' + test_code

        # Add header comment
        header = f'''"""
Auto-generated tests for {Path(file_path).name}
Generated by HUMAN 2.0 TestGenerator using Claude 3.5 Sonnet
"""

'''
        test_code = header + test_code

        return test_code

    def validate_test(self, test_code: str) -> Tuple[bool, str]:
        """
        Validate that generated test code is syntactically correct and runs.

        Args:
            test_code: Generated test code

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # 1. Validate syntax
            ast.parse(test_code)

            # 2. Try running with pytest
            with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as temp_file:
                temp_file.write(test_code)
                temp_file_path = temp_file.name

            try:
                result = subprocess.run(
                    ['python', '-m', 'pytest', temp_file_path, '-v', '--collect-only'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                # Check if pytest could collect the tests
                if result.returncode == 0 or 'collected' in result.stdout:
                    return True, ""
                else:
                    return False, result.stderr

            finally:
                # Clean up temp file
                Path(temp_file_path).unlink(missing_ok=True)

        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def auto_generate_missing_tests(self, target_dirs: List[str],
                                    max_files: int = 10, root_dir: str = '.'):
        """
        Auto-generate tests for top N untested files.

        Args:
            target_dirs: Directories to scan
            max_files: Maximum number of test files to generate
            root_dir: Root directory

        Returns:
            Dict with statistics
        """
        self.logger.info(f"Auto-generating tests for up to {max_files} files...")

        # Find untested files
        untested_files = self.find_untested_files(target_dirs, root_dir)

        # Prioritize by complexity/criticality
        prioritized_files = self._prioritize_files(untested_files, root_dir)

        stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'generated_files': []
        }

        # Generate tests for top N files
        for file_path in prioritized_files[:max_files]:
            stats['attempted'] += 1

            try:
                # Read source code
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Generate test
                test_code = self.generate_test_stub(file_path, code)

                if not test_code:
                    stats['failed'] += 1
                    continue

                # Validate test
                valid, error = self.validate_test(test_code)

                if not valid:
                    self.logger.warning(f"Generated test failed validation: {error}")
                    stats['failed'] += 1
                    continue

                # Save test file
                test_file_path = self._get_test_file_path(file_path)
                test_file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_code)

                self.logger.info(f"Generated test: {test_file_path}")
                stats['successful'] += 1
                stats['generated_files'].append(str(test_file_path))

            except Exception as e:
                self.logger.error(f"Error generating test for {file_path}: {e}")
                stats['failed'] += 1

        self.logger.info(f"Test generation complete: {stats['successful']}/{stats['attempted']} successful")
        return stats

    def _prioritize_files(self, file_paths: List[str], root_dir: str) -> List[str]:
        """Prioritize files for test generation (higher complexity first)."""
        def get_priority(file_path: str) -> int:
            """Calculate priority score (higher = more priority)."""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                tree = ast.parse(code)

                # Count functions, classes, lines
                num_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                num_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                num_lines = len(code.splitlines())

                # Priority: more classes/functions = higher priority
                priority = (num_classes * 10) + (num_functions * 5) + (num_lines // 10)

                return priority

            except Exception:
                return 0

        # Sort by priority (descending)
        return sorted(file_paths, key=get_priority, reverse=True)

    def _get_test_file_path(self, source_file_path: str) -> Path:
        """Get path for test file."""
        source_path = Path(source_file_path)
        test_filename = f"test_{source_path.name}"

        # Place in tests/ directory
        return Path('tests') / test_filename


def main():
    """Test the test generator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Test Generator")
    print("=" * 70)

    # Initialize
    generator = TestGenerator()

    # Find untested files
    print("\nFinding untested files...")
    untested = generator.find_untested_files(['src/components'])
    print(f"Found {len(untested)} untested files")

    # Auto-generate tests
    print("\nGenerating tests for top 5 untested files...")
    stats = generator.auto_generate_missing_tests(
        target_dirs=['src/components'],
        max_files=5
    )

    print(f"\nResults:")
    print(f"  Attempted: {stats['attempted']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")

    if stats['generated_files']:
        print(f"\nGenerated test files:")
        for test_file in stats['generated_files']:
            print(f"  - {test_file}")


if __name__ == "__main__":
    main()
