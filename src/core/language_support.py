"""
Multi-Language Support for HUMAN 2.0
Supports TypeScript, JavaScript, Java, and Python.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import subprocess
import json


class Language(Enum):
    """Supported languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"


@dataclass
class LanguageConfig:
    """Configuration for a programming language"""
    name: str
    extensions: List[str]
    comment_style: str  # "//", "#", "/* */"
    has_types: bool
    analyzer_command: Optional[str] = None
    formatter_command: Optional[str] = None


# Language configurations
LANGUAGE_CONFIGS = {
    Language.PYTHON: LanguageConfig(
        name="Python",
        extensions=[".py", ".pyw"],
        comment_style="#",
        has_types=True,
        analyzer_command="pylint",
        formatter_command="black"
    ),
    Language.TYPESCRIPT: LanguageConfig(
        name="TypeScript",
        extensions=[".ts", ".tsx"],
        comment_style="//",
        has_types=True,
        analyzer_command="eslint",
        formatter_command="prettier"
    ),
    Language.JAVASCRIPT: LanguageConfig(
        name="JavaScript",
        extensions=[".js", ".jsx", ".mjs"],
        comment_style="//",
        has_types=False,
        analyzer_command="eslint",
        formatter_command="prettier"
    ),
    Language.JAVA: LanguageConfig(
        name="Java",
        extensions=[".java"],
        comment_style="//",
        has_types=True,
        analyzer_command="checkstyle",
        formatter_command="google-java-format"
    )
}


class LanguageDetector:
    """Detect programming language from file path"""

    @staticmethod
    def detect(file_path: str) -> Optional[Language]:
        """Detect language from file extension"""
        for language, config in LANGUAGE_CONFIGS.items():
            if any(file_path.endswith(ext) for ext in config.extensions):
                return language
        return None


class LanguageAnalyzer:
    """Multi-language code analyzer"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, code: str, language: Language) -> Dict[str, Any]:
        """Analyze code in any supported language"""
        if language == Language.PYTHON:
            return self._analyze_python(code)
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            return self._analyze_typescript_javascript(code, language)
        elif language == Language.JAVA:
            return self._analyze_java(code)
        else:
            return {'error': f'Unsupported language: {language}'}

    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """Analyze Python code"""
        import ast
        try:
            tree = ast.parse(code)
            return {
                'language': 'python',
                'valid_syntax': True,
                'num_functions': sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
                'num_classes': sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
                'num_imports': sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
                'lines_of_code': len(code.splitlines()),
                'complexity_estimate': self._estimate_complexity(code)
            }
        except SyntaxError as e:
            return {
                'language': 'python',
                'valid_syntax': False,
                'error': str(e)
            }

    def _analyze_typescript_javascript(self, code: str, language: Language) -> Dict[str, Any]:
        """Analyze TypeScript/JavaScript code"""
        # Basic regex-based analysis (could use proper parsers like esprima)
        return {
            'language': language.value,
            'valid_syntax': self._check_js_ts_syntax(code),
            'num_functions': len(re.findall(r'\bfunction\s+\w+|const\s+\w+\s*=\s*\(|=>\s*{', code)),
            'num_classes': len(re.findall(r'\bclass\s+\w+', code)),
            'num_imports': len(re.findall(r'\bimport\s+.*from|require\(', code)),
            'has_typescript_types': 'interface' in code or ': ' in code if language == Language.TYPESCRIPT else False,
            'lines_of_code': len(code.splitlines()),
            'complexity_estimate': self._estimate_complexity(code)
        }

    def _analyze_java(self, code: str) -> Dict[str, Any]:
        """Analyze Java code"""
        return {
            'language': 'java',
            'valid_syntax': self._check_java_syntax(code),
            'num_methods': len(re.findall(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', code)),
            'num_classes': len(re.findall(r'\bclass\s+\w+', code)),
            'num_interfaces': len(re.findall(r'\binterface\s+\w+', code)),
            'num_imports': len(re.findall(r'\bimport\s+', code)),
            'lines_of_code': len(code.splitlines()),
            'complexity_estimate': self._estimate_complexity(code)
        }

    def _check_js_ts_syntax(self, code: str) -> bool:
        """Basic JavaScript/TypeScript syntax check"""
        # Check for balanced braces
        braces = 0
        for char in code:
            if char == '{':
                braces += 1
            elif char == '}':
                braces -= 1
            if braces < 0:
                return False
        return braces == 0

    def _check_java_syntax(self, code: str) -> bool:
        """Basic Java syntax check"""
        # Check for balanced braces and basic structure
        has_class = 'class ' in code
        braces = code.count('{') == code.count('}')
        return has_class and braces

    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity (simplified)"""
        # Count decision points
        complexity = 1  # Base complexity
        keywords = ['if', 'else', 'elif', 'for', 'while', 'case', 'catch', '&&', '||']
        for keyword in keywords:
            complexity += code.count(keyword)
        return min(complexity, 50)  # Cap at 50


class LanguageImprover:
    """Multi-language code improver"""

    def __init__(self, llm_client):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client

    async def improve(self, code: str, language: Language, strategy: str = "general") -> str:
        """Improve code in any supported language"""
        config = LANGUAGE_CONFIGS[language]

        prompt = self._build_improvement_prompt(code, config, strategy)

        try:
            improved_code = await self.llm_client.call_async(prompt)
            return self._extract_code(improved_code, config)
        except Exception as e:
            self.logger.error(f"Failed to improve {config.name} code: {e}")
            return code

    def _build_improvement_prompt(self, code: str, config: LanguageConfig, strategy: str) -> str:
        """Build improvement prompt for specific language"""
        language_specific_tips = {
            "TypeScript": """
- Use proper TypeScript types (interfaces, type aliases)
- Leverage generics for reusability
- Use async/await for promises
- Follow functional programming patterns where appropriate
            """,
            "JavaScript": """
- Use modern ES6+ features (arrow functions, destructuring, spread)
- Use const/let instead of var
- Use async/await for promises
- Apply functional programming patterns
            """,
            "Java": """
- Follow Java naming conventions
- Use proper access modifiers
- Leverage generics and collections
- Apply SOLID principles
- Use try-with-resources for resource management
            """
        }

        tips = language_specific_tips.get(config.name, "")

        return f"""You are a {config.name} code improvement expert.

Improve this {config.name} code following best practices:

```{config.name.lower()}
{code}
```

Strategy: {strategy}

{config.name}-specific guidelines:
{tips}

General guidelines:
- Improve readability and maintainability
- Reduce complexity
- Add proper error handling
- Optimize performance
- Add helpful comments where needed
- Ensure proper formatting

Return ONLY the improved code, no explanations:"""

    def _extract_code(self, response: str, config: LanguageConfig) -> str:
        """Extract code from LLM response"""
        # Try to extract code from markdown blocks
        pattern = f"```{config.name.lower()}\\n(.+?)\\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic code block
        pattern = "```\\n(.+?)\\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if no code block found
        return response.strip()


class TypeScriptAnalyzer(LanguageAnalyzer):
    """Specialized TypeScript analyzer"""

    def analyze_typescript(self, code: str) -> Dict[str, Any]:
        """Deep TypeScript analysis"""
        basic_analysis = self._analyze_typescript_javascript(code, Language.TYPESCRIPT)

        # Additional TypeScript-specific analysis
        basic_analysis.update({
            'num_interfaces': len(re.findall(r'\binterface\s+\w+', code)),
            'num_types': len(re.findall(r'\btype\s+\w+\s*=', code)),
            'num_enums': len(re.findall(r'\benum\s+\w+', code)),
            'has_generics': '<T' in code or '<K' in code or '<V' in code,
            'uses_async_await': 'async ' in code and 'await ' in code
        })

        return basic_analysis


class JavaAnalyzer(LanguageAnalyzer):
    """Specialized Java analyzer"""

    def analyze_java(self, code: str) -> Dict[str, Any]:
        """Deep Java analysis"""
        basic_analysis = self._analyze_java(code)

        # Additional Java-specific analysis
        basic_analysis.update({
            'num_annotations': len(re.findall(r'@\w+', code)),
            'has_generics': '<' in code and '>' in code,
            'uses_streams': 'stream()' in code,
            'uses_lambda': '->' in code,
            'package_declared': 'package ' in code
        })

        return basic_analysis


class MultiLanguageSupport:
    """Unified multi-language support"""

    def __init__(self, llm_client):
        self.logger = logging.getLogger(__name__)
        self.analyzer = LanguageAnalyzer()
        self.improver = LanguageImprover(llm_client)
        self.detector = LanguageDetector()
        self.ts_analyzer = TypeScriptAnalyzer()
        self.java_analyzer = JavaAnalyzer()

    def detect_language(self, file_path: str) -> Optional[Language]:
        """Detect language from file path"""
        return self.detector.detect(file_path)

    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Analyze code (auto-detect language)"""
        language = self.detect_language(file_path)
        if not language:
            return {'error': 'Unknown language'}

        if language == Language.TYPESCRIPT:
            return self.ts_analyzer.analyze_typescript(code)
        elif language == Language.JAVA:
            return self.java_analyzer.analyze_java(code)
        else:
            return self.analyzer.analyze(code, language)

    async def improve_code(self, code: str, file_path: str, strategy: str = "general") -> str:
        """Improve code (auto-detect language)"""
        language = self.detect_language(file_path)
        if not language:
            self.logger.warning(f"Unknown language for {file_path}")
            return code

        return await self.improver.improve(code, language, strategy)

    def get_language_config(self, file_path: str) -> Optional[LanguageConfig]:
        """Get language configuration"""
        language = self.detect_language(file_path)
        if language:
            return LANGUAGE_CONFIGS[language]
        return None

    def is_supported(self, file_path: str) -> bool:
        """Check if file is supported"""
        return self.detect_language(file_path) is not None


if __name__ == "__main__":
    # Test multi-language support
    logging.basicConfig(level=logging.INFO)

    # Test TypeScript
    ts_code = """
    interface User {
        name: string;
        age: number;
    }

    function greet(user: User): string {
        return `Hello, ${user.name}!`;
    }
    """

    analyzer = LanguageAnalyzer()
    result = analyzer.analyze(ts_code, Language.TYPESCRIPT)
    print("TypeScript Analysis:", json.dumps(result, indent=2))

    # Test Java
    java_code = """
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
    }
    """

    result = analyzer.analyze(java_code, Language.JAVA)
    print("Java Analysis:", json.dumps(result, indent=2))
