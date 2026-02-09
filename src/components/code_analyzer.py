import os
import ast
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import tempfile
from pylint.lint import Run
from pylint.reporters import JSONReporter
import re

root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')


class ASTAnalyzer:
    """Handles AST-based code analysis."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def compute_metrics(self, code: str) -> Dict[str, int]:
        try:
            tree = ast.parse(code)
            metrics = {
                'num_functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                'num_classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                'num_imports': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
                'num_lines': len(code.splitlines()),
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {}
            
    def extract_ast_info(self, code: str) -> Dict[str, List[str]]:
        try:
            tree = ast.parse(code)
            info = {
                'imports': [],
                'functions': [],
                'classes': [],
                'variables': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        info['imports'].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        info['imports'].append(f"{module}.{name.name}")
                elif isinstance(node, ast.FunctionDef):
                    info['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    info['classes'].append(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    info['variables'].append(node.id)
                    
            return info
        except Exception as e:
            self.logger.error(f"Error extracting AST info: {str(e)}")
            return {}
            
    def extract_functions(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        try:
            if language != 'python':
                return []
                
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list],
                        'has_docstring': ast.get_docstring(node) is not None,
                        'body_lines': len(node.body)
                    })
                    
            return functions
            
        except Exception as e:
            self.logger.error(f"Error extracting functions: {str(e)}")
            return []
        
    def extract_classes(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        try:
            if language != 'python':
                return []
                
            tree = ast.parse(code)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'bases': [ast.unparse(b) if hasattr(ast, 'unparse') else str(b) for b in node.bases],
                        'methods': methods,
                        'has_docstring': ast.get_docstring(node) is not None
                    })
                    
            return classes
            
        except Exception as e:
            self.logger.error(f"Error extracting classes: {str(e)}")
            return []
        
    def extract_imports(self, code: str, language: str = 'python') -> List[str]:
        try:
            if language != 'python':
                return []
                
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
                        
            return imports
            
        except Exception as e:
            self.logger.error(f"Error extracting imports: {str(e)}")
            return []


class ComplexityAnalyzer:
    """Handles complexity calculations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def calculate_complexity(self, code) -> float:
        try:
            if isinstance(code, str):
                tree = ast.parse(code)
            else:
                tree = code
                
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.comprehension):
                    complexity += 1
                    if node.ifs:
                        complexity += len(node.ifs)
                        
            return float(complexity)
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 1.0
        
    def calculate_maintainability(self, code: str) -> float:
        try:
            lines = code.splitlines()
            total_lines = len(lines)
            
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            tree = ast.parse(code)
            
            docstring_count = 0
            function_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if ast.get_docstring(node):
                        docstring_count += 1
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
            
            comment_ratio = (comment_lines + docstring_count) / max(total_lines, 1)
            avg_function_length = total_lines / max(function_count, 1)
            
            score = 1.0
            score -= max(0, (avg_function_length - 20) / 100)
            score += comment_ratio * 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating maintainability: {str(e)}")
            return 0.5


class IssueDetector:
    """Detects code quality issues."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def find_issues(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            reporter = JSONReporter()
            Run([file_path], reporter=reporter, do_exit=False)
            return reporter.messages
        except Exception as e:
            self.logger.error(f"Error finding issues: {str(e)}")
            return []
            
    def check_common_issues(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                analysis['issues'].append({
                    'type': 'long_function',
                    'line': node.lineno,
                    'message': f"Function '{node.name}' is too long ({len(node.body)} lines)"
                })
                
            if isinstance(node, ast.Call) and len(node.args) > 5:
                analysis['issues'].append({
                    'type': 'complex_call',
                    'line': node.lineno,
                    'message': "Function call has too many arguments"
                })
                
    def find_optimizations(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                analysis['optimizations'].append({
                    'type': 'list_to_generator',
                    'line': node.lineno,
                    'message': "Consider using a generator expression instead of list comprehension"
                })
                
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('join', 'split', 'replace'):
                    analysis['optimizations'].append({
                        'type': 'string_operation',
                        'line': node.lineno,
                        'message': f"Consider optimizing string operation '{node.func.attr}'"
                    })
                    
    def generate_suggestions(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.returns:
                analysis['suggestions'].append({
                    'type': 'add_type_hints',
                    'line': node.lineno,
                    'message': f"Add return type hint to function '{node.name}'"
                })
                
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                analysis['suggestions'].append({
                    'type': 'add_docstring',
                    'line': node.lineno,
                    'message': f"Add docstring to {node.__class__.__name__} '{node.name}'"
                })


class CodeImprover:
    """Handles code improvement operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def reduce_complexity(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_node_complexity(node)
                    if complexity > 10:
                        code = self._split_function(code, node)
                        
            return code
            
        except Exception as e:
            self.logger.error(f"Error reducing complexity: {str(e)}")
            return code
            
    def _calculate_node_complexity(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
        
    def _split_function(self, code: str, node: ast.FunctionDef) -> str:
        return code
            
    def apply_pep8(self, code: str) -> str:
        try:
            import autopep8
            return autopep8.fix_code(code)
            
        except Exception as e:
            self.logger.error(f"Error applying PEP 8: {str(e)}")
            return code
            
    def fix_security(self, code: str) -> str:
        try:
            unsafe_patterns = {
                r'eval\(': 'ast.literal_eval(',
                r'exec\(': '# Removed unsafe exec call',
                r'os\.system\(': 'subprocess.run(',
                r'subprocess\.call\(': 'subprocess.run('
            }
            
            for pattern, replacement in unsafe_patterns.items():
                code = re.sub(pattern, replacement, code)
                
            return code
            
        except Exception as e:
            self.logger.error(f"Error fixing security: {str(e)}")
            return code
            
    def optimize_performance(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    if self._can_use_list_comprehension(node):
                        code = self._convert_to_list_comprehension(code, node)
                        
            return code
            
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {str(e)}")
            return code
            
    def _can_use_list_comprehension(self, node: ast.For) -> bool:
        return False
        
    def _convert_to_list_comprehension(self, code: str, node: ast.For) -> str:
        return code


class CodeAnalyzer:
    """Code analysis tool."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        
        self.ast_analyzer = ASTAnalyzer(self.logger)
        self.complexity_analyzer = ComplexityAnalyzer(self.logger)
        self.issue_detector = IssueDetector(self.logger)
        self.code_improver = CodeImprover(self.logger)
        
        self.improvement_suggestions = {
            'complexity': [],
            'style': [],
            'security': [],
            'performance': []
        }
        
    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            metrics = self.ast_analyzer.compute_metrics(code)
            issues = self.issue_detector.find_issues(temp_file_path)
            ast_info = self.ast_analyzer.extract_ast_info(code)
            
            return {
                'success': True,
                'metrics': metrics,
                'issues': issues,
                'ast_info': ast_info,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing code: {str(e)}")
            return {
                'success': False,
                'metrics': {},
                'issues': [],
                'ast_info': {},
                'error': str(e)
            }

    def _analyze_structure(self, code: str, language: str) -> Dict[str, Any]:
        return {
            'functions': self.ast_analyzer.extract_functions(code, language),
            'classes': self.ast_analyzer.extract_classes(code, language),
            'imports': self.ast_analyzer.extract_imports(code, language)
        }
        
    def _suggest_improvements(self, metrics: Dict[str, float], 
                            structure: Dict[str, Any]) -> List[str]:
        suggestions = []