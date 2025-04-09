import os
import ast
import logging
from typing import Dict, List, Any
from datetime import datetime

class CodeAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase for potential improvements"""
        analysis = {
            'total_files': 0,
            'total_lines': 0,
            'complexity_score': 0,
            'code_quality_issues': [],
            'potential_optimizations': [],
            'suggested_improvements': []
        }
        
        # Walk through all Python files
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Analyze file
                        file_analysis = self._analyze_file(content, file_path)
                        
                        # Update overall analysis
                        analysis['total_files'] += 1
                        analysis['total_lines'] += file_analysis['lines']
                        analysis['complexity_score'] += file_analysis['complexity']
                        
                        # Add issues and suggestions
                        analysis['code_quality_issues'].extend(file_analysis['issues'])
                        analysis['potential_optimizations'].extend(file_analysis['optimizations'])
                        analysis['suggested_improvements'].extend(file_analysis['suggestions'])
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
                        
        return analysis
        
    def _analyze_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file"""
        analysis = {
            'lines': len(content.splitlines()),
            'complexity': 0,
            'issues': [],
            'optimizations': [],
            'suggestions': []
        }
        
        try:
            # Parse the AST
            tree = ast.parse(content)
            
            # Analyze complexity
            analysis['complexity'] = self._calculate_complexity(tree)
            
            # Check for common issues
            self._check_common_issues(tree, file_path, analysis)
            
            # Look for optimization opportunities
            self._find_optimizations(tree, file_path, analysis)
            
            # Generate suggestions
            self._generate_suggestions(tree, file_path, analysis)
            
        except SyntaxError as e:
            analysis['issues'].append({
                'type': 'syntax_error',
                'line': e.lineno,
                'message': str(e)
            })
            
        return analysis
        
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
        
    def _check_common_issues(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        """Check for common code quality issues"""
        for node in ast.walk(tree):
            # Check for long functions
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                analysis['issues'].append({
                    'type': 'long_function',
                    'line': node.lineno,
                    'message': f"Function '{node.name}' is too long ({len(node.body)} lines)"
                })
                
            # Check for complex expressions
            if isinstance(node, ast.Call) and len(node.args) > 5:
                analysis['issues'].append({
                    'type': 'complex_call',
                    'line': node.lineno,
                    'message': "Function call has too many arguments"
                })
                
    def _find_optimizations(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        """Find potential optimization opportunities"""
        for node in ast.walk(tree):
            # Check for list comprehensions that could be generator expressions
            if isinstance(node, ast.ListComp):
                analysis['optimizations'].append({
                    'type': 'list_to_generator',
                    'line': node.lineno,
                    'message': "Consider using a generator expression instead of list comprehension"
                })
                
            # Check for repeated string operations
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('join', 'split', 'replace'):
                    analysis['optimizations'].append({
                        'type': 'string_operation',
                        'line': node.lineno,
                        'message': f"Consider optimizing string operation '{node.func.attr}'"
                    })
                    
    def _generate_suggestions(self, tree: ast.AST, file_path: str, analysis: Dict[str, Any]):
        """Generate improvement suggestions"""
        # Add type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.returns:
                analysis['suggestions'].append({
                    'type': 'add_type_hints',
                    'line': node.lineno,
                    'message': f"Add return type hint to function '{node.name}'"
                })
                
        # Add docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                analysis['suggestions'].append({
                    'type': 'add_docstring',
                    'line': node.lineno,
                    'message': f"Add docstring to {node.__class__.__name__} '{node.name}'"
                })
                
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on code analysis"""
        analysis = self.analyze_codebase()
        improvements = []
        
        # Convert issues to improvements
        for issue in analysis['code_quality_issues']:
            improvements.append({
                'type': 'code_quality',
                'description': issue['message'],
                'priority': 'high',
                'file': issue.get('file', 'unknown'),
                'line': issue.get('line', 0)
            })
            
        # Convert optimizations to improvements
        for opt in analysis['potential_optimizations']:
            improvements.append({
                'type': 'optimization',
                'description': opt['message'],
                'priority': 'medium',
                'file': opt.get('file', 'unknown'),
                'line': opt.get('line', 0)
            })
            
        # Add suggestions
        for suggestion in analysis['suggested_improvements']:
            improvements.append({
                'type': 'enhancement',
                'description': suggestion['message'],
                'priority': 'low',
                'file': suggestion.get('file', 'unknown'),
                'line': suggestion.get('line', 0)
            })
            
        return improvements 