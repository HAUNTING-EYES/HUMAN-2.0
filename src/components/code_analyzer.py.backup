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

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class CodeAnalyzer:
    """Code analysis tool."""
    
    def __init__(self, base_dir: str = None):
        """Initialize code analyzer.
        
        Args:
            base_dir: Base directory for code analysis
        """
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize code analysis tools
        self.ast_analyzer = ast.parse
        self.pattern_analyzer = re.compile
        
        # Initialize code improvement suggestions
        self.improvement_suggestions = {
            'complexity': [],
            'style': [],
            'security': [],
            'performance': []
        }
        
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code and return metrics.
        
        Args:
            code: Python code string to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Create a temporary file for pylint analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            metrics = self._compute_metrics(code)
            issues = self._find_issues(temp_file_path)
            ast_info = self._extract_ast_info(code)
            
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
            
    def _compute_metrics(self, code: str) -> Dict[str, int]:
        """Compute code metrics using AST analysis."""
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
            
    def _find_issues(self, file_path: str) -> List[Dict[str, Any]]:
        """Find code issues using pylint."""
        try:
            reporter = JSONReporter()
            Run([file_path], reporter=reporter, do_exit=False)
            return reporter.messages
        except Exception as e:
            self.logger.error(f"Error finding issues: {str(e)}")
            return []
            
    def _extract_ast_info(self, code: str) -> Dict[str, List[str]]:
        """Extract information from the AST."""
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

    def _analyze_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure."""
        return {
            'functions': self._extract_functions(code, language),
            'classes': self._extract_classes(code, language),
            'imports': self._extract_imports(code, language)
        }
        
    def _suggest_improvements(self, metrics: Dict[str, float], 
                            structure: Dict[str, Any]) -> List[str]:
        """Suggest potential code improvements."""
        suggestions = []
        
        # Example improvement suggestions based on metrics
        if metrics['complexity'] > 10:
            suggestions.append("Consider breaking down complex functions")
        if metrics['maintainability'] < 0.5:
            suggestions.append("Improve code documentation and structure")
            
        return suggestions
        
    def _calculate_complexity(self, code) -> float:
        """Calculate cyclomatic complexity using AST analysis.
        
        Args:
            code: Python code string or AST node
            
        Returns:
            Cyclomatic complexity score
        """
        try:
            if isinstance(code, str):
                tree = ast.parse(code)
            else:
                tree = code
                
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                # Add complexity for branching statements
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    # Add for each and/or operation
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.comprehension):
                    complexity += 1
                    if node.ifs:
                        complexity += len(node.ifs)
                        
            return float(complexity)
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 1.0
        
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability index based on code metrics.
        
        Args:
            code: Python code string
            
        Returns:
            Maintainability index between 0 and 1
        """
        try:
            lines = code.splitlines()
            total_lines = len(lines)
            
            # Count comments and docstrings
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            # Parse AST for more analysis
            tree = ast.parse(code)
            
            # Count docstrings
            docstring_count = 0
            function_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if ast.get_docstring(node):
                        docstring_count += 1
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
            
            # Calculate metrics
            comment_ratio = (comment_lines + docstring_count) / max(total_lines, 1)
            avg_function_length = total_lines / max(function_count, 1)
            
            # Maintainability score (higher is better)
            # Penalize long functions and reward documentation
            score = 1.0
            score -= max(0, (avg_function_length - 20) / 100)  # Penalty for long functions
            score += comment_ratio * 0.3  # Bonus for documentation
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating maintainability: {str(e)}")
            return 0.5
        
    def _extract_functions(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        """Extract function definitions from code.
        
        Args:
            code: Source code string
            language: Programming language (currently only Python supported)
            
        Returns:
            List of function information dictionaries
        """
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
        
    def _extract_classes(self, code: str, language: str = 'python') -> List[Dict[str, Any]]:
        """Extract class definitions from code.
        
        Args:
            code: Source code string
            language: Programming language (currently only Python supported)
            
        Returns:
            List of class information dictionaries
        """
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
        
    def _extract_imports(self, code: str, language: str = 'python') -> List[str]:
        """Extract import statements from code.
        
        Args:
            code: Source code string
            language: Programming language (currently only Python supported)
            
        Returns:
            List of imported module names
        """
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
            analysis['complexity'] = self._calculate_complexity(content)
            
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

    def improve_code(self, code: str) -> Dict[str, Any]:
        """Improve code based on analysis results.
        
        Args:
            code: Code to improve
            
        Returns:
            Dictionary containing improvement results
        """
        try:
            # Analyze code
            analysis = self.analyze_code(code)
            
            # Generate improvements
            improvements = []
            
            # Check complexity
            if analysis['complexity']['cyclomatic_complexity'] > 10:
                improvements.append({
                    'type': 'complexity',
                    'message': 'Consider breaking down complex functions',
                    'suggestion': 'Split function into smaller, more focused functions'
                })
                
            # Check style
            if not analysis['style']['follows_pep8']:
                improvements.append({
                    'type': 'style',
                    'message': 'Code does not follow PEP 8 style guide',
                    'suggestion': 'Format code according to PEP 8 standards'
                })
                
            # Check security
            if analysis['security']['vulnerabilities']:
                improvements.append({
                    'type': 'security',
                    'message': 'Potential security vulnerabilities detected',
                    'suggestion': 'Review and fix identified security issues'
                })
                
            # Check performance
            if analysis['performance']['bottlenecks']:
                improvements.append({
                    'type': 'performance',
                    'message': 'Performance bottlenecks detected',
                    'suggestion': 'Optimize identified performance bottlenecks'
                })
                
            # Apply improvements
            improved_code = code
            for improvement in improvements:
                if improvement['type'] == 'complexity':
                    improved_code = self._reduce_complexity(improved_code)
                elif improvement['type'] == 'style':
                    improved_code = self._apply_pep8(improved_code)
                elif improvement['type'] == 'security':
                    improved_code = self._fix_security(improved_code)
                elif improvement['type'] == 'performance':
                    improved_code = self._optimize_performance(improved_code)
                    
            return {
                'success': True,
                'improvements': improvements,
                'improved_code': improved_code
            }
            
        except Exception as e:
            self.logger.error(f"Error improving code: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _reduce_complexity(self, code: str) -> str:
        """Reduce code complexity.
        
        Args:
            code: Code to simplify
            
        Returns:
            Simplified code
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Find complex functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function complexity
                    complexity = self._calculate_complexity(node)
                    if complexity > 10:
                        # Split complex function
                        code = self._split_function(code, node)
                        
            return code
            
        except Exception as e:
            self.logger.error(f"Error reducing complexity: {str(e)}")
            return code
            
    def _apply_pep8(self, code: str) -> str:
        """Apply PEP 8 style guide to code.
        
        Args:
            code: Code to format
            
        Returns:
            Formatted code
        """
        try:
            # Use autopep8 to format code
            import autopep8
            return autopep8.fix_code(code)
            
        except Exception as e:
            self.logger.error(f"Error applying PEP 8: {str(e)}")
            return code
            
    def _fix_security(self, code: str) -> str:
        """Fix security vulnerabilities in code.
        
        Args:
            code: Code to secure
            
        Returns:
            Secured code
        """
        try:
            # Replace unsafe functions
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
            
    def _optimize_performance(self, code: str) -> str:
        """Optimize code performance.
        
        Args:
            code: Code to optimize
            
        Returns:
            Optimized code
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Find performance bottlenecks
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Check for list comprehension opportunities
                    if self._can_use_list_comprehension(node):
                        code = self._convert_to_list_comprehension(code, node)
                        
            return code
            
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {str(e)}")
            return code 