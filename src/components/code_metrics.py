import ast
import re
from typing import Dict, Any, List, Tuple
import radon.metrics as rm
import radon.complexity as rc
import logging

class CodeMetrics:
    """Code metrics calculator."""
    
    def calculate_metrics(self, code: str) -> dict:
        """Calculate code quality metrics.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate complexity (based on AST node count)
        try:
            tree = ast.parse(code)
            node_count = sum(1 for _ in ast.walk(tree))
            metrics['complexity'] = 1.0 / (1.0 + node_count / 10.0)  # Normalize
        except:
            metrics['complexity'] = 0.5
            
        # Calculate documentation score
        doc_lines = len(re.findall(r'"""[\s\S]*?"""', code))
        total_lines = len(code.split('\n'))
        metrics['documentation'] = min(1.0, doc_lines / max(1, total_lines / 10))
        
        # Calculate maintainability
        metrics['maintainability'] = self._calculate_maintainability(code)
        
        # Calculate security score
        metrics['security'] = self._calculate_security(code)
        
        # Calculate style score
        metrics['style'] = self._calculate_style(code)
        
        # Calculate test coverage (dummy value for now)
        metrics['test_coverage'] = 0.5
        
        # Calculate overall score
        metrics['overall_score'] = sum(metrics.values()) / len(metrics)
        
        return metrics
        
    def __init__(self):
        """Initialize code metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_metrics(self, code: str) -> Dict[str, float]:
        """Calculate code quality metrics.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary of metrics and their values
        """
        try:
            # Calculate metrics
            metrics = {
                "complexity": self._calculate_complexity(code),
                "maintainability": self._calculate_maintainability(code),
                "security": self._calculate_security(code),
                "style": self._calculate_style(code),
                "documentation": self._calculate_documentation(code),
                "test_coverage": self._calculate_test_coverage(code)
            }
            
            # Calculate overall score
            metrics["overall_score"] = sum(metrics.values()) / len(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "complexity": 0.0,
                "maintainability": 0.0,
                "security": 0.0,
                "style": 0.0,
                "documentation": 0.0,
                "test_coverage": 0.0,
                "overall_score": 0.0
            }
            
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score.
        
        Args:
            code: Source code string
            
        Returns:
            Complexity score (0-1)
        """
        try:
            # Calculate cyclomatic complexity using radon
            blocks = rc.cc_visit(code)
            if not blocks:
                return 1.0  # Empty or very simple code
                
            # Get average complexity
            total_complexity = sum(block.complexity for block in blocks)
            avg_complexity = total_complexity / len(blocks)
                
            # Normalize score (lower complexity is better)
            max_complexity = 10  # Threshold for very complex code
            score = max(0, 1 - (avg_complexity / max_complexity))
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 0.0
            
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability score."""
        score = 0.7  # Base score
        
        # Check for type hints
        if 'def' in code and ':' in code and '->' in code:
            score += 0.1
            
        # Check for docstrings
        if '"""' in code:
            score += 0.1
            
        # Check for reasonable line length
        if all(len(line) <= 80 for line in code.split('\n')):
            score += 0.1
            
        return min(1.0, score)
        
    def _calculate_security(self, code: str) -> float:
        """Calculate security score."""
        score = 1.0  # Start with perfect score
        
        # Check for dangerous patterns
        dangerous_patterns = [
            'eval(',
            'exec(',
            'os.system(',
            '__import__(',
            'subprocess.run(',
            'input('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                score -= 0.2
                
        return max(0.0, score)
        
    def _calculate_style(self, code: str) -> float:
        """Calculate style score."""
        score = 0.7  # Base score
        
        # Check indentation consistency
        if not '\t' in code:
            score += 0.1
            
        # Check naming conventions
        if not re.search(r'[A-Z]', code.split('def ')[0]):  # No capitals outside class names
            score += 0.1
            
        # Check for trailing whitespace
        if not any(line.rstrip() != line for line in code.split('\n')):
            score += 0.1
            
        return min(1.0, score)
            
    def _calculate_documentation(self, code: str) -> float:
        """Calculate documentation score.
        
        Args:
            code: Source code string
            
        Returns:
            Documentation score (0-1)
        """
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Count docstrings and nodes that should have docstrings
            docstring_count = 0
            total_nodes = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    total_nodes += 1
                    try:
                        docstring = ast.get_docstring(node, clean=False)
                        if docstring and len(docstring.strip()) > 0:
                            docstring_count += 1
                    except:
                        pass
                        
            # Calculate score
            if total_nodes == 0:
                return 1.0  # Empty file is well documented
            
            # Weight docstrings more heavily
            return min(1.0, docstring_count / total_nodes * 1.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating documentation: {str(e)}")
            return 0.0
            
    def _calculate_test_coverage(self, code: str) -> float:
        """Calculate test coverage score.
        
        Args:
            code: Source code
            
        Returns:
            Test coverage score (0-1)
        """
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Count test functions and regular functions
            test_functions = []
            regular_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        test_functions.append(node)
                    else:
                        regular_functions.append(node)
                        
            # Calculate coverage
            if not regular_functions:
                return 0.0  # No functions to test
            
            # Check for assertions in test functions
            has_assertions = False
            for test_func in test_functions:
                for node in ast.walk(test_func):
                    if isinstance(node, ast.Assert):
                        has_assertions = True
                        break
                    
            # Return positive score if we have test functions with assertions
            if test_functions and has_assertions:
                return min(1.0, len(test_functions) / len(regular_functions))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating test coverage: {str(e)}")
            return 0.0
            
    def get_improvement_suggestions(self, code: str) -> List[Dict[str, Any]]:
        """Generate suggestions for code improvement.
        
        Args:
            code: Source code to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            # Check complexity
            complexity = self._calculate_complexity(code)
            if complexity < 0.7:
                suggestions.append({
                    "type": "complexity",
                    "severity": "high",
                    "description": "Code is too complex. Consider breaking down into smaller functions."
                })
                
            # Check security
            security = self._calculate_security(code)
            if security < 0.7:
                suggestions.append({
                    "type": "security",
                    "severity": "critical",
                    "description": "Security issues detected. Check for unsafe eval(), SQL injection risks, and proper error handling."
                })
                
            # Check style
            style = self._calculate_style(code)
            if style < 0.7:
                suggestions.append({
                    "type": "style",
                    "severity": "medium",
                    "description": "Style issues found. Check line lengths, indentation, and blank lines."
                })
                
            # Check documentation
            documentation = self._calculate_documentation(code)
            if documentation < 0.7:
                suggestions.append({
                    "type": "documentation",
                    "severity": "medium",
                    "description": "Documentation is insufficient. Add docstrings to functions and classes."
                })
                
            # Check test coverage
            test_coverage = self._calculate_test_coverage(code)
            if test_coverage < 0.5:
                suggestions.append({
                    "type": "test_coverage",
                    "severity": "high",
                    "description": "Test coverage is low. Add unit tests for functions."
                })
                
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            suggestions.append({
                "type": "error",
                "severity": "high",
                "description": f"Error analyzing code: {str(e)}"
            })
            
        return suggestions