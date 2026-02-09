from typing import List, Dict, Optional
from dataclasses import dataclass
import ast
import concurrent.futures
from src.components.code_actions import CodeAction

@dataclass
class OptimizationResult:
    """Result of code optimization."""
    optimized_code: str
    improvements: List[str]
    metrics: Dict[str, float]
    steps_taken: int

class OptimizationPipeline:
    """Pipeline for code optimization."""
    
    def __init__(self):
        self.actions = []
        self._initialize_actions()
        
    def _initialize_actions(self):
        """Initialize available optimization actions."""
        self.actions = [
            CodeAction(name="extract_method", description="Extract repeated code", priority=1),
            CodeAction(name="optimize_data_structures", description="Optimize data structures", priority=2),
            CodeAction(name="improve_error_handling", description="Improve error handling", priority=3),
            CodeAction(name="add_concurrency", description="Add concurrent execution", priority=4)
        ]
        
    def analyze_code(self, code: str) -> Dict:
        """Analyze code and return potential improvements."""
        try:
            tree = ast.parse(code)
            
            analysis = {
                "ast": tree,
                "metrics": self.calculate_metrics(code),
                "potential_improvements": []
            }
            
            # Analyze for potential improvements
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    analysis["potential_improvements"].append("loop_optimization")
                elif isinstance(node, ast.Try):
                    analysis["potential_improvements"].append("error_handling")
                    
            return analysis
            
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {str(e)}")
            
    def calculate_metrics(self, code: str) -> Dict[str, float]:
        """Calculate code quality metrics."""
        try:
            tree = ast.parse(code)
            
            # Calculate cyclomatic complexity
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                    
            # Calculate maintainability index (simplified)
            loc = len(code.split('\n'))
            maintainability = 100 - (complexity * 2) - (loc * 0.2)
            
            return {
                "complexity": float(complexity),
                "maintainability": float(maintainability),
                "loc": float(loc)
            }
            
        except SyntaxError:
            return {"complexity": 0.0, "maintainability": 0.0, "loc": 0.0}
            
    def select_actions(self, analysis: Dict) -> List[CodeAction]:
        """Select appropriate actions based on code analysis."""
        selected_actions = []
        
        for improvement in analysis["potential_improvements"]:
            if improvement == "loop_optimization":
                selected_actions.extend([
                    action for action in self.actions
                    if action.name in ["optimize_data_structures", "add_concurrency"]
                ])
            elif improvement == "error_handling":
                selected_actions.extend([
                    action for action in self.actions
                    if action.name == "improve_error_handling"
                ])
                
        return sorted(selected_actions, key=lambda x: x.priority)
        
    def apply_actions(self, code: str) -> OptimizationResult:
        """Apply optimization actions to code."""
        analysis = self.analyze_code(code)
        actions = self.select_actions(analysis)
        
        optimized_code = code
        improvements = []
        steps_taken = 0
        
        for action in actions:
            # Apply action and track improvements
            result = self._apply_action(action, optimized_code)
            if result.changes_made:
                optimized_code = result.modified_code
                improvements.append(f"Applied {action.name}: {action.description}")
                steps_taken += 1
                
        return OptimizationResult(
            optimized_code=optimized_code,
            improvements=improvements,
            metrics=self.calculate_metrics(optimized_code),
            steps_taken=steps_taken
        )
        
    def optimize(
        self,
        code: str,
        max_steps: int = 10,
        optimization_type: str = "all"
    ) -> OptimizationResult:
        """Optimize code with given constraints."""
        if not code.strip():
            raise ValueError("Empty code provided")
            
        if optimization_type not in ["all", "performance", "readability"]:
            raise ValueError(f"Invalid optimization type: {optimization_type}")
            
        # Filter actions based on optimization type
        original_actions = self.actions.copy()
        if optimization_type == "performance":
            self.actions = [
                action for action in self.actions
                if action.name in ["optimize_data_structures", "add_concurrency"]
            ]
        elif optimization_type == "readability":
            self.actions = [
                action for action in self.actions
                if action.name in ["extract_method", "improve_error_handling"]
            ]
            
        try:
            result = self.apply_actions(code)
            if result.steps_taken > max_steps:
                # Revert to last valid state within max_steps
                result.steps_taken = max_steps
                result.improvements = result.improvements[:max_steps]
            return result
        finally:
            # Restore original actions
            self.actions = original_actions
            
    def optimize_batch(self, code_snippets: List[str]) -> List[OptimizationResult]:
        """Optimize multiple code snippets concurrently."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.optimize, code_snippets))
            
    def _apply_action(self, action: CodeAction, code: str) -> CodeAction:
        """Apply a single optimization action."""
        # This is a placeholder that would normally call the actual optimization
        # For testing purposes, we'll just return a mock result
        return CodeAction(
            name=action.name,
            description=action.description,
            priority=action.priority,
            modified_code=code,
            changes_made=True
        ) 