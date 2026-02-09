import ast
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Union
from pathlib import Path
from knowledge.processor import KnowledgeProcessor, Document, DocumentType

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Enum representing different code pattern types that can be detected."""
    PRINT_STATEMENT = 'print_statement'
    COMPLEX_LOGIC = 'complex_logic'
    INEFFICIENT_LOOP = 'inefficient_loop'
    TYPE_HINTS_MISSING = 'type_hints_missing'
    DOCUMENTATION_MISSING = 'documentation_missing'
    POOR_ERROR_HANDLING = 'poor_error_handling'
    MAGIC_NUMBERS = 'magic_numbers'
    LONG_FUNCTION = 'long_function'
    SECURITY_ISSUE = 'security_issue'
    PERFORMANCE_ISSUE = 'performance_issue'
    BEST_PRACTICE_VIOLATION = 'best_practice_violation'

@dataclass
class CodePattern:
    """Class representing a detected code pattern."""
    pattern_type: PatternType
    line_start: int
    line_end: int
    confidence: float
    description: str
    suggested_fix: str
    code_segment: str
    knowledge_source: Optional[str] = None
    implementation_priority: float = 0.0

class EnhancedCodeGenerator:
    """Class that analyzes code to detect patterns and suggest improvements."""
    
    def __init__(self, knowledge_processor: Optional[KnowledgeProcessor] = None):
        """Initialize the EnhancedCodeGenerator with pattern detectors and knowledge processor."""
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.knowledge_processor = knowledge_processor
        self.implementation_history = {}
    
    def _initialize_pattern_detectors(self) -> Dict[PatternType, Callable]:
        """Initialize all pattern detectors for code analysis."""
        return {
            PatternType.PRINT_STATEMENT: self._detect_print_statements,
            PatternType.COMPLEX_LOGIC: self._detect_complex_logic,
            PatternType.INEFFICIENT_LOOP: self._detect_inefficient_loops,
            PatternType.DOCUMENTATION_MISSING: self._detect_missing_documentation,
            PatternType.POOR_ERROR_HANDLING: self._detect_poor_error_handling,
            PatternType.MAGIC_NUMBERS: self._detect_magic_numbers,
            PatternType.LONG_FUNCTION: self._detect_long_functions,
            PatternType.TYPE_HINTS_MISSING: self._detect_missing_type_hints,
            PatternType.SECURITY_ISSUE: self._detect_security_issues,
            PatternType.PERFORMANCE_ISSUE: self._detect_performance_issues,
            PatternType.BEST_PRACTICE_VIOLATION: self._detect_best_practice_violations
        }
    
    def learn_from_document(self, document_path: Union[str, Path]) -> None:
        """Learn from an external document and update pattern detection capabilities."""
        if not self.knowledge_processor:
            logger.warning("Knowledge processor not initialized. Cannot learn from document.")
            return
            
        document = Document.from_file(document_path)
        result = self.knowledge_processor.process_document(document)
        
        collection_name = self.knowledge_processor.store_documents(result.documents)
        self._update_pattern_detectors_from_knowledge(collection_name)
    
    def _update_pattern_detectors_from_knowledge(self, collection_name: str) -> None:
        """Update pattern detectors based on new knowledge from the vector store."""
        if not self.knowledge_processor:
            return
            
        docs, _ = self.knowledge_processor.query_knowledge(
            "code patterns security performance best practices",
            collection_name=collection_name
        )
        
        for doc in docs:
            self._incorporate_knowledge(doc.page_content)
    
    def _incorporate_knowledge(self, knowledge: str) -> None:
        """Incorporate new knowledge into pattern detectors."""
        if "security" in knowledge.lower():
            self._update_security_patterns(knowledge)
        if "performance" in knowledge.lower():
            self._update_performance_patterns(knowledge)
        if "best practice" in knowledge.lower():
            self._update_best_practice_patterns(knowledge)
    
    def analyze_and_improve_code(self, code: str) -> Dict[str, Any]:
        """Analyze code and suggest improvements based on learned knowledge."""
        analysis_result = self.analyze_code(code)
        
        if not analysis_result["analysis_successful"]:
            return analysis_result
            
        prioritized_patterns = self._prioritize_patterns(analysis_result["pattern_details"])
        improvements = self._generate_improvements(prioritized_patterns)
        
        return {
            "analysis_successful": True,
            "patterns_detected": analysis_result["patterns_detected"],
            "pattern_details": analysis_result["pattern_details"],
            "prioritized_improvements": improvements
        }
    
    def _prioritize_patterns(self, patterns: List[CodePattern]) -> List[CodePattern]:
        """Prioritize patterns based on learned knowledge and implementation history."""
        for pattern in patterns:
            if pattern.pattern_type == PatternType.SECURITY_ISSUE:
                pattern.implementation_priority = 1.0
            elif pattern.pattern_type == PatternType.PERFORMANCE_ISSUE:
                pattern.implementation_priority = 0.8
            elif pattern.pattern_type == PatternType.BEST_PRACTICE_VIOLATION:
                pattern.implementation_priority = 0.6
            else:
                pattern.implementation_priority = 0.4
                
            if pattern.pattern_type in self.implementation_history:
                pattern.implementation_priority *= 0.5
        
        return sorted(patterns, key=lambda x: x.implementation_priority, reverse=True)
    
    def _generate_improvements(self, patterns: List[CodePattern]) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on patterns and learned knowledge."""
        improvements = []
        
        for pattern in patterns:
            improvement = {
                "pattern_type": pattern.pattern_type.value,
                "description": pattern.description,
                "suggested_fix": pattern.suggested_fix,
                "priority": pattern.implementation_priority,
                "lines": (pattern.line_start, pattern.line_end),
                "knowledge_source": pattern.knowledge_source
            }
            
            if self.knowledge_processor:
                docs, _ = self.knowledge_processor.query_knowledge(
                    f"implementation guide for {pattern.pattern_type.value}",
                    k=1
                )
                if docs:
                    improvement["implementation_guide"] = docs[0].page_content
            
            improvements.append(improvement)
        
        return improvements
    
    def implement_improvement(self, code: str, improvement: Dict[str, Any]) -> str:
        """Implement a suggested improvement in the code."""
        pattern_type = improvement["pattern_type"]
        lines = improvement["lines"]
        
        self.implementation_history[pattern_type] = self.implementation_history.get(pattern_type, 0) + 1
        
        # TODO: Implement actual code modification logic
        return code
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for various patterns and return results."""
        try:
            tree = ast.parse(code)
            patterns_detected = []
            pattern_details = []
            
            for pattern_type, detector in self.pattern_detectors.items():
                patterns = detector(tree, code)
                if patterns:
                    patterns_detected.append(pattern_type)
                    pattern_details.extend(patterns)
            
            return {
                "analysis_successful": True,
                "patterns_detected": patterns_detected,
                "pattern_details": pattern_details
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                "analysis_successful": False,
                "error": str(e)
            }
    
    def _detect_print_statements(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect print statements in the code."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                line_start = getattr(node, 'lineno', 0)
                line_end = getattr(node, 'end_lineno', line_start)
                patterns.append(CodePattern(
                    pattern_type=PatternType.PRINT_STATEMENT,
                    line_start=line_start,
                    line_end=line_end,
                    confidence=0.95,
                    description="Print statement detected",
                    suggested_fix="Consider using logging instead of print statements",
                    code_segment=code.split('\n')[line_start-1:line_end]
                ))
        return patterns
    
    def _detect_complex_logic(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect complex logic patterns like deeply nested conditionals."""
        patterns = []
        
        # Add parent references to all nodes
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                nesting_level = self._get_nesting_level(tree, node)
                if nesting_level > 2:
                    line_start = getattr(node, 'lineno', 0)
                    line_end = self._get_node_end_line(node)
                    patterns.append(CodePattern(
                        pattern_type=PatternType.COMPLEX_LOGIC,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.9,
                        description=f"Deeply nested conditional ({nesting_level} levels)",
                        suggested_fix="Consider refactoring into smaller functions or using early returns",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
        return patterns
    
    def _detect_inefficient_loops(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect inefficient loop patterns."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range' and isinstance(node.iter.args[0], ast.Call):
                        if isinstance(node.iter.args[0].func, ast.Name) and node.iter.args[0].func.id == 'len':
                            line_start = getattr(node, 'lineno', 0)
                            line_end = self._get_node_end_line(node)
                            patterns.append(CodePattern(
                                pattern_type=PatternType.INEFFICIENT_LOOP,
                                line_start=line_start,
                                line_end=line_end,
                                confidence=0.9,
                                description="Inefficient loop using range(len())",
                                suggested_fix="Use enumerate() or direct iteration over the sequence",
                                code_segment=code.split('\n')[line_start-1:line_end]
                            ))
            
            if isinstance(node, ast.For):
                parent = self._get_parent_node(tree, node)
                if isinstance(parent, ast.For):
                    line_start = getattr(node, 'lineno', 0)
                    line_end = self._get_node_end_line(node)
                    patterns.append(CodePattern(
                        pattern_type=PatternType.INEFFICIENT_LOOP,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.85,
                        description="Nested loop detected",
                        suggested_fix="Consider using itertools.product() or vectorizing the operation",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
        return patterns
    
    def _detect_missing_type_hints(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect missing type hints in function definitions."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.returns:
                    line_start = getattr(node, 'lineno', 0)
                    line_end = self._get_node_end_line(node)
                    patterns.append(CodePattern(
                        pattern_type=PatternType.TYPE_HINTS_MISSING,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.9,
                        description="Missing return type hint",
                        suggested_fix="Add return type annotation",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
                
                for arg in node.args.args:
                    if not arg.annotation:
                        line_start = getattr(arg, 'lineno', getattr(node, 'lineno', 0))
                        patterns.append(CodePattern(
                            pattern_type=PatternType.TYPE_HINTS_MISSING,
                            line_start=line_start,
                            line_end=line_start,
                            confidence=0.9,
                            description=f"Missing type hint for parameter '{arg.arg}'",
                            suggested_fix=f"Add type annotation for parameter '{arg.arg}'",
                            code_segment=code.split('\n')[line_start-1:line_start]
                        ))
        return patterns
    
    def _detect_missing_documentation(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect missing or incomplete documentation."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    line_start = getattr(node, 'lineno', 0)
                    line_end = self._get_node_end_line(node)
                    patterns.append(CodePattern(
                        pattern_type=PatternType.DOCUMENTATION_MISSING,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.95,
                        description=f"Missing docstring for {node.__class__.__name__}",
                        suggested_fix="Add a docstring following Google style guide",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
        return patterns
    
    def _detect_poor_error_handling(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect poor error handling patterns."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        line_start = getattr(handler, 'lineno', 0)
                        line_end = self._get_node_end_line(handler)
                        patterns.append(CodePattern(
                            pattern_type=PatternType.POOR_ERROR_HANDLING,
                            line_start=line_start,
                            line_end=line_end,
                            confidence=0.95,
                            description="Bare except clause detected",
                            suggested_fix="Specify the exception type(s) to catch",
                            code_segment=code.split('\n')[line_start-1:line_end]
                        ))
        return patterns
    
    def _detect_magic_numbers(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect magic numbers in the code."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, (float, int)) and value not in (0, 1, 2):
                    # Check if the number is used in a constant assignment
                    parent = self._get_parent_node(tree, node)
                    if isinstance(parent, ast.Assign):
                        continue  # Skip if it's a constant assignment
                        
                    line_start = getattr(node, 'lineno', 0)
                    patterns.append(CodePattern(
                        pattern_type=PatternType.MAGIC_NUMBERS,
                        line_start=line_start,
                        line_end=line_start,
                        confidence=0.9,
                        description=f"Magic number detected: {value}",
                        suggested_fix="Define the number as a named constant with clear meaning",
                        code_segment=code.split('\n')[line_start-1:line_start]
                    ))
        return patterns
    
    def _detect_long_functions(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect long functions and functions with too many parameters."""
        patterns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                line_start = getattr(node, 'lineno', 0)
                line_end = self._get_node_end_line(node)
                
                param_count = len(node.args.args)
                if param_count > 5:
                    patterns.append(CodePattern(
                        pattern_type=PatternType.LONG_FUNCTION,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.9,
                        description=f"Function has too many parameters ({param_count})",
                        suggested_fix="Consider using a data class or dictionary for parameters",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
                
                function_lines = line_end - line_start + 1
                if function_lines > 20:
                    patterns.append(CodePattern(
                        pattern_type=PatternType.LONG_FUNCTION,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.85,
                        description=f"Function is too long ({function_lines} lines)",
                        suggested_fix="Break down into smaller functions",
                        code_segment=code.split('\n')[line_start-1:line_end]
                    ))
        return patterns
    
    def _get_nesting_level(self, tree: ast.AST, node: ast.If) -> int:
        """Calculate the nesting level of an if statement."""
        level = 1  # Start at 1 since we're already in an if statement
        current = node
        
        # Walk up the tree to find parent if statements
        while True:
            parent = self._get_parent_node(tree, current)
            if parent is None or not isinstance(parent, ast.If):
                break
            level += 1
            current = parent
        
        return level
    
    def _get_node_end_line(self, node: ast.AST) -> int:
        """Get the end line number of a node."""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno
        return getattr(node, 'lineno', 0)
    
    def _get_parent_node(self, tree: ast.AST, target: ast.AST) -> Optional[ast.AST]:
        """Get the parent node of a target node in the AST."""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child == target:
                    return node
        return None
    
    def _detect_security_issues(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect security-related issues in the code."""
        patterns = []
        # TODO: Implement security issue detection using learned knowledge
        return patterns
    
    def _detect_performance_issues(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect performance-related issues in the code."""
        patterns = []
        # TODO: Implement performance issue detection using learned knowledge
        return patterns
    
    def _detect_best_practice_violations(self, tree: ast.AST, code: str) -> List[CodePattern]:
        """Detect violations of best practices in the code."""
        patterns = []
        # TODO: Implement best practice violation detection using learned knowledge
        return patterns 