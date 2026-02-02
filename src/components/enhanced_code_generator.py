import ast
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Union
from pathlib import Path
from knowledge.processor import KnowledgeProcessor, Document, DocumentType

logger = logging.getLogger(__name__)

class PatternType(Enum):
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
    pattern_type: PatternType
    line_start: int
    line_end: int
    confidence: float
    description: str
    suggested_fix: str
    code_segment: str
    knowledge_source: Optional[str] = None
    implementation_priority: float = 0.0

class ASTAnalyzer:
    def __init__(self, tree: ast.AST, code: str):
        self.tree = tree
        self.code = code
        self.code_lines = code.split('\n')
        self._add_parent_references()
    
    def _add_parent_references(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
    
    def get_parent_node(self, target: ast.AST) -> Optional[ast.AST]:
        return getattr(target, 'parent', None)
    
    def get_node_end_line(self, node: ast.AST) -> int:
        return getattr(node, 'end_lineno', getattr(node, 'lineno', 0))
    
    def get_code_segment(self, line_start: int, line_end: int) -> str:
        return self.code_lines[line_start-1:line_end]
    
    def get_nesting_level(self, node: ast.If) -> int:
        level = 1
        current = node
        
        while True:
            parent = self.get_parent_node(current)
            if parent is None or not isinstance(parent, ast.If):
                break
            level += 1
            current = parent
        
        return level

class PatternDetector:
    PRIORITY_MAP = {
        PatternType.SECURITY_ISSUE: 1.0,
        PatternType.PERFORMANCE_ISSUE: 0.8,
        PatternType.BEST_PRACTICE_VIOLATION: 0.6,
    }
    DEFAULT_PRIORITY = 0.4
    
    def __init__(self, analyzer: ASTAnalyzer):
        self.analyzer = analyzer
    
    def detect_print_statements(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if self._is_print_call(node):
                patterns.append(self._create_print_pattern(node))
        return patterns
    
    def _is_print_call(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'print')
    
    def _create_print_pattern(self, node: ast.AST) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.PRINT_STATEMENT,
            line_start=line_start,
            line_end=line_end,
            confidence=0.95,
            description="Print statement detected",
            suggested_fix="Consider using logging instead of print statements",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def detect_complex_logic(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.If):
                nesting_level = self.analyzer.get_nesting_level(node)
                if nesting_level > 2:
                    patterns.append(self._create_complex_logic_pattern(node, nesting_level))
        return patterns
    
    def _create_complex_logic_pattern(self, node: ast.If, nesting_level: int) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.COMPLEX_LOGIC,
            line_start=line_start,
            line_end=line_end,
            confidence=0.9,
            description=f"Deeply nested conditional ({nesting_level} levels)",
            suggested_fix="Consider refactoring into smaller functions or using early returns",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def detect_inefficient_loops(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.For):
                if self._is_range_len_pattern(node):
                    patterns.append(self._create_range_len_pattern(node))
                elif self._is_nested_loop(node):
                    patterns.append(self._create_nested_loop_pattern(node))
        return patterns
    
    def _is_range_len_pattern(self, node: ast.For) -> bool:
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            return False
        if node.iter.func.id != 'range' or not node.iter.args:
            return False
        first_arg = node.iter.args[0]
        return (isinstance(first_arg, ast.Call) and 
                isinstance(first_arg.func, ast.Name) and 
                first_arg.func.id == 'len')
    
    def _is_nested_loop(self, node: ast.For) -> bool:
        parent = self.analyzer.get_parent_node(node)
        return isinstance(parent, ast.For)
    
    def _create_range_len_pattern(self, node: ast.For) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.INEFFICIENT_LOOP,
            line_start=line_start,
            line_end=line_end,
            confidence=0.9,
            description="Inefficient loop using range(len())",
            suggested_fix="Use enumerate() or direct iteration over the sequence",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def _create_nested_loop_pattern(self, node: ast.For) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.INEFFICIENT_LOOP,
            line_start=line_start,
            line_end=line_end,
            confidence=0.85,
            description="Nested loop detected",
            suggested_fix="Consider using itertools.product() or vectorizing the operation",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def detect_missing_type_hints(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.FunctionDef):
                patterns.extend(self._check_function_type_hints(node))
        return patterns
    
    def _check_function_type_hints(self, node: ast.FunctionDef) -> List[CodePattern]:
        patterns = []
        if not node.returns:
            patterns.append(self._create_missing_return_type_pattern(node))
        patterns.extend(self._check_parameter_type_hints(node))
        return patterns
    
    def _create_missing_return_type_pattern(self, node: ast.FunctionDef) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.TYPE_HINTS_MISSING,
            line_start=line_start,
            line_end=line_end,
            confidence=0.9,
            description="Missing return type hint",
            suggested_fix="Add return type annotation",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def _check_parameter_type_hints(self, node: ast.FunctionDef) -> List[CodePattern]:
        patterns = []
        for arg in node.args.args:
            if not arg.annotation:
                patterns.append(self._create_missing_param_type_pattern(arg, node))
        return patterns
    
    def _create_missing_param_type_pattern(self, arg: ast.arg, node: ast.FunctionDef) -> CodePattern:
        line_start = getattr(arg, 'lineno', getattr(node, 'lineno', 0))
        return CodePattern(
            pattern_type=PatternType.TYPE_HINTS_MISSING,
            line_start=line_start,
            line_end=line_start,
            confidence=0.9,
            description=f"Missing type hint for parameter '{arg.arg}'",
            suggested_fix=f"Add type annotation for parameter '{arg.arg}'",
            code_segment=self.analyzer.get_code_segment(line_start, line_start)
        )
    
    def detect_missing_documentation(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    patterns.append(self._create_missing_doc_pattern(node))
        return patterns
    
    def _create_missing_doc_pattern(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        return CodePattern(
            pattern_type=PatternType.DOCUMENTATION_MISSING,
            line_start=line_start,
            line_end=line_end,
            confidence=0.95,
            description=f"Missing docstring for {node.__class__.__name__}",
            suggested_fix="Add a docstring following Google style guide",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def detect_poor_error_handling(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.Try):
                patterns.extend(self._check_exception_handlers(node))
        return patterns
    
    def _check_exception_handlers(self, node: ast.Try) -> List[CodePattern]:
        patterns = []
        for handler in node.handlers:
            if handler.type is None:
                patterns.append(self._create_bare_except_pattern(handler))
        return patterns
    
    def _create_bare_except_pattern(self, handler: ast.ExceptHandler) -> CodePattern:
        line_start = getattr(handler, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(handler)
        return CodePattern(
            pattern_type=PatternType.POOR_ERROR_HANDLING,
            line_start=line_start,
            line_end=line_end,
            confidence=0.95,
            description="Bare except clause detected",
            suggested_fix="Specify the exception type(s) to catch",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )
    
    def detect_magic_numbers(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.Constant):
                if self._is_magic_number(node):
                    patterns.append(self._create_magic_number_pattern(node))
        return patterns
    
    def _is_magic_number(self, node: ast.Constant) -> bool:
        value = node.value
        if not isinstance(value, (float, int)) or value in (0, 1, 2):
            return False
        parent = self.analyzer.get_parent_node(node)
        return not isinstance(parent, ast.Assign)
    
    def _create_magic_number_pattern(self, node: ast.Constant) -> CodePattern:
        line_start = getattr(node, 'lineno', 0)
        return CodePattern(
            pattern_type=PatternType.MAGIC_NUMBERS,
            line_start=line_start,
            line_end=line_start,
            confidence=0.9,
            description=f"Magic number detected: {node.value}",
            suggested_fix="Define the number as a named constant with clear meaning",
            code_segment=self.analyzer.get_code_segment(line_start, line_start)
        )
    
    def detect_long_functions(self) -> List[CodePattern]:
        patterns = []
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, ast.FunctionDef):
                patterns.extend(self._check_function_complexity(node))
        return patterns
    
    def _check_function_complexity(self, node: ast.FunctionDef) -> List[CodePattern]:
        patterns = []
        line_start = getattr(node, 'lineno', 0)
        line_end = self.analyzer.get_node_end_line(node)
        
        param_count = len(node.args.args)
        if param_count > 5:
            patterns.append(self._create_too_many_params_pattern(node, param_count, line_start, line_end))
        
        function_lines = line_end - line_start + 1
        if function_lines > 20:
            patterns.append(self._create_long_function_pattern(node, function_lines, line_start, line_end))
        
        return patterns
    
    def _create_too_many_params_pattern(self, node: ast.FunctionDef, param_count: int, 
                                       line_start: int, line_end: int) -> CodePattern:
        return CodePattern(
            pattern_type=PatternType.LONG_FUNCTION,
            line_start=line_start,
            line_end=line_end,
            confidence=0.9,
            description=f"Function has too many parameters ({param_count})",
            suggested_fix="Consider using a data class or dictionary for parameters",
            code_segment=self.analyzer.get_code_segment(line_start, line_end)
        )