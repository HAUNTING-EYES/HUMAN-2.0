#!/usr/bin/env python3
"""
HUMAN 2.0 - Analyzer Agent
Analyzes code for complexity, quality issues, and improvement opportunities.

Responsibilities:
- AST-based code analysis
- Complexity metrics (cyclomatic, cognitive)
- Code smell detection
- Anti-pattern identification
- Dependency impact analysis
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources


@dataclass
class CodeSmell:
    """Detected code smell"""
    smell_type: str
    severity: str  # "low", "medium", "high"
    location: str  # e.g., "line 42"
    description: str
    suggestion: str


@dataclass
class AnalysisResult:
    """Result of code analysis"""
    file_path: str
    timestamp: datetime

    # Metrics
    lines_of_code: int
    complexity: float  # Cyclomatic complexity
    cognitive_complexity: float
    maintainability_index: float

    # Dependencies
    dependencies: List[str]
    reverse_dependencies: List[str]
    criticality_score: float

    # Issues
    code_smells: List[CodeSmell]
    anti_patterns: List[str]

    # Recommendations
    refactoring_opportunities: List[str]
    priority_score: float  # 0-1: how urgently needs improvement


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for code analysis and quality assessment.

    Subscribes to:
    - cycle_started: Analyze files in cycle plan
    - improvement_needed: On-demand analysis

    Publishes:
    - code_analyzed: Analysis results ready
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Analyzer Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Analysis configuration
        self.config = {
            'max_complexity': 10,
            'min_maintainability': 65,
            'max_function_length': 50,
            'max_class_length': 300
        }

        self.logger.info(f"AnalyzerAgent initialized with config: {self.config}")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.CYCLE_STARTED, self.on_cycle_started, self.name)
        self.event_bus.subscribe('improvement_needed', self.on_improvement_needed, self.name)
        self.logger.info(f"Subscribed to: {EventTypes.CYCLE_STARTED}, improvement_needed")

    async def on_cycle_started(self, event: Event):
        """Handle cycle started event"""
        self.logger.info(f"Cycle started: {event.data.get('cycle_number')}")

        # Get files to analyze from cycle plan
        files_to_analyze = event.data.get('files_to_analyze', [])

        for file_path in files_to_analyze:
            task = {
                'type': 'analyze',
                'file_path': file_path,
                'reason': 'cycle_plan'
            }
            result = await self.execute_task(task)

    async def on_improvement_needed(self, event: Event):
        """Handle improvement needed event"""
        file_path = event.data.get('file_path')
        self.logger.info(f"Improvement needed for: {file_path}")

        task = {
            'type': 'analyze',
            'file_path': file_path,
            'reason': 'improvement_requested'
        }
        result = await self.execute_task(task)

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Analyze a code file.

        Args:
            task: Task with file_path to analyze

        Returns:
            Analysis result
        """
        file_path = task['file_path']
        self.logger.info(f"Analyzing: {file_path}")

        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")

        # Parse AST
        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")

        # Perform analysis
        analysis = AnalysisResult(
            file_path=file_path,
            timestamp=datetime.now(),
            lines_of_code=self._count_loc(code),
            complexity=self._calculate_complexity(tree),
            cognitive_complexity=self._calculate_cognitive_complexity(tree),
            maintainability_index=self._calculate_maintainability(tree, code),
            dependencies=self.resources.get_dependencies(file_path),
            reverse_dependencies=self.resources.get_reverse_dependencies(file_path),
            criticality_score=self.resources.get_file_criticality(file_path),
            code_smells=self._detect_code_smells(tree, code),
            anti_patterns=self._detect_anti_patterns(tree, code),
            refactoring_opportunities=self._identify_refactoring_opportunities(tree, code),
            priority_score=0.0  # Will be calculated
        )

        # Calculate priority score
        analysis.priority_score = self._calculate_priority(analysis)

        # Publish code_analyzed event
        await self.publish_event(
            EventTypes.CODE_ANALYZED,
            {
                'file_path': file_path,
                'analysis': asdict(analysis),
                'reason': task.get('reason', 'unknown')
            },
            EventPriority.NORMAL
        )

        return {
            'success': True,
            'file_path': file_path,
            'analysis': asdict(analysis)
        }

    def _count_loc(self, code: str) -> int:
        """Count lines of code (excluding blank lines and comments)"""
        lines = code.split('\n')
        loc = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                loc += 1
        return loc

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> float:
        """Calculate cognitive complexity (simpler approximation)"""
        cognitive = 0
        nesting_level = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                cognitive += (1 + nesting_level)
                nesting_level += 1
            elif isinstance(node, ast.FunctionDef):
                nesting_level = 0  # Reset for new function

        return cognitive

    def _calculate_maintainability(self, tree: ast.AST, code: str) -> float:
        """
        Calculate maintainability index (0-100).
        Higher is better.
        """
        # Simplified version of Microsoft's maintainability index
        loc = self._count_loc(code)
        complexity = self._calculate_complexity(tree)

        # Halstead volume approximation (simplified)
        operators = len([n for n in ast.walk(tree) if isinstance(n, ast.operator)])
        operands = len([n for n in ast.walk(tree) if isinstance(n, ast.Name)])
        volume = (operators + operands) * 0.5 if (operators + operands) > 0 else 1

        # Formula: 171 - 5.2*ln(V) - 0.23*G - 16.2*ln(L)
        # Simplified: weight by complexity and LOC
        mi = max(0, min(100, 100 - (complexity * 2) - (loc / 10)))

        return mi

    def _detect_code_smells(self, tree: ast.AST, code: str) -> List[CodeSmell]:
        """Detect common code smells"""
        smells = []

        for node in ast.walk(tree):
            # Long function
            if isinstance(node, ast.FunctionDef):
                func_lines = len(ast.unparse(node).split('\n'))
                if func_lines > self.config['max_function_length']:
                    smells.append(CodeSmell(
                        smell_type="long_function",
                        severity="medium",
                        location=f"line {node.lineno}",
                        description=f"Function '{node.name}' has {func_lines} lines",
                        suggestion=f"Break down into smaller functions (<{self.config['max_function_length']} lines)"
                    ))

            # Long class
            if isinstance(node, ast.ClassDef):
                class_lines = len(ast.unparse(node).split('\n'))
                if class_lines > self.config['max_class_length']:
                    smells.append(CodeSmell(
                        smell_type="long_class",
                        severity="medium",
                        location=f"line {node.lineno}",
                        description=f"Class '{node.name}' has {class_lines} lines",
                        suggestion=f"Split into smaller classes (<{self.config['max_class_length']} lines)"
                    ))

            # Too many parameters
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 5:
                    smells.append(CodeSmell(
                        smell_type="too_many_parameters",
                        severity="low",
                        location=f"line {node.lineno}",
                        description=f"Function '{node.name}' has {param_count} parameters",
                        suggestion="Consider using a parameter object or dataclass"
                    ))

            # Duplicated code (simple check for similar strings)
            # This would need more sophisticated analysis in production

        return smells

    def _detect_anti_patterns(self, tree: ast.AST, code: str) -> List[str]:
        """Detect anti-patterns"""
        anti_patterns = []

        for node in ast.walk(tree):
            # God object (class with too many methods)
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    anti_patterns.append(f"God Object: Class '{node.name}' has {len(methods)} methods")

            # Circular imports would be detected via dependency graph
            # Check with shared resources

        # Check for circular dependencies
        cycles = self._detect_circular_imports()
        if cycles:
            anti_patterns.append(f"Circular dependencies detected: {len(cycles)} cycles")

        return anti_patterns

    def _detect_circular_imports(self) -> List[List[str]]:
        """Detect circular import dependencies"""
        import networkx as nx

        graph = self.resources.get_dependency_graph()

        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except:
            return []

    def _identify_refactoring_opportunities(self, tree: ast.AST, code: str) -> List[str]:
        """Identify refactoring opportunities"""
        opportunities = []

        complexity = self._calculate_complexity(tree)
        if complexity > self.config['max_complexity']:
            opportunities.append(f"High complexity ({complexity}) - Consider refactoring")

        maintainability = self._calculate_maintainability(tree, code)
        if maintainability < self.config['min_maintainability']:
            opportunities.append(f"Low maintainability ({maintainability:.1f}) - Simplify code")

        # Check for missing tests
        file_path = getattr(tree, 'filename', '')
        if file_path and not file_path.startswith('test_'):
            # Check if test file exists
            test_file = Path(file_path).parent.parent / 'tests' / f'test_{Path(file_path).name}'
            if not test_file.exists():
                opportunities.append("Missing unit tests - Generate tests")

        return opportunities

    def _calculate_priority(self, analysis: AnalysisResult) -> float:
        """
        Calculate priority score (0-1) for improvement.
        Higher = more urgent.
        """
        priority = 0.0

        # Weight by complexity
        if analysis.complexity > self.config['max_complexity']:
            priority += 0.3

        # Weight by maintainability
        if analysis.maintainability_index < self.config['min_maintainability']:
            priority += 0.2

        # Weight by criticality (how many files depend on this)
        priority += analysis.criticality_score * 0.2

        # Weight by number of code smells
        priority += min(0.2, len(analysis.code_smells) * 0.05)

        # Weight by anti-patterns
        priority += min(0.1, len(analysis.anti_patterns) * 0.05)

        return min(1.0, priority)

    def validate_output(self, output: Any) -> bool:
        """Validate analysis output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success'):
            return True  # Errors are valid outputs

        # Check required fields
        analysis = output.get('analysis')
        if not analysis:
            return False

        required_fields = ['file_path', 'complexity', 'maintainability_index', 'priority_score']
        for field in required_fields:
            if field not in analysis:
                return False

        return True


if __name__ == "__main__":
    # Test analyzer agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_analyzer():
        """Test analyzer agent"""
        # Setup
        bus = EventBus()
        resources = SharedResources()
        agent = AnalyzerAgent("analyzer", bus, resources)

        # Build dependency graph first
        resources.dependency_analyzer.build_graph(['src'])

        # Test file analysis
        test_file = 'src/core/event_bus.py'
        task = {
            'type': 'analyze',
            'file_path': test_file,
            'reason': 'test'
        }

        result = await agent.execute_task(task)
        print(f"\nAnalysis Result:")
        print(f"  File: {result.get('file_path')}")
        print(f"  Success: {result.get('success')}")

        if result.get('success'):
            analysis = result['analysis']
            print(f"  LOC: {analysis['lines_of_code']}")
            print(f"  Complexity: {analysis['complexity']}")
            print(f"  Maintainability: {analysis['maintainability_index']:.1f}")
            print(f"  Criticality: {analysis['criticality_score']:.2f}")
            print(f"  Priority: {analysis['priority_score']:.2f}")
            print(f"  Code Smells: {len(analysis['code_smells'])}")
            print(f"  Anti-Patterns: {len(analysis['anti_patterns'])}")
            print(f"  Refactoring Opportunities: {len(analysis['refactoring_opportunities'])}")

        # Test event subscription
        await bus.publish(create_event(
            'improvement_needed',
            {'file_path': test_file},
            'test'
        ))

        await asyncio.sleep(0.5)  # Let events process

        # Get agent status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Name: {status['name']}")
        print(f"  Status: {status['status']}")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")
        print(f"  Success Rate: {status['metrics']['success_rate']:.1%}")

    asyncio.run(test_analyzer())
