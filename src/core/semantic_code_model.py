"""
Semantic Code Model - Deep understanding of code beyond syntax.

PHASE 2 ENHANCEMENT: Gives HUMAN 2.0 true semantic understanding of codebase.
Instead of just syntax analysis, understands:
- What each component does
- How components relate
- Data flow patterns
- Architectural patterns
- Business logic

This is the foundation for intelligent improvements.
"""

import ast
import os
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic
import logging


@dataclass
class ComponentUnderstanding:
    """Semantic understanding of a code component"""
    file_path: str
    component_type: str  # 'module', 'class', 'function'
    component_name: str

    # Semantic understanding (LLM-generated)
    purpose: str  # What does it do?
    responsibilities: List[str]  # What is it responsible for?
    role_in_system: str  # How does it fit in the larger system?
    key_abstractions: List[str]  # What abstractions does it provide?

    # Technical details
    public_interface: List[str]  # Public methods/functions
    dependencies: List[str]  # What it depends on
    dependents: List[str]  # What depends on it

    # Data flow
    inputs: List[str]  # What data does it consume?
    outputs: List[str]  # What data does it produce?
    side_effects: List[str]  # What side effects?

    # Quality
    complexity_score: float
    maintainability_score: float

    # Metadata
    last_analyzed: datetime
    confidence_score: float  # How confident are we in this understanding?


@dataclass
class DataFlow:
    """Represents data flowing through the system"""
    data_name: str
    source: str  # Which component produces it
    transformations: List[Tuple[str, str]]  # (component, transformation_description)
    sinks: List[str]  # Which components consume it
    data_type: str  # Type of data


@dataclass
class ArchitecturalPattern:
    """Identified architectural pattern"""
    pattern_name: str
    components_involved: List[str]
    description: str
    confidence: float


class SemanticCodeModel:
    """
    Semantic understanding of codebase using LLM-powered analysis.

    This goes beyond AST parsing to understand:
    - What code actually does (semantics)
    - How pieces fit together (architecture)
    - Why code exists (business logic)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize semantic code model.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            model: Claude model to use for analysis
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

        # Storage
        self.component_understanding: Dict[str, ComponentUnderstanding] = {}
        self.data_flows: List[DataFlow] = []
        self.architectural_patterns: List[ArchitecturalPattern] = []

        self.logger = logging.getLogger(__name__)

    def analyze_component(self, file_path: str, code: str) -> ComponentUnderstanding:
        """
        Deep semantic analysis of a code component.

        Args:
            file_path: Path to file
            code: Source code

        Returns:
            Semantic understanding of the component
        """
        self.logger.info(f"Semantically analyzing: {file_path}")

        # Parse AST for technical details
        technical_details = self._extract_technical_details(code)

        # Use LLM for semantic understanding
        semantic_understanding = self._understand_semantics_via_llm(file_path, code)

        # Combine technical + semantic
        understanding = ComponentUnderstanding(
            file_path=file_path,
            component_type='module',  # Could detect class/function
            component_name=Path(file_path).stem,

            # From LLM
            purpose=semantic_understanding.get('purpose', 'Unknown'),
            responsibilities=semantic_understanding.get('responsibilities', []),
            role_in_system=semantic_understanding.get('role_in_system', 'Unknown'),
            key_abstractions=semantic_understanding.get('key_abstractions', []),
            inputs=semantic_understanding.get('inputs', []),
            outputs=semantic_understanding.get('outputs', []),
            side_effects=semantic_understanding.get('side_effects', []),

            # From AST
            public_interface=technical_details['public_interface'],
            dependencies=technical_details['dependencies'],
            dependents=[],  # Will be computed later

            # Quality metrics (would come from analyzer)
            complexity_score=0.0,
            maintainability_score=0.0,

            last_analyzed=datetime.now(),
            confidence_score=semantic_understanding.get('confidence', 0.8)
        )

        # Store
        self.component_understanding[file_path] = understanding

        return understanding

    def _extract_technical_details(self, code: str) -> Dict[str, Any]:
        """Extract technical details from code using AST"""

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                'public_interface': [],
                'dependencies': []
            }

        public_interface = []
        dependencies = set()

        for node in ast.walk(tree):
            # Find public functions/classes
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):
                    public_interface.append(node.name)

            # Find imports (dependencies)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module)

        return {
            'public_interface': public_interface,
            'dependencies': list(dependencies)
        }

    def _understand_semantics_via_llm(self, file_path: str, code: str) -> Dict[str, Any]:
        """
        Use Claude to understand what the code actually does.

        This is where the magic happens - going from syntax to semantics.
        """

        if not self.client:
            self.logger.warning("No Claude client available, returning default understanding")
            return {
                'purpose': 'Unknown (no API key)',
                'responsibilities': [],
                'role_in_system': 'Unknown',
                'key_abstractions': [],
                'inputs': [],
                'outputs': [],
                'side_effects': [],
                'confidence': 0.0
            }

        prompt = f"""Analyze this Python code and provide deep semantic understanding.

FILE: {file_path}

CODE:
```python
{code[:3000]}  # Limit to first 3000 chars
```

Provide analysis in JSON format:
{{
    "purpose": "One sentence: what does this code do?",
    "responsibilities": ["responsibility 1", "responsibility 2", ...],
    "role_in_system": "How does this fit in the larger system?",
    "key_abstractions": ["abstraction 1", "abstraction 2", ...],
    "inputs": ["what data/parameters does it consume?"],
    "outputs": ["what data does it produce?"],
    "side_effects": ["file I/O", "network calls", "state mutations", etc],
    "confidence": 0.8
}}

Focus on WHAT it does and WHY, not HOW.
Return ONLY the JSON, no other text.
"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for analysis
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()

            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            understanding = json.loads(response_text)

            self.logger.info(f"Semantic analysis complete: {understanding.get('purpose', 'Unknown')}")

            return understanding

        except Exception as e:
            self.logger.error(f"Failed to understand semantics via LLM: {e}")
            return {
                'purpose': 'Analysis failed',
                'responsibilities': [],
                'role_in_system': 'Unknown',
                'key_abstractions': [],
                'inputs': [],
                'outputs': [],
                'side_effects': [],
                'confidence': 0.0
            }

    def build_semantic_knowledge_graph(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Build a semantic knowledge graph of the codebase.

        Args:
            file_paths: List of files to analyze

        Returns:
            Knowledge graph with semantic relationships
        """
        self.logger.info(f"Building semantic knowledge graph for {len(file_paths)} files")

        # Analyze all components
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    self.analyze_component(file_path, code)
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")

        # Build relationships
        self._build_dependency_relationships()
        self._detect_data_flows()
        self._detect_architectural_patterns()

        return {
            'components': len(self.component_understanding),
            'data_flows': len(self.data_flows),
            'patterns': len(self.architectural_patterns)
        }

    def _build_dependency_relationships(self):
        """Build dependency graph and identify dependents"""

        for file_path, understanding in self.component_understanding.items():
            # Find dependents (files that depend on this one)
            for other_file, other_understanding in self.component_understanding.items():
                if file_path == other_file:
                    continue

                # If other file imports this one
                if any(dep in file_path for dep in other_understanding.dependencies):
                    understanding.dependents.append(other_file)

    def _detect_data_flows(self):
        """Detect how data flows through the system"""

        # This is a simplified version - could be much more sophisticated
        for file_path, understanding in self.component_understanding.items():
            for output in understanding.outputs:
                # Find where this data goes
                sinks = []
                for other_file, other_understanding in self.component_understanding.items():
                    if file_path == other_file:
                        continue
                    if output in other_understanding.inputs:
                        sinks.append(other_file)

                if sinks:
                    flow = DataFlow(
                        data_name=output,
                        source=file_path,
                        transformations=[],
                        sinks=sinks,
                        data_type='unknown'
                    )
                    self.data_flows.append(flow)

    def _detect_architectural_patterns(self):
        """Detect architectural patterns in the codebase"""

        # Look for common patterns

        # Pattern: Agent pattern (many files with "agent" in name/purpose)
        agent_components = [
            file_path for file_path, understanding in self.component_understanding.items()
            if 'agent' in file_path.lower() or 'agent' in understanding.purpose.lower()
        ]

        if len(agent_components) >= 3:
            self.architectural_patterns.append(ArchitecturalPattern(
                pattern_name='Multi-Agent System',
                components_involved=agent_components,
                description='System uses multiple autonomous agents for coordination',
                confidence=0.9
            ))

        # Pattern: Event-driven
        event_components = [
            file_path for file_path, understanding in self.component_understanding.items()
            if 'event' in file_path.lower() or 'event' in understanding.purpose.lower()
        ]

        if event_components:
            self.architectural_patterns.append(ArchitecturalPattern(
                pattern_name='Event-Driven Architecture',
                components_involved=event_components,
                description='System uses event-based communication',
                confidence=0.8
            ))

    def get_component_purpose(self, file_path: str) -> str:
        """Get the purpose of a component"""
        understanding = self.component_understanding.get(file_path)
        return understanding.purpose if understanding else "Unknown"

    def find_related_components(self, file_path: str) -> List[str]:
        """Find components related to this one"""
        understanding = self.component_understanding.get(file_path)
        if not understanding:
            return []

        related = set()
        related.update(understanding.dependencies)
        related.update(understanding.dependents)

        return list(related)

    def save(self, output_path: str):
        """Save semantic model to file"""
        data = {
            'components': {
                path: asdict(understanding)
                for path, understanding in self.component_understanding.items()
            },
            'data_flows': [asdict(flow) for flow in self.data_flows],
            'patterns': [asdict(pattern) for pattern in self.architectural_patterns]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Semantic model saved to {output_path}")

    def load(self, input_path: str):
        """Load semantic model from file"""
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Load components
        for path, comp_dict in data.get('components', {}).items():
            comp_dict['last_analyzed'] = datetime.fromisoformat(comp_dict['last_analyzed'])
            self.component_understanding[path] = ComponentUnderstanding(**comp_dict)

        # Load data flows
        for flow_dict in data.get('data_flows', []):
            self.data_flows.append(DataFlow(**flow_dict))

        # Load patterns
        for pattern_dict in data.get('patterns', []):
            self.architectural_patterns.append(ArchitecturalPattern(**pattern_dict))

        self.logger.info(f"Semantic model loaded from {input_path}")
