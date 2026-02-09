import os
import ast
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
from src.components.code_analyzer import CodeAnalyzer
from src.components.code_metrics import CodeMetrics

class SelfAnalysis:
    """Advanced self-analysis system for understanding and improving own codebase."""
    
    def __init__(self, base_dir: str):
        """Initialize self-analysis system.
        
        Args:
            base_dir: Base directory containing AI code
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.code_analyzer = CodeAnalyzer()
        self.code_metrics = CodeMetrics()
        
        # Initialize dependency graph
        self.dependency_graph = nx.DiGraph()
        
    def analyze_self(self) -> Dict[str, Any]:
        """Perform comprehensive self-analysis of the entire codebase."""
        analysis = {
            'components': self._analyze_components(),
            'dependencies': self._analyze_dependencies(),
            'architecture': self._analyze_architecture(),
            'metrics': self._analyze_overall_metrics(),
            'bottlenecks': self._identify_bottlenecks(),
            'improvement_areas': self._identify_improvement_areas()
        }
        return analysis
        
    def _analyze_components(self) -> List[Dict[str, Any]]:
        """Analyze individual components and their roles."""
        components = []
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    # Parse AST
                    try:
                        tree = ast.parse(code)
                        
                        # Extract classes and their roles
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                docstring = ast.get_docstring(node)
                                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                                
                                components.append({
                                    'name': node.name,
                                    'file': str(file.relative_to(self.base_dir)),
                                    'role': docstring,
                                    'methods': methods,
                                    'metrics': self.code_metrics.calculate_metrics(code)
                                })
                    except:
                        self.logger.error(f"Error analyzing {file}")
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        return components
        
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze component dependencies and interactions."""
        dependencies = {}
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    try:
                        tree = ast.parse(code)
                        imports = []
                        
                        # Extract imports
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                imports.extend(n.name for n in node.names)
                            elif isinstance(node, ast.ImportFrom):
                                module = node.module or ''
                                imports.extend(f"{module}.{n.name}" for n in node.names)
                                
                        # Use POSIX path format for consistency
                        file_path = str(file.relative_to(self.base_dir).as_posix())
                        dependencies[file_path] = imports
                        
                        # Update dependency graph
                        for imp in imports:
                            self.dependency_graph.add_edge(
                                file_path,
                                imp
                            )
                    except:
                        self.logger.error(f"Error analyzing dependencies in {file}")
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        return dependencies
        
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze overall system architecture."""
        return {
            'layers': self._identify_layers(),
            'patterns': self._identify_patterns(),
            'coupling': self._analyze_coupling(),
            'cohesion': self._analyze_cohesion()
        }
        
    def _identify_layers(self) -> List[Dict[str, List[str]]]:
        """Identify architectural layers."""
        layers = {
            'interface': [],
            'business_logic': [],
            'data': []
        }
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                relative_path = str(file.relative_to(self.base_dir))
                
                # Classify based on file location and content
                if 'interface' in relative_path:
                    layers['interface'].append(relative_path)
                elif 'components' in relative_path:
                    layers['business_logic'].append(relative_path)
                elif 'data' in relative_path:
                    layers['data'].append(relative_path)
                    
        return layers
        
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify design patterns in use."""
        patterns = []
        
        pattern_signatures = {
            'singleton': self._detect_singleton,
            'factory': self._detect_factory,
            'observer': self._detect_observer,
            'strategy': self._detect_strategy
        }
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    for pattern, detector in pattern_signatures.items():
                        if detector(code):
                            patterns.append({
                                'pattern': pattern,
                                'file': str(file.relative_to(self.base_dir))
                            })
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                        
        return patterns
        
    def _detect_singleton(self, code: str) -> bool:
        """Detect singleton pattern."""
        return '_instance' in code and '__new__' in code
        
    def _detect_factory(self, code: str) -> bool:
        """Detect factory pattern."""
        return 'create' in code and 'return' in code and 'class' in code
        
    def _detect_observer(self, code: str) -> bool:
        """Detect observer pattern."""
        return 'subscribe' in code or 'notify' in code
        
    def _detect_strategy(self, code: str) -> bool:
        """Detect strategy pattern."""
        return '@abstractmethod' in code and 'class' in code
        
    def _analyze_coupling(self) -> float:
        """Analyze coupling between components."""
        if not self.dependency_graph:
            return 0.0
            
        # Calculate coupling using graph metrics
        return nx.density(self.dependency_graph)
        
    def _analyze_cohesion(self) -> Dict[str, float]:
        """Analyze cohesion within components."""
        cohesion_scores = {}
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    try:
                        tree = ast.parse(code)
                        methods = []
                        shared_variables = set()
                        
                        # Find methods and shared variables
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                methods.append(node)
                            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                                shared_variables.add(node.id)
                                
                        # Calculate cohesion score
                        if methods:
                            # Count shared variable usage
                            shared_usage = 0
                            for method in methods:
                                for node in ast.walk(method):
                                    if isinstance(node, ast.Name) and node.id in shared_variables:
                                        shared_usage += 1
                                        
                            cohesion_scores[str(file.relative_to(self.base_dir))] = \
                                shared_usage / (len(methods) * len(shared_variables)) if shared_variables else 0.0
                    except:
                        self.logger.error(f"Error analyzing cohesion in {file}")
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        return cohesion_scores
        
    def _analyze_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall system metrics."""
        total_metrics = {
            'complexity': 0.0,
            'maintainability': 0.0,
            'security': 0.0,
            'style': 0.0,
            'documentation': 0.0,
            'test_coverage': 0.0
        }
        file_count = 0
        
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    metrics = self.code_metrics.calculate_metrics(code)
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    file_count += 1
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        # Calculate averages
        if file_count > 0:
            for key in total_metrics:
                total_metrics[key] /= file_count
                
        return total_metrics
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance and architectural bottlenecks."""
        bottlenecks = []
        
        # Check for highly coupled components
        if self.dependency_graph:
            centrality = nx.degree_centrality(self.dependency_graph)
            for node, score in centrality.items():
                if score > 0.7:  # High coupling threshold
                    bottlenecks.append({
                        'type': 'high_coupling',
                        'component': node,
                        'score': score,
                        'suggestion': 'Consider reducing dependencies'
                    })
                    
        # Check for complex components
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    metrics = self.code_metrics.calculate_metrics(code)
                    if metrics['complexity'] > 0.7:  # High complexity threshold
                        bottlenecks.append({
                            'type': 'high_complexity',
                            'component': str(file.relative_to(self.base_dir)),
                            'score': metrics['complexity'],
                            'suggestion': 'Consider refactoring into smaller components'
                        })
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        return bottlenecks
        
    def _identify_improvement_areas(self) -> List[Dict[str, Any]]:
        """Identify areas for potential improvement."""
        improvements = []
        
        # Analyze metrics for each component
        for file in self.base_dir.rglob('*.py'):
            if file.is_file():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    metrics = self.code_metrics.calculate_metrics(code)
                    suggestions = self.code_metrics.get_improvement_suggestions(code)
                    
                    for suggestion in suggestions:
                        improvements.append({
                            'component': str(file.relative_to(self.base_dir)),
                            'type': suggestion['type'],
                            'severity': suggestion['severity'],
                            'description': suggestion['description'],
                            'current_score': metrics.get(suggestion['type'], 0.0)
                        })
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {str(e)}")
                    
        return improvements 