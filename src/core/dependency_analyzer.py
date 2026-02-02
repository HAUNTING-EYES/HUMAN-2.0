#!/usr/bin/env python3
"""
HUMAN 2.0 Dependency Analyzer
Builds import/dependency graphs to understand code relationships.

This enables context-aware code improvements by understanding:
- What files does X import (dependencies)
- What files import X (reverse dependencies / what will break)
- How critical is file X (how many files depend on it)
- Circular dependencies
"""

import ast
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx


class DependencyAnalyzer:
    """
    Analyzes code dependencies using AST and builds dependency graph.

    Enables understanding code relationships for intelligent improvements.
    """

    def __init__(self, root_dir: str = '.'):
        """
        Initialize dependency analyzer.

        Args:
            root_dir: Root directory of the project
        """
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(root_dir).resolve()

        # Dependency graph (directed): file A -> file B means A imports B
        self.graph = nx.DiGraph()

        # Reverse dependencies: file -> set of files that import it
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        # File criticality scores (0-1): how many files depend on this
        self.file_criticality: Dict[str, float] = {}

        # Module path mapping: module.name -> file path
        self.module_to_file: Dict[str, str] = {}

        # File to module mapping: file path -> module.name
        self.file_to_module: Dict[str, str] = {}

        self.logger.info(f"DependencyAnalyzer initialized with root: {self.root_dir}")

    def build_graph(self, target_dirs: List[str]) -> nx.DiGraph:
        """
        Build complete dependency graph for target directories.

        Args:
            target_dirs: List of directories to analyze (relative to root)

        Returns:
            NetworkX DiGraph with dependency relationships
        """
        self.logger.info(f"Building dependency graph for: {target_dirs}")

        # Reset state
        self.graph.clear()
        self.reverse_deps.clear()
        self.file_criticality.clear()
        self.module_to_file.clear()
        self.file_to_module.clear()

        # 1. Find all Python files
        python_files = self._find_python_files(target_dirs)
        self.logger.info(f"Found {len(python_files)} Python files")

        # 2. Build module mapping
        self._build_module_mapping(python_files)

        # 3. Extract imports from each file
        for file_path in python_files:
            self._analyze_file_imports(file_path)

        # 4. Calculate criticality scores
        self._calculate_criticality()

        # 5. Detect circular dependencies
        cycles = self._detect_circular_dependencies()
        if cycles:
            self.logger.warning(f"Detected {len(cycles)} circular dependency groups")

        self.logger.info(f"Dependency graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        return self.graph

    def get_dependencies(self, file_path: str) -> List[str]:
        """
        Get direct dependencies (what this file imports).

        Args:
            file_path: Path to file (absolute or relative to root)

        Returns:
            List of file paths that this file imports
        """
        file_path = self._normalize_path(file_path)

        if file_path not in self.graph:
            self.logger.warning(f"File not in graph: {file_path}")
            return []

        # Get successor nodes (files this imports)
        dependencies = list(self.graph.successors(file_path))
        return dependencies

    def get_reverse_dependencies(self, file_path: str) -> List[str]:
        """
        Get reverse dependencies (what imports this file).

        This tells you what will break if you change this file!

        Args:
            file_path: Path to file (absolute or relative to root)

        Returns:
            List of file paths that import this file
        """
        file_path = self._normalize_path(file_path)

        if file_path not in self.graph:
            self.logger.warning(f"File not in graph: {file_path}")
            return []

        # Get predecessor nodes (files that import this)
        reverse_deps = list(self.graph.predecessors(file_path))
        return reverse_deps

    def get_related_files(self, file_path: str, depth: int = 2) -> List[str]:
        """
        Get files within N hops in dependency graph.

        Args:
            file_path: Path to file (absolute or relative to root)
            depth: Number of hops in graph (default: 2)

        Returns:
            List of related file paths
        """
        file_path = self._normalize_path(file_path)

        if file_path not in self.graph:
            self.logger.warning(f"File not in graph: {file_path}")
            return []

        related = set()

        # BFS to find nodes within depth hops (both directions)
        visited = {file_path}
        current_level = {file_path}

        for _ in range(depth):
            next_level = set()

            for node in current_level:
                # Successors (dependencies)
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        related.add(successor)
                        next_level.add(successor)
                        visited.add(successor)

                # Predecessors (reverse dependencies)
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in visited:
                        related.add(predecessor)
                        next_level.add(predecessor)
                        visited.add(predecessor)

            current_level = next_level

        return list(related)

    def calculate_criticality(self, file_path: str) -> float:
        """
        Calculate criticality score (0-1) based on reverse dependencies.

        Higher score = more files depend on this = more critical

        Args:
            file_path: Path to file (absolute or relative to root)

        Returns:
            Criticality score (0-1)
        """
        file_path = self._normalize_path(file_path)

        if file_path in self.file_criticality:
            return self.file_criticality[file_path]

        return 0.0

    def _find_python_files(self, target_dirs: List[str]) -> List[str]:
        """Find all Python files in target directories."""
        python_files = []

        for target_dir in target_dirs:
            target_path = self.root_dir / target_dir

            if not target_path.exists():
                self.logger.warning(f"Target directory does not exist: {target_path}")
                continue

            # Find all .py files recursively
            for py_file in target_path.rglob('*.py'):
                # Skip test files, __pycache__, etc.
                if any(skip in str(py_file) for skip in ['test_', '__pycache__', '.pytest_cache', 'archived_']):
                    continue

                python_files.append(str(py_file))

        return python_files

    def _build_module_mapping(self, python_files: List[str]):
        """Build mapping between module names and file paths."""
        for file_path in python_files:
            file_path_obj = Path(file_path)

            # Calculate module name relative to root
            try:
                rel_path = file_path_obj.relative_to(self.root_dir)

                # Convert path to module name
                # e.g., src/core/dependency_analyzer.py -> src.core.dependency_analyzer
                module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
                module_name = '.'.join(module_parts)

                self.module_to_file[module_name] = file_path
                self.file_to_module[file_path] = module_name

            except ValueError:
                # File is not relative to root
                self.logger.warning(f"File not relative to root: {file_path}")

    def _analyze_file_imports(self, file_path: str):
        """Analyze imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Parse AST
            tree = ast.parse(code, filename=file_path)

            # Add file to graph
            self.graph.add_node(file_path)

            # Extract imports
            imports = self._extract_imports(tree, file_path)

            # Resolve imports to file paths
            for import_module in imports:
                imported_file = self._resolve_import(import_module, file_path)

                if imported_file:
                    # Add edge: this file imports imported_file
                    self.graph.add_edge(file_path, imported_file)

                    # Track reverse dependency
                    self.reverse_deps[imported_file].add(file_path)

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")

    def _extract_imports(self, tree: ast.AST, file_path: str) -> Set[str]:
        """Extract import statements from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # import foo, bar
                for alias in node.names:
                    imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                # from foo import bar
                if node.module:
                    imports.add(node.module)

                    # Also add submodule if importing specific items
                    for alias in node.names:
                        if not alias.name.startswith('_'):
                            full_module = f"{node.module}.{alias.name}"
                            imports.add(full_module)

        return imports

    def _resolve_import(self, module_name: str, importing_file: str) -> Optional[str]:
        """
        Resolve module name to file path.

        Args:
            module_name: Module name (e.g., 'src.core.dependency_analyzer')
            importing_file: File that's doing the import

        Returns:
            File path or None if can't resolve
        """
        # Try exact match
        if module_name in self.module_to_file:
            return self.module_to_file[module_name]

        # Try with __init__.py
        init_module = f"{module_name}.__init__"
        if init_module in self.module_to_file:
            return self.module_to_file[init_module]

        # Try to find partial matches (for relative imports)
        for known_module, file_path in self.module_to_file.items():
            if known_module.endswith(module_name) or module_name.endswith(known_module):
                return file_path

        # Can't resolve (likely external package)
        return None

    def _calculate_criticality(self):
        """Calculate criticality scores for all files using PageRank-style algorithm."""
        if self.graph.number_of_nodes() == 0:
            return

        # Use PageRank to calculate criticality
        # Higher PageRank = more important in dependency graph
        try:
            pagerank_scores = nx.pagerank(self.graph.reverse())  # Reverse to get importance of being depended on

            # Normalize to 0-1
            if pagerank_scores:
                max_score = max(pagerank_scores.values())
                if max_score > 0:
                    self.file_criticality = {
                        file: score / max_score
                        for file, score in pagerank_scores.items()
                    }
                else:
                    self.file_criticality = {file: 0.0 for file in pagerank_scores}

        except Exception as e:
            self.logger.error(f"Error calculating criticality: {e}")
            # Fallback: use simple in-degree (number of reverse dependencies)
            for node in self.graph.nodes():
                in_degree = self.graph.in_degree(node)
                self.file_criticality[node] = min(1.0, in_degree / 10.0)  # Normalize roughly

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the graph.

        Returns:
            List of cycles (each cycle is a list of file paths)
        """
        try:
            # Find all simple cycles
            cycles = list(nx.simple_cycles(self.graph))

            if cycles:
                self.logger.warning("Circular dependencies detected:")
                for i, cycle in enumerate(cycles, 1):
                    cycle_str = " -> ".join([Path(f).name for f in cycle])
                    self.logger.warning(f"  Cycle {i}: {cycle_str}")

            return cycles

        except Exception as e:
            self.logger.error(f"Error detecting cycles: {e}")
            return []

    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path to absolute path."""
        path = Path(file_path)

        if path.is_absolute():
            return str(path.resolve())
        else:
            return str((self.root_dir / file_path).resolve())

    def get_stats(self) -> Dict[str, any]:
        """Get dependency graph statistics."""
        return {
            'total_files': self.graph.number_of_nodes(),
            'total_dependencies': self.graph.number_of_edges(),
            'average_dependencies_per_file': (
                self.graph.number_of_edges() / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            'most_critical_files': sorted(
                self.file_criticality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'circular_dependencies': len(list(nx.simple_cycles(self.graph)))
        }

    def visualize_dependencies(self, file_path: str, max_depth: int = 2) -> str:
        """
        Generate text visualization of dependencies for a file.

        Args:
            file_path: File to visualize
            max_depth: Maximum depth to show

        Returns:
            Text visualization
        """
        file_path = self._normalize_path(file_path)

        if file_path not in self.graph:
            return f"File not in graph: {file_path}"

        lines = [f"Dependencies for: {Path(file_path).name}"]
        lines.append(f"Criticality: {self.calculate_criticality(file_path):.2f}")
        lines.append("")

        # Show dependencies (what this imports)
        deps = self.get_dependencies(file_path)
        lines.append(f"Imports ({len(deps)}):")
        for dep in deps[:10]:  # Limit to 10
            lines.append(f"  - {Path(dep).name}")
        if len(deps) > 10:
            lines.append(f"  ... and {len(deps) - 10} more")
        lines.append("")

        # Show reverse dependencies (what imports this)
        rev_deps = self.get_reverse_dependencies(file_path)
        lines.append(f"Imported by ({len(rev_deps)}):")
        for rev_dep in rev_deps[:10]:  # Limit to 10
            lines.append(f"  - {Path(rev_dep).name}")
        if len(rev_deps) > 10:
            lines.append(f"  ... and {len(rev_deps) - 10} more")

        return "\n".join(lines)


def main():
    """Test dependency analyzer."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Dependency Analyzer Test")
    print("=" * 70)

    # Initialize
    analyzer = DependencyAnalyzer(root_dir='.')

    # Build graph
    print("\nBuilding dependency graph...")
    graph = analyzer.build_graph(['src'])

    # Print stats
    stats = analyzer.get_stats()
    print(f"\nGraph Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total dependencies: {stats['total_dependencies']}")
    print(f"  Avg dependencies/file: {stats['average_dependencies_per_file']:.2f}")
    print(f"  Circular dependencies: {stats['circular_dependencies']}")

    print(f"\nMost Critical Files:")
    for file_path, criticality in stats['most_critical_files'][:5]:
        print(f"  {Path(file_path).name}: {criticality:.3f}")

    # Example: Visualize dependencies for a specific file
    if stats['total_files'] > 0:
        example_file = list(graph.nodes())[0]
        print(f"\n{analyzer.visualize_dependencies(example_file)}")


if __name__ == "__main__":
    main()
