#!/usr/bin/env python3
"""
HUMAN 2.0 Self-Improvement V2
Enhanced self-improvement with full context and multi-model ensemble.

Major improvements over V1:
1. Full src/ directory support (not just src/components/)
2. Dependency-aware context injection
3. Semantic context from ChromaDB
4. Multi-model ensemble (Claude for code + Llama for reasoning)
5. Auto-test generation for untested files
6. Stricter safety (requires tests)
"""

import os
import sys
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import V1 components we'll reuse
from core.self_improvement_v1 import LLMCodeImprover, SafeCodeTester

# Import new Phase 1 components
from core.dependency_analyzer import DependencyAnalyzer
from core.code_embedder import CodeEmbedder
from core.test_generator import TestGenerator

# Import existing components
from components.code_analyzer import CodeAnalyzer
from components.code_metrics import CodeMetrics

# Load environment
load_dotenv()


class ClaudeCodeImprover:
    """
    Claude 3.5 Sonnet for code improvement (via OpenAI-compatible API).

    Better at code generation than Llama.
    """

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = 'gpt-4'  # Actually Claude 3.5 Sonnet
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    def generate_improvement_with_context(self, context: Dict[str, Any]) -> str:
        """
        Generate improved code using full context from dependencies + ChromaDB.

        Args:
            context: Dict containing:
                - target_code: Code to improve
                - target_file: File path
                - analysis: LLM analysis of issues
                - dependencies: Files this imports
                - reverse_dependencies: Files that import this
                - similar_code: Similar code from ChromaDB
                - past_improvements: Past improvement history
                - external_patterns: Learned patterns from GitHub/web

        Returns:
            Improved code
        """
        # Build comprehensive context prompt
        target_code = context['target_code']
        file_path = context['target_file']
        analysis = context.get('analysis', {})

        # Dependency context
        deps_context = ""
        if context.get('dependencies'):
            deps_context = f"\n\nThis file imports: {', '.join([Path(d).name for d in context['dependencies'][:5]])}"
        if context.get('reverse_dependencies'):
            rev_deps = context['reverse_dependencies']
            deps_context += f"\n\n⚠️ WARNING: {len(rev_deps)} files depend on this (what breaks if you change it):"
            deps_context += f"\n{', '.join([Path(d).name for d in rev_deps[:5]])}"

        # Semantic context from ChromaDB
        semantic_context = ""
        if context.get('similar_code'):
            similar = context['similar_code']
            semantic_context = f"\n\nSimilar code in codebase:"
            for i, sim in enumerate(similar[:2], 1):
                semantic_context += f"\n{i}. {sim['metadata'].get('relative_path', 'unknown')}"

        # Past improvements context
        past_context = ""
        if context.get('past_improvements'):
            past = context['past_improvements']
            if past:
                past_context = f"\n\nPast improvements to this file: {len(past)} times"

        # External patterns context
        external_context = ""
        if context.get('external_patterns'):
            patterns = context['external_patterns']
            if patterns:
                external_context = f"\n\nRelevant patterns from GitHub/web: {len(patterns)} patterns found"

        # Issues context
        issues_str = ""
        if analysis:
            issues_str = "\n".join([
                f"- Bugs: {', '.join(analysis.get('bugs', []))}",
                f"- Performance: {', '.join(analysis.get('performance_issues', []))}",
                f"- Quality: {', '.join(analysis.get('quality_issues', []))}",
                f"- Security: {', '.join(analysis.get('security_issues', []))}"
            ])

        # Build final prompt
        prompt = f"""You are an expert Python developer improving code for a self-improving AGI system.

File: {file_path}

Original Code:
```python
{target_code}
```

Issues Found:
{issues_str}

Context:
{deps_context}
{semantic_context}
{past_context}
{external_context}

Requirements:
1. Fix all identified issues
2. Maintain the same functionality
3. Preserve all imports and dependencies
4. DO NOT break files that depend on this
5. Improve code quality and performance
6. Return ONLY valid Python code, no explanations

Return only the improved code:"""

        return self._call_claude(prompt)

    def _call_claude(self, prompt: str) -> str:
        """Call Claude via OpenAI-compatible API"""
        import requests

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3,
                'max_tokens': 4000  # Increased for larger responses
            }

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=300  # Increased to 5 minutes for large files
            )

            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['message']['content']

            # Extract code from markdown if present
            if '```python' in content:
                content = content.split('```python')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            return content

        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return ""


class EnhancedSafeCodeTester(SafeCodeTester):
    """
    Enhanced tester that auto-generates tests if missing.

    Stricter than V1: fails if no tests unless auto-generated.
    """

    def __init__(self, test_generator: TestGenerator):
        super().__init__()
        self.test_generator = test_generator

    def run_tests(self, file_path: str) -> Tuple[bool, str]:
        """
        Run tests for modified file.

        If no tests exist, auto-generate them first.
        """
        test_file = self._find_test_file(file_path)

        if not test_file:
            self.logger.warning(f"No tests for {file_path}, attempting to auto-generate...")

            # Try to auto-generate tests
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                test_code = self.test_generator.generate_test_stub(file_path, code)

                if test_code:
                    # Validate generated test
                    valid, error = self.test_generator.validate_test(test_code)

                    if valid:
                        # Save test file
                        test_file_path = self.test_generator._get_test_file_path(file_path)
                        test_file_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(test_file_path, 'w', encoding='utf-8') as f:
                            f.write(test_code)

                        self.logger.info(f"Auto-generated test: {test_file_path}")
                        test_file = str(test_file_path)
                    else:
                        self.logger.error(f"Generated test failed validation: {error}")
                        return False, f"No tests and auto-generation failed: {error}"
                else:
                    return False, "No tests and auto-generation failed"

            except Exception as e:
                self.logger.error(f"Error auto-generating tests: {e}")
                return False, f"No tests and auto-generation failed: {e}"

        # Run tests (original implementation)
        return super().run_tests(file_path)


class SelfImprovementV2:
    """
    Enhanced self-improvement system with full context.

    Major improvements over V1:
    - Full src/ directory support (not just src/components/)
    - Dependency-aware improvements
    - Semantic context from ChromaDB
    - Multi-model ensemble (Claude + Llama)
    - Auto-test generation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize V2 self-improvement system.

        Args:
            config: Configuration dict or None to load from file
        """
        self.logger = logging.getLogger(__name__)

        # Load config
        if config is None:
            config_path = Path('config/deep_before_wide.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
                    config = full_config.get('self_improvement', {})
            else:
                config = {}

        self.config = config

        # Configuration
        self.target_dirs = config.get('target_dirs', ['src'])
        self.exclude_patterns = config.get('exclude_patterns', [
            '**/test_*', '**/__pycache__', '**/archived_*', '**/.pytest_cache'
        ])
        self.max_files_per_cycle = config.get('max_files_per_cycle', 5)
        self.require_tests = config.get('require_tests', True)
        self.auto_generate_tests = config.get('auto_generate_tests', True)
        self.min_criticality_for_auto_fix = config.get('min_criticality_for_auto_fix', 0.3)

        # Initialize Phase 1 components
        self.logger.info("Initializing Phase 1 components...")

        self.dependency_analyzer = DependencyAnalyzer()
        self.code_embedder = CodeEmbedder()
        self.test_generator = TestGenerator()

        # Build dependency graph
        self.logger.info(f"Building dependency graph for: {self.target_dirs}")
        self.dependency_graph = self.dependency_analyzer.build_graph(self.target_dirs)

        # Initialize existing components
        self.code_analyzer = CodeAnalyzer()
        self.code_metrics = CodeMetrics()

        # Multi-model ensemble
        self.logger.info("Initializing multi-model ensemble...")
        self.claude_improver = ClaudeCodeImprover()  # For code generation
        self.llama_improver = LLMCodeImprover()  # For reasoning/analysis

        # Enhanced tester with auto-generation
        self.tester = EnhancedSafeCodeTester(self.test_generator)

        # State tracking
        self.improvement_history = []
        self.generation = 0
        self.stats = {
            'total_files_analyzed': 0,
            'total_improvements': 0,
            'successful_improvements': 0,
            'failed_improvements': 0,
            'bugs_fixed': 0,
            'tests_auto_generated': 0
        }

        # Safety settings
        self.backup_dir = Path('backups/self_improvement_v2')
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("SelfImprovementV2 initialized - Enhanced with full context")

    def improve_codebase(self, max_files: Optional[int] = None):
        """
        Improve codebase with full context.

        Args:
            max_files: Maximum files to improve (default: from config)

        Returns:
            Report dict with improvement statistics
        """
        if max_files is None:
            max_files = self.max_files_per_cycle

        self.logger.info(f"Starting self-improvement cycle V2 (Generation {self.generation})")
        self.logger.info(f"Target directories: {self.target_dirs}")

        # 1. Find improvement candidates
        candidates = self._find_improvement_candidates(max_files)

        self.logger.info(f"Found {len(candidates)} files to analyze")

        # 2. Improve each file
        for file_path in candidates:
            self._improve_file(file_path)

        # 3. Generate report
        report = self._generate_report()

        self.generation += 1

        return report

    def improve_file(self, file_path: str) -> Dict[str, Any]:
        """Improve a single file with full context"""
        return self._improve_file(file_path)

    def _improve_file(self, file_path: str) -> Dict[str, Any]:
        """
        Improve a single file with full context.

        This is where the magic happens - combines dependency analysis,
        semantic search, and multi-model LLM improvement.
        """
        self.logger.info(f"Analyzing: {file_path}")

        result = {
            'file': file_path,
            'success': False,
            'changes_made': False,
            'original_metrics': {},
            'improved_metrics': {},
            'issues_found': {},
            'context_used': {},
            'error': None
        }

        try:
            # 1. Read current code
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()

            # 2. Get baseline metrics
            result['original_metrics'] = self.code_metrics.calculate_metrics(original_code)

            # 3. Build comprehensive context
            context = self._build_improvement_context(file_path, original_code)
            result['context_used'] = {
                'dependencies': len(context.get('dependencies', [])),
                'reverse_dependencies': len(context.get('reverse_dependencies', [])),
                'similar_code_found': len(context.get('similar_code', [])),
                'has_tests': context.get('has_tests', False)
            }

            # 4. Analyze with Llama (reasoning)
            self.logger.info(f"Analyzing {file_path} with Llama for reasoning...")
            llm_analysis = self.llama_improver.analyze_code(original_code, file_path)
            result['issues_found'] = llm_analysis

            # 5. Check if improvement is needed
            severity = llm_analysis.get('severity', 'low')
            can_auto_fix = llm_analysis.get('can_auto_fix', False)

            # Check criticality score
            criticality = self.dependency_analyzer.calculate_criticality(file_path)

            if severity == 'low' and criticality < self.min_criticality_for_auto_fix:
                self.logger.info(f"No significant issues in {file_path} (criticality: {criticality:.2f})")
                result['success'] = True
                return result

            # 6. Generate improvement with Claude (code generation)
            self.logger.info(f"Generating improvements for {file_path} with Claude...")
            context['analysis'] = llm_analysis
            improved_code = self.claude_improver.generate_improvement_with_context(context)

            if not improved_code or improved_code == original_code:
                self.logger.warning(f"No improvement generated for {file_path}")
                result['error'] = "LLM did not generate improved code"
                return result

            # 7. Validate syntax
            valid, syntax_error = self.tester.validate_syntax(improved_code)
            if not valid:
                self.logger.error(f"Improved code has syntax error: {syntax_error}")
                result['error'] = f"Syntax error: {syntax_error}"
                self.stats['failed_improvements'] += 1
                return result

            # 8. Backup original
            self._backup_file(file_path, original_code)

            # 9. Apply improvement
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)

            # 10. Run tests (auto-generates if missing)
            tests_pass, test_output = self.tester.run_tests(file_path)

            if not tests_pass:
                self.logger.error(f"Tests failed after improvement: {file_path}")
                # Rollback
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)

                result['error'] = f"Tests failed: {test_output[:200]}"
                self.stats['failed_improvements'] += 1
                return result

            # 11. Calculate new metrics
            result['improved_metrics'] = self.code_metrics.calculate_metrics(improved_code)

            # 12. Record improvement in ChromaDB for meta-learning
            self.code_embedder.record_improvement(
                file_path=file_path,
                before_code=original_code,
                after_code=improved_code,
                success=True,
                metrics=result['improved_metrics'],
                strategy='context_aware_v2'
            )

            # 13. Success!
            result['success'] = True
            result['changes_made'] = True

            self.stats['successful_improvements'] += 1
            self.stats['bugs_fixed'] += len(llm_analysis.get('bugs', []))

            self.logger.info(f"✅ Successfully improved {file_path}")

            # Save to history
            self.improvement_history.append({
                'timestamp': datetime.now().isoformat(),
                'generation': self.generation,
                'file': file_path,
                'result': result
            })

            return result

        except Exception as e:
            self.logger.error(f"Error improving {file_path}: {e}")
            result['error'] = str(e)
            self.stats['failed_improvements'] += 1
            return result

    def _build_improvement_context(self, file_path: str, code: str) -> Dict[str, Any]:
        """
        Build comprehensive context for improvement.

        This is the key innovation in V2 - we provide the LLM with:
        - Dependency context (what imports what)
        - Semantic context (similar code)
        - Historical context (past improvements)
        - External context (learned patterns)
        """
        context = {
            'target_file': file_path,
            'target_code': code
        }

        try:
            # Dependency context (optimized for performance)
            context['dependencies'] = self.dependency_analyzer.get_dependencies(file_path)
            context['reverse_dependencies'] = self.dependency_analyzer.get_reverse_dependencies(file_path)
            context['related_files'] = self.dependency_analyzer.get_related_files(file_path, depth=1)  # Reduced from 2 to 1
            context['criticality'] = self.dependency_analyzer.calculate_criticality(file_path)

            # Semantic context from ChromaDB
            improvement_ctx = self.code_embedder.get_improvement_context(file_path, code)
            context.update(improvement_ctx)

            # Test context
            context['has_tests'] = self.test_generator.has_tests(file_path)
            context['test_file'] = self.test_generator.find_test_file(file_path)

        except Exception as e:
            self.logger.error(f"Error building context for {file_path}: {e}")

        return context

    def _find_improvement_candidates(self, max_files: int) -> List[str]:
        """
        Find files that could benefit from improvement.

        Prioritizes by:
        1. High criticality (many reverse dependencies)
        2. High complexity
        3. Low test coverage
        """
        candidates = []

        # Scan all target directories
        for target_dir in self.target_dirs:
            target_path = Path(target_dir)

            if not target_path.exists():
                self.logger.warning(f"Target directory not found: {target_path}")
                continue

            for py_file in target_path.rglob('*.py'):
                # Skip based on exclude patterns
                should_skip = False
                for pattern in self.exclude_patterns:
                    if Path(str(py_file)).match(pattern):
                        should_skip = True
                        break

                if should_skip:
                    continue

                # Skip __init__ files
                if py_file.name == '__init__.py':
                    continue

                candidates.append(str(py_file))

        # Prioritize candidates
        prioritized = self._prioritize_candidates(candidates)

        return prioritized[:max_files]

    def _prioritize_candidates(self, candidates: List[str]) -> List[str]:
        """Prioritize files for improvement"""

        def get_priority_score(file_path: str) -> float:
            """Calculate priority score (higher = more priority)"""
            score = 0.0

            try:
                # 1. Criticality (0-1): More reverse deps = higher priority
                criticality = self.dependency_analyzer.calculate_criticality(file_path)
                score += criticality * 50  # Weight: 50

                # 2. Complexity: More complex = higher priority
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                metrics = self.code_metrics.calculate_metrics(code)
                complexity = metrics.get('complexity', 0.5)
                score += complexity * 30  # Weight: 30

                # 3. Test coverage: No tests = higher priority
                has_tests = self.test_generator.has_tests(file_path)
                if not has_tests:
                    score += 20  # Weight: 20

            except Exception as e:
                self.logger.error(f"Error calculating priority for {file_path}: {e}")
                score = 0.0

            return score

        # Sort by priority (descending)
        prioritized = sorted(candidates, key=get_priority_score, reverse=True)

        return prioritized

    def _backup_file(self, file_path: str, code: str):
        """Backup file before modification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{Path(file_path).stem}_{timestamp}.py"

        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(code)

        self.logger.debug(f"Backed up {file_path} to {backup_path}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate improvement cycle report"""
        return {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats.copy(),
            'improvements': self.improvement_history[-self.max_files_per_cycle:],
            'config': {
                'target_dirs': self.target_dirs,
                'max_files_per_cycle': self.max_files_per_cycle,
                'auto_generate_tests': self.auto_generate_tests
            }
        }


def main():
    """Test SelfImprovementV2"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Self-Improvement V2")
    print("=" * 70)

    # Initialize
    improver = SelfImprovementV2()

    # Run improvement cycle
    print("\nRunning improvement cycle...")
    report = improver.improve_codebase(max_files=3)

    # Print results
    print(f"\nResults:")
    print(f"  Generation: {report['generation']}")
    print(f"  Successful: {report['stats']['successful_improvements']}")
    print(f"  Failed: {report['stats']['failed_improvements']}")
    print(f"  Bugs fixed: {report['stats']['bugs_fixed']}")

    # Save report
    report_path = Path(f"reports/self_improvement_v2_gen_{report['generation']}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
