#!/usr/bin/env python3
"""
HUMAN 2.0 Meta-Learner
Learns which improvement strategies work best and optimizes system parameters.

This is the "learning to learn" component - the system optimizes its own
improvement strategies based on past success/failure patterns.

Key capabilities:
1. Record improvement outcomes (success, metrics, strategy used)
2. Analyze patterns in successful vs failed improvements
3. Select best strategy for each file based on similarity to past improvements
4. Optimize system parameters (thresholds, strategies, etc.)
5. Track improvement over time (is the system getting better at improving itself?)
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.code_embedder import CodeEmbedder

# Load environment
load_dotenv()


class MetaLearner:
    """
    Meta-learning system that optimizes improvement strategies.

    Learns from improvement history to:
    - Select best strategy for each file
    - Optimize system parameters
    - Identify successful patterns
    - Avoid failure patterns
    """

    def __init__(self):
        """Initialize meta-learner"""
        self.logger = logging.getLogger(__name__)

        # Initialize code embedder for history search
        self.code_embedder = CodeEmbedder()

        # State tracking
        self.improvement_history = []
        self.strategy_success_rates = defaultdict(list)
        self.parameter_history = []

        # Load existing history if available
        self._load_history()

        self.logger.info("MetaLearner initialized")

    def record_improvement_outcome(self, improvement: Dict[str, Any]):
        """
        Record outcome of improvement attempt.

        Args:
            improvement: Dict with:
                - file_path: Path to file
                - before_code: Code before improvement
                - after_code: Code after improvement
                - success: Boolean success flag
                - strategy: Strategy used
                - before_metrics: Metrics before
                - after_metrics: Metrics after
                - context: Context used for improvement
        """
        # Add timestamp
        improvement['timestamp'] = datetime.now().isoformat()

        # Add to history
        self.improvement_history.append(improvement)

        # Track strategy success
        strategy = improvement.get('strategy', 'default')
        success = improvement.get('success', False)
        self.strategy_success_rates[strategy].append(1.0 if success else 0.0)

        # Store in ChromaDB for similarity search
        self.code_embedder.record_improvement(
            file_path=improvement['file_path'],
            before_code=improvement['before_code'],
            after_code=improvement['after_code'],
            success=success,
            metrics=improvement.get('after_metrics', {}),
            strategy=strategy
        )

        self.logger.info(f"Recorded improvement: {improvement['file_path']} (success={success}, strategy={strategy})")

        # Periodically save history
        if len(self.improvement_history) % 10 == 0:
            self._save_history()

    def select_best_strategy(
        self,
        file_path: str,
        code: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Select best improvement strategy based on past success.

        Uses similarity search to find similar past improvements
        and recommends the strategy that worked best.

        Args:
            file_path: Path to file being improved
            code: Current code
            analysis: Analysis of issues

        Returns:
            Strategy name
        """
        self.logger.info(f"Selecting best strategy for: {file_path}")

        try:
            # Find similar past improvements
            similar_improvements = self.code_embedder.find_similar_code(code, n_results=10)

            if not similar_improvements:
                self.logger.info("No similar improvements found, using default strategy")
                return 'context_aware_v2'

            # Get improvements from ChromaDB improvements collection
            improvement_context = self.code_embedder.get_improvement_context(file_path, code)
            past_improvements = improvement_context.get('past_improvements', [])

            if not past_improvements:
                self.logger.info("No past improvements found, using default strategy")
                return 'context_aware_v2'

            # Calculate success rates for each strategy
            strategy_scores = defaultdict(list)

            for imp in past_improvements:
                metadata = imp.get('metadata', {})
                strategy = metadata.get('strategy', 'default')
                success = metadata.get('success', 'false') == 'true'  # ChromaDB stores as string

                strategy_scores[strategy].append(1.0 if success else 0.0)

            # Select strategy with highest success rate
            if strategy_scores:
                best_strategy = max(
                    strategy_scores.items(),
                    key=lambda x: np.mean(x[1])
                )[0]

                success_rate = np.mean(strategy_scores[best_strategy])

                self.logger.info(f"Selected strategy: {best_strategy} (success rate: {success_rate:.2f})")

                return best_strategy
            else:
                return 'context_aware_v2'

        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}")
            return 'context_aware_v2'  # Fallback

    def optimize_improvement_params(self) -> Dict[str, Any]:
        """
        Optimize improvement parameters based on history.

        Analyzes successful vs failed improvements to find optimal:
        - Complexity threshold for auto-fix
        - Criticality threshold
        - Preferred strategies
        - Patterns to avoid

        Returns:
            Dict of optimized parameters
        """
        self.logger.info("Optimizing improvement parameters...")

        if len(self.improvement_history) < 10:
            self.logger.warning("Not enough history for optimization (need 10+, have {})".format(
                len(self.improvement_history)
            ))
            return self._get_default_params()

        # Separate successes and failures
        successes = [imp for imp in self.improvement_history if imp.get('success')]
        failures = [imp for imp in self.improvement_history if not imp.get('success')]

        self.logger.info(f"Analyzing {len(successes)} successes and {len(failures)} failures")

        params = {}

        # 1. Calculate overall success rate
        params['success_rate'] = len(successes) / len(self.improvement_history)

        # 2. Find preferred strategies
        params['preferred_strategies'] = self._extract_successful_strategies(successes)

        # 3. Find patterns to avoid
        params['avoid_patterns'] = self._extract_failure_patterns(failures)

        # 4. Find optimal complexity threshold
        params['optimal_complexity_threshold'] = self._find_optimal_threshold(
            successes, failures, 'complexity'
        )

        # 5. Find optimal criticality threshold
        params['optimal_criticality_threshold'] = self._find_optimal_threshold(
            successes, failures, 'criticality'
        )

        # 6. Calculate improvement trend (is system getting better?)
        params['improvement_trend'] = self._calculate_improvement_trend()

        # Save parameters
        self.parameter_history.append({
            'timestamp': datetime.now().isoformat(),
            'params': params
        })

        self._save_history()

        self.logger.info(f"Optimized parameters: success_rate={params['success_rate']:.2f}")

        return params

    def _extract_successful_strategies(self, successes: List[Dict[str, Any]]) -> List[str]:
        """Extract strategies that worked best"""
        strategy_counts = defaultdict(int)

        for imp in successes:
            strategy = imp.get('strategy', 'default')
            strategy_counts[strategy] += 1

        # Sort by count
        sorted_strategies = sorted(
            strategy_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [s[0] for s in sorted_strategies[:5]]  # Top 5

    def _extract_failure_patterns(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns common in failures"""
        patterns = []

        # Analyze failure reasons
        error_types = defaultdict(int)

        for imp in failures:
            error = imp.get('error', '')

            if 'syntax' in error.lower():
                error_types['syntax_error'] += 1
            elif 'test' in error.lower():
                error_types['test_failure'] += 1
            elif 'timeout' in error.lower():
                error_types['timeout'] += 1
            else:
                error_types['other'] += 1

        # Convert to patterns
        for error_type, count in error_types.items():
            if count > 2:  # Only if happened multiple times
                patterns.append(f"avoid_{error_type}")

        return patterns

    def _find_optimal_threshold(
        self,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        metric_name: str
    ) -> float:
        """
        Find optimal threshold for a metric.

        Uses ROC-like analysis to find threshold that maximizes success.

        Args:
            successes: Successful improvements
            failures: Failed improvements
            metric_name: Metric to optimize (e.g., 'complexity')

        Returns:
            Optimal threshold value
        """
        # Extract metric values
        success_values = []
        failure_values = []

        for imp in successes:
            context = imp.get('context', {})
            if metric_name in context:
                success_values.append(context[metric_name])

        for imp in failures:
            context = imp.get('context', {})
            if metric_name in context:
                failure_values.append(context[metric_name])

        if not success_values or not failure_values:
            # Not enough data, return default
            return 0.5

        # Try different thresholds
        thresholds = np.linspace(0.0, 1.0, 20)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            # Count correct predictions
            true_positives = sum(1 for v in success_values if v >= threshold)
            true_negatives = sum(1 for v in failure_values if v < threshold)

            total = len(success_values) + len(failure_values)
            score = (true_positives + true_negatives) / total

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def _calculate_improvement_trend(self) -> str:
        """
        Calculate if system is getting better at improving itself.

        Returns:
            'improving', 'stable', or 'declining'
        """
        if len(self.improvement_history) < 20:
            return 'insufficient_data'

        # Split history into recent and older
        recent = self.improvement_history[-10:]
        older = self.improvement_history[-20:-10]

        recent_success_rate = sum(1 for imp in recent if imp.get('success')) / len(recent)
        older_success_rate = sum(1 for imp in older if imp.get('success')) / len(older)

        diff = recent_success_rate - older_success_rate

        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters when not enough history"""
        return {
            'success_rate': 0.0,
            'preferred_strategies': ['context_aware_v2'],
            'avoid_patterns': [],
            'optimal_complexity_threshold': 0.5,
            'optimal_criticality_threshold': 0.3,
            'improvement_trend': 'insufficient_data'
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get meta-learning statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            'total_improvements': len(self.improvement_history),
            'total_successes': sum(1 for imp in self.improvement_history if imp.get('success')),
            'total_failures': sum(1 for imp in self.improvement_history if not imp.get('success')),
            'success_rate': 0.0,
            'strategy_success_rates': {},
            'recent_trend': 'unknown'
        }

        if self.improvement_history:
            stats['success_rate'] = stats['total_successes'] / len(self.improvement_history)

        # Calculate success rates per strategy
        for strategy, results in self.strategy_success_rates.items():
            if results:
                stats['strategy_success_rates'][strategy] = {
                    'count': len(results),
                    'success_rate': np.mean(results)
                }

        # Get recent trend
        if len(self.improvement_history) >= 20:
            stats['recent_trend'] = self._calculate_improvement_trend()

        return stats

    def _load_history(self):
        """Load improvement history from file"""
        history_file = Path('data/meta_learning/history.json')

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                self.improvement_history = data.get('improvements', [])
                self.parameter_history = data.get('parameters', [])

                self.logger.info(f"Loaded {len(self.improvement_history)} historical improvements")

                # Rebuild strategy success rates
                for imp in self.improvement_history:
                    strategy = imp.get('strategy', 'default')
                    success = imp.get('success', False)
                    self.strategy_success_rates[strategy].append(1.0 if success else 0.0)

            except Exception as e:
                self.logger.error(f"Error loading history: {e}")

    def _save_history(self):
        """Save improvement history to file"""
        history_file = Path('data/meta_learning/history.json')
        history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                'improvements': self.improvement_history,
                'parameters': self.parameter_history,
                'last_updated': datetime.now().isoformat()
            }

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Saved history: {len(self.improvement_history)} improvements")

        except Exception as e:
            self.logger.error(f"Error saving history: {e}")


def main():
    """Test meta-learner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Meta-Learner Test")
    print("=" * 70)

    # Initialize
    meta = MetaLearner()

    # Print statistics
    stats = meta.get_statistics()
    print(f"\nCurrent Statistics:")
    print(f"  Total improvements: {stats['total_improvements']}")
    print(f"  Success rate: {stats['success_rate']:.2f}")
    print(f"  Recent trend: {stats['recent_trend']}")

    if stats['strategy_success_rates']:
        print(f"\n  Strategy Success Rates:")
        for strategy, data in stats['strategy_success_rates'].items():
            print(f"    {strategy}: {data['success_rate']:.2f} ({data['count']} attempts)")

    # If enough history, optimize parameters
    if stats['total_improvements'] >= 10:
        print("\nOptimizing parameters...")
        params = meta.optimize_improvement_params()

        print(f"\nOptimized Parameters:")
        print(f"  Overall success rate: {params['success_rate']:.2f}")
        print(f"  Preferred strategies: {params['preferred_strategies']}")
        print(f"  Optimal complexity threshold: {params['optimal_complexity_threshold']:.2f}")
        print(f"  Improvement trend: {params['improvement_trend']}")


if __name__ == "__main__":
    main()
