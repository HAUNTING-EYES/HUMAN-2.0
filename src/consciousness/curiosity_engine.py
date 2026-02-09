"""
Curiosity Engine - PHASE 3
Drives exploration and learning of novel techniques.
"""

import random
import logging
from typing import List, Dict, Any


class CuriosityEngine:
    """
    Drives curiosity-based exploration.
    Makes the AI try novel improvements and learn new techniques.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exploration_topics = [
            "async/await patterns",
            "design patterns",
            "performance optimization",
            "functional programming",
            "type systems",
            "testing strategies",
            "architectural patterns",
            "code generation",
            "meta-programming",
            "concurrency patterns"
        ]
        self.explored_topics = set()

    def get_curiosity_topics(self, n: int = 2) -> List[str]:
        """
        Get topics to explore based on curiosity.

        Args:
            n: Number of topics

        Returns:
            List of topics to explore
        """
        # Prioritize unexplored topics
        unexplored = [t for t in self.exploration_topics if t not in self.explored_topics]

        if unexplored:
            topics = random.sample(unexplored, min(n, len(unexplored)))
        else:
            # Re-explore with new angle
            topics = random.sample(self.exploration_topics, n)

        self.logger.info(f"Curiosity topics: {topics}")
        return topics

    def mark_explored(self, topic: str):
        """Mark a topic as explored"""
        self.explored_topics.add(topic)

    def should_explore(self, success_rate: float) -> bool:
        """
        Decide if we should explore (vs exploit).

        Lower success rate -> more exploration
        Higher success rate -> more exploitation

        Args:
            success_rate: Current improvement success rate

        Returns:
            True if should explore
        """
        # Epsilon-greedy: explore 20% of the time, more if failing
        explore_probability = 0.2 + (1 - success_rate) * 0.3

        should = random.random() < explore_probability

        if should:
            self.logger.info(f"Curiosity triggered! (p={explore_probability:.2f})")

        return should
