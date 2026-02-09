"""
Creative Goal Generator - PHASE 3
Generates novel, creative goals using LLM instead of rule-based logic.
"""

import os
import json
import anthropic
import logging
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Goal:
    """A goal for the system to achieve"""
    description: str
    priority: float  # 0-1
    category: str  # 'testing', 'refactoring', 'learning', 'performance', etc
    success_criteria: List[str]
    estimated_cycles: int


class CreativeGoalGenerator:
    """Generates creative, novel goals using Claude"""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        self.model = model
        self.logger = logging.getLogger(__name__)

    def generate_creative_goals(self, current_state: Dict[str, Any], num_goals: int = 5) -> List[Goal]:
        """
        Generate creative goals based on current system state.

        Args:
            current_state: Current metrics, issues, progress
            num_goals: Number of goals to generate

        Returns:
            List of creative goals
        """
        if not self.client:
            self.logger.warning("No API key, returning default goals")
            return self._generate_default_goals(current_state)

        prompt = f"""You are the goal-setting engine for an autonomous AGI system that improves its own code.

CURRENT STATE:
- Test Coverage: {current_state.get('coverage', 0):.1%}
- Avg Complexity: {current_state.get('complexity', 0):.1f}
- Success Rate: {current_state.get('success_rate', 0):.1%}
- Recent Improvements: {current_state.get('recent_improvements', [])}
- Known Issues: {current_state.get('issues', [])}

Generate {num_goals} AMBITIOUS but ACHIEVABLE goals that would significantly improve this codebase.

Be CREATIVE - don't just suggest obvious metrics improvements. Think about:
- Novel refactoring strategies
- Learning cutting-edge techniques
- Architectural improvements
- Performance optimization
- Developer experience enhancements

Return JSON array:
[
  {{
    "description": "Clear, specific goal description",
    "priority": 0.8,
    "category": "performance|testing|refactoring|learning|architecture",
    "success_criteria": ["criterion 1", "criterion 2"],
    "estimated_cycles": 5
  }},
  ...
]

Return ONLY the JSON array."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.9,  # High creativity
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text.strip()

            # Extract JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            goals_data = json.loads(response)

            goals = [
                Goal(
                    description=g['description'],
                    priority=g.get('priority', 0.5),
                    category=g.get('category', 'general'),
                    success_criteria=g.get('success_criteria', []),
                    estimated_cycles=g.get('estimated_cycles', 3)
                )
                for g in goals_data[:num_goals]
            ]

            self.logger.info(f"Generated {len(goals)} creative goals")
            return goals

        except Exception as e:
            self.logger.error(f"Failed to generate creative goals: {e}")
            return self._generate_default_goals(current_state)

    def _generate_default_goals(self, current_state: Dict[str, Any]) -> List[Goal]:
        """Fallback rule-based goals"""
        goals = []

        coverage = current_state.get('coverage', 0)
        if coverage < 0.8:
            goals.append(Goal(
                description=f"Increase test coverage from {coverage:.1%} to 80%",
                priority=0.9,
                category='testing',
                success_criteria=["Coverage >= 80%"],
                estimated_cycles=5
            ))

        return goals
