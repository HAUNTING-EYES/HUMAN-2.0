"""
Monte Carlo Tree Search (MCTS) Planner
Uses MCTS for strategic planning and decision-making in the AGI system.
"""

import logging
import random
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Types of actions the AGI can take"""
    IMPROVE_CODE = "improve_code"
    LEARN_TOPIC = "learn_topic"
    RUN_TESTS = "run_tests"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    DOCUMENT = "document"
    ANALYZE = "analyze"


@dataclass
class State:
    """Represents a state in the planning space"""
    cycle_number: int
    test_coverage: float
    avg_complexity: float
    success_rate: float
    active_goals: List[str]
    completed_goals: List[str]
    learned_topics: List[str]

    def clone(self) -> 'State':
        """Create a copy of this state"""
        return State(
            cycle_number=self.cycle_number,
            test_coverage=self.test_coverage,
            avg_complexity=self.avg_complexity,
            success_rate=self.success_rate,
            active_goals=self.active_goals.copy(),
            completed_goals=self.completed_goals.copy(),
            learned_topics=self.learned_topics.copy()
        )

    def is_goal_achieved(self) -> bool:
        """Check if main goals are achieved"""
        return (
            self.test_coverage >= 0.8 and
            self.avg_complexity <= 8.0 and
            self.success_rate >= 0.9
        )


@dataclass
class Action:
    """Represents an action the AGI can take"""
    action_type: ActionType
    target: str  # File path, topic, etc.
    priority: float = 0.5
    estimated_cost: int = 1  # Cycles needed
    estimated_benefit: float = 0.5  # Expected improvement

    def __repr__(self):
        return f"{self.action_type.value}({self.target})"


@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: State
    action: Optional[Action] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[Action] = field(default_factory=list)

    @property
    def average_reward(self) -> float:
        """Average reward from this node"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.average_reward

        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        return self.state.is_goal_achieved()


class MCTSPlanner:
    """Monte Carlo Tree Search planner for AGI decision-making"""

    def __init__(self, simulation_budget: int = 1000, exploration_constant: float = 1.41):
        """
        Initialize MCTS planner.

        Args:
            simulation_budget: Number of MCTS simulations to run
            exploration_constant: UCB1 exploration constant (higher = more exploration)
        """
        self.logger = logging.getLogger(__name__)
        self.simulation_budget = simulation_budget
        self.exploration_constant = exploration_constant

    def plan(self, initial_state: State, available_actions: List[Action]) -> List[Action]:
        """
        Plan the best sequence of actions using MCTS.

        Args:
            initial_state: Current state
            available_actions: Available actions to choose from

        Returns:
            Optimal sequence of actions
        """
        self.logger.info(f"Starting MCTS planning with {self.simulation_budget} simulations")

        # Create root node
        root = MCTSNode(
            state=initial_state,
            untried_actions=available_actions.copy()
        )

        # Run MCTS simulations
        for i in range(self.simulation_budget):
            # Selection + Expansion
            node = self._tree_policy(root)

            # Simulation
            reward = self._default_policy(node.state)

            # Backpropagation
            self._backpropagate(node, reward)

            if (i + 1) % 100 == 0:
                self.logger.debug(f"Completed {i + 1} simulations")

        # Extract best action sequence
        best_sequence = self._extract_best_sequence(root)

        self.logger.info(f"MCTS planning complete. Best sequence: {best_sequence}")
        return best_sequence

    def _tree_policy(self, node: MCTSNode) -> MCTSNode:
        """
        Select or expand a node using UCB1.

        Returns the node to simulate from.
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # Expand: try an untried action
                return self._expand(node)
            else:
                # Select: choose best child by UCB1
                node = self._best_child(node)

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action"""
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Apply action to get new state
        new_state = self._apply_action(node.state, action)

        # Create child node
        child = MCTSNode(
            state=new_state,
            action=action,
            parent=node,
            untried_actions=self._get_available_actions(new_state)
        )

        node.children.append(child)
        return child

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB1 score"""
        return max(node.children, key=lambda c: c.ucb1(self.exploration_constant))

    def _default_policy(self, state: State) -> float:
        """
        Simulate a random playout from the given state.

        Returns the final reward.
        """
        current_state = state.clone()
        depth = 0
        max_depth = 10  # Limit simulation depth

        while not current_state.is_goal_achieved() and depth < max_depth:
            # Choose random action
            actions = self._get_available_actions(current_state)
            if not actions:
                break

            action = random.choice(actions)
            current_state = self._apply_action(current_state, action)
            depth += 1

        # Calculate reward
        return self._calculate_reward(current_state)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward through the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _extract_best_sequence(self, root: MCTSNode, max_length: int = 5) -> List[Action]:
        """Extract the best action sequence from the tree"""
        sequence = []
        node = root

        while node.children and len(sequence) < max_length:
            # Choose child with most visits (most robust)
            node = max(node.children, key=lambda c: c.visits)
            if node.action:
                sequence.append(node.action)

        return sequence

    def _apply_action(self, state: State, action: Action) -> State:
        """
        Simulate applying an action to a state.

        This is a simplified model - in reality would be more complex.
        """
        new_state = state.clone()
        new_state.cycle_number += action.estimated_cost

        # Simulate action effects
        if action.action_type == ActionType.IMPROVE_CODE:
            # Might reduce complexity
            new_state.avg_complexity *= (1 - 0.1 * action.estimated_benefit)
            new_state.success_rate = min(1.0, new_state.success_rate + 0.05 * action.estimated_benefit)

        elif action.action_type == ActionType.LEARN_TOPIC:
            # Improve success rate by learning
            new_state.learned_topics.append(action.target)
            new_state.success_rate = min(1.0, new_state.success_rate + 0.1 * action.estimated_benefit)

        elif action.action_type == ActionType.RUN_TESTS:
            # Improve coverage
            new_state.test_coverage = min(1.0, new_state.test_coverage + 0.15 * action.estimated_benefit)

        elif action.action_type == ActionType.REFACTOR:
            # Reduce complexity significantly
            new_state.avg_complexity *= (1 - 0.2 * action.estimated_benefit)

        elif action.action_type == ActionType.OPTIMIZE:
            # Improve all metrics slightly
            new_state.avg_complexity *= (1 - 0.05 * action.estimated_benefit)
            new_state.success_rate = min(1.0, new_state.success_rate + 0.03 * action.estimated_benefit)

        elif action.action_type == ActionType.DOCUMENT:
            # Small improvement to success rate
            new_state.success_rate = min(1.0, new_state.success_rate + 0.02 * action.estimated_benefit)

        elif action.action_type == ActionType.ANALYZE:
            # Improve understanding (success rate)
            new_state.success_rate = min(1.0, new_state.success_rate + 0.05 * action.estimated_benefit)

        # Add some randomness to simulate uncertainty
        new_state.test_coverage += random.uniform(-0.02, 0.02)
        new_state.avg_complexity += random.uniform(-0.5, 0.5)
        new_state.success_rate += random.uniform(-0.02, 0.02)

        # Keep values in valid ranges
        new_state.test_coverage = max(0.0, min(1.0, new_state.test_coverage))
        new_state.avg_complexity = max(1.0, min(50.0, new_state.avg_complexity))
        new_state.success_rate = max(0.0, min(1.0, new_state.success_rate))

        return new_state

    def _get_available_actions(self, state: State) -> List[Action]:
        """Get available actions for a state"""
        actions = []

        # Always can improve code
        if state.avg_complexity > 8:
            actions.append(Action(
                ActionType.REFACTOR,
                "high_complexity_files",
                priority=0.9,
                estimated_cost=2,
                estimated_benefit=0.8
            ))

        # Can run tests if coverage is low
        if state.test_coverage < 0.8:
            actions.append(Action(
                ActionType.RUN_TESTS,
                "missing_tests",
                priority=0.8,
                estimated_cost=1,
                estimated_benefit=0.7
            ))

        # Can learn if success rate is low
        if state.success_rate < 0.9:
            actions.append(Action(
                ActionType.LEARN_TOPIC,
                "best_practices",
                priority=0.7,
                estimated_cost=2,
                estimated_benefit=0.6
            ))

        # Can always analyze
        actions.append(Action(
            ActionType.ANALYZE,
            "codebase",
            priority=0.5,
            estimated_cost=1,
            estimated_benefit=0.4
        ))

        # Can improve specific files
        actions.append(Action(
            ActionType.IMPROVE_CODE,
            "core_files",
            priority=0.6,
            estimated_cost=1,
            estimated_benefit=0.5
        ))

        return actions

    def _calculate_reward(self, state: State) -> float:
        """Calculate reward for a state"""
        # Multi-objective reward function
        coverage_reward = state.test_coverage
        complexity_reward = 1.0 - (state.avg_complexity / 50.0)  # Lower is better
        success_reward = state.success_rate

        # Weighted combination
        reward = (
            0.4 * coverage_reward +
            0.3 * complexity_reward +
            0.3 * success_reward
        )

        # Bonus for achieving goals
        if state.is_goal_achieved():
            reward += 1.0

        # Penalty for taking too many cycles
        cycle_penalty = state.cycle_number * 0.01
        reward -= cycle_penalty

        return reward


class StrategicPlanner:
    """High-level strategic planner using MCTS"""

    def __init__(self, mcts_planner: MCTSPlanner):
        self.logger = logging.getLogger(__name__)
        self.mcts = mcts_planner

    def create_strategic_plan(self, current_state: Dict[str, Any],
                             available_files: List[str],
                             available_topics: List[str]) -> Dict[str, Any]:
        """
        Create a strategic plan using MCTS.

        Args:
            current_state: Current system state
            available_files: Files that can be improved
            available_topics: Topics that can be learned

        Returns:
            Strategic plan with recommended actions
        """
        self.logger.info("Creating strategic plan with MCTS")

        # Convert to MCTS state
        state = State(
            cycle_number=current_state.get('cycle_number', 0),
            test_coverage=current_state.get('test_coverage', 0.0),
            avg_complexity=current_state.get('avg_complexity', 15.0),
            success_rate=current_state.get('success_rate', 0.5),
            active_goals=current_state.get('active_goals', []),
            completed_goals=current_state.get('completed_goals', []),
            learned_topics=current_state.get('learned_topics', [])
        )

        # Generate available actions
        actions = self._generate_actions(state, available_files, available_topics)

        # Run MCTS planning
        best_sequence = self.mcts.plan(state, actions)

        # Convert to strategic plan
        plan = {
            'timestamp': datetime.now().isoformat(),
            'current_state': {
                'cycle': state.cycle_number,
                'coverage': state.test_coverage,
                'complexity': state.avg_complexity,
                'success_rate': state.success_rate
            },
            'recommended_actions': [
                {
                    'type': action.action_type.value,
                    'target': action.target,
                    'priority': action.priority,
                    'estimated_cycles': action.estimated_cost
                }
                for action in best_sequence
            ],
            'expected_outcome': self._predict_outcome(state, best_sequence)
        }

        return plan

    def _generate_actions(self, state: State, files: List[str], topics: List[str]) -> List[Action]:
        """Generate available actions"""
        actions = []

        # File improvement actions
        for file in files[:5]:  # Top 5 files
            actions.append(Action(
                ActionType.IMPROVE_CODE,
                file,
                priority=0.7,
                estimated_cost=1,
                estimated_benefit=0.6
            ))

        # Learning actions
        for topic in topics[:3]:  # Top 3 topics
            actions.append(Action(
                ActionType.LEARN_TOPIC,
                topic,
                priority=0.6,
                estimated_cost=2,
                estimated_benefit=0.5
            ))

        # Testing action
        if state.test_coverage < 0.8:
            actions.append(Action(
                ActionType.RUN_TESTS,
                "missing_coverage",
                priority=0.9,
                estimated_cost=1,
                estimated_benefit=0.8
            ))

        # Refactoring action
        if state.avg_complexity > 10:
            actions.append(Action(
                ActionType.REFACTOR,
                "complex_modules",
                priority=0.8,
                estimated_cost=2,
                estimated_benefit=0.7
            ))

        return actions

    def _predict_outcome(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Predict outcome of executing action sequence"""
        simulated_state = state.clone()

        for action in actions:
            simulated_state = self.mcts._apply_action(simulated_state, action)

        return {
            'final_coverage': round(simulated_state.test_coverage, 2),
            'final_complexity': round(simulated_state.avg_complexity, 1),
            'final_success_rate': round(simulated_state.success_rate, 2),
            'total_cycles': simulated_state.cycle_number - state.cycle_number,
            'goal_achieved': simulated_state.is_goal_achieved()
        }


if __name__ == "__main__":
    # Test MCTS planner
    logging.basicConfig(level=logging.INFO)

    # Create initial state
    initial_state = State(
        cycle_number=0,
        test_coverage=0.6,
        avg_complexity=12.0,
        success_rate=0.5,
        active_goals=["increase_coverage", "reduce_complexity"],
        completed_goals=[],
        learned_topics=[]
    )

    # Create planner
    mcts = MCTSPlanner(simulation_budget=500, exploration_constant=1.41)

    # Get available actions
    actions = [
        Action(ActionType.RUN_TESTS, "test_suite", priority=0.9, estimated_cost=1, estimated_benefit=0.8),
        Action(ActionType.REFACTOR, "core_module", priority=0.8, estimated_cost=2, estimated_benefit=0.7),
        Action(ActionType.LEARN_TOPIC, "async_patterns", priority=0.6, estimated_cost=2, estimated_benefit=0.5),
        Action(ActionType.IMPROVE_CODE, "utils.py", priority=0.7, estimated_cost=1, estimated_benefit=0.6),
    ]

    # Plan
    best_sequence = mcts.plan(initial_state, actions)

    print("\nBest action sequence:")
    for i, action in enumerate(best_sequence, 1):
        print(f"{i}. {action}")
