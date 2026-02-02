import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import ast
from dataclasses import dataclass
from .code_actions import AdvancedCodeModifier

@dataclass
class CodeState:
    code: str
    metrics: Dict[str, float]
    action_history: List[str]

class CodeOptimizationEnv(gym.Env):
    """Custom Environment for code optimization."""
    
    def __init__(self, initial_code: str):
        super().__init__()
        
        self.code_modifier = AdvancedCodeModifier()
        self.initial_code = initial_code
        self.current_state = None
        
        # Define action space (discrete actions from code_modifier)
        self.action_space = spaces.Discrete(len(self.code_modifier.actions))
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'code_embedding': spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32),
            'metrics': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'action_history': spaces.MultiDiscrete([len(self.code_modifier.actions)] * 5)
        })
        
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.current_state = CodeState(
            code=self.initial_code,
            metrics=self._calculate_metrics(self.initial_code),
            action_history=[]
        )
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one optimization step."""
        if self.current_state is None:
            raise RuntimeError("Environment needs to be reset before stepping")
            
        # Get action details
        action_details = self.code_modifier.actions[action]
        
        # Apply code modification
        modified_code = self._apply_action(action)
        
        # Calculate new metrics
        new_metrics = self._calculate_metrics(modified_code)
        
        # Update state
        self.current_state.code = modified_code
        self.current_state.metrics = new_metrics
        self.current_state.action_history.append(action_details.name)
        if len(self.current_state.action_history) > 5:
            self.current_state.action_history.pop(0)
            
        # Calculate reward
        reward = self._calculate_reward(new_metrics)
        
        # Check if episode should end
        done = self._is_done()
        truncated = False
        info = {
            'metrics': new_metrics,
            'action': action_details.name,
            'code_length': len(modified_code)
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _apply_action(self, action: int) -> str:
        """Apply selected code modification action."""
        action_details = self.code_modifier.actions[action]
        
        if action_details.name == "extract_method":
            similar_blocks = self._find_similar_blocks(self.current_state.code)
            if similar_blocks:
                return self.code_modifier.extract_method(
                    self.current_state.code,
                    similar_blocks
                )
                
        elif action_details.name == "introduce_design_pattern":
            pattern_type = self._detect_appropriate_pattern(self.current_state.code)
            if pattern_type:
                return self.code_modifier.introduce_design_pattern(
                    self.current_state.code,
                    pattern_type
                )
                
        elif action_details.name == "optimize_data_structures":
            return self.code_modifier.optimize_data_structures(self.current_state.code)
            
        elif action_details.name == "add_concurrency":
            return self.code_modifier.add_concurrency(self.current_state.code)
            
        elif action_details.name == "improve_error_handling":
            return self.code_modifier.improve_error_handling(self.current_state.code)
            
        return self.current_state.code
    
    def _calculate_metrics(self, code: str) -> Dict[str, float]:
        """Calculate code quality metrics."""
        return {
            'complexity': self._calculate_complexity(code),
            'maintainability': self._calculate_maintainability(code),
            'performance': self._estimate_performance(code),
            'reliability': self._calculate_reliability(code),
            'security': self._analyze_security(code)
        }
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity."""
        tree = ast.parse(code)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return float(complexity)
    
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability index."""
        # Simplified maintainability calculation
        loc = len(code.splitlines())
        complexity = self._calculate_complexity(code)
        return 100.0 / (1 + np.log(complexity * loc))
    
    def _estimate_performance(self, code: str) -> float:
        """Estimate code performance score."""
        tree = ast.parse(code)
        score = 1.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                score *= 0.9  # Penalize loops
            elif isinstance(node, ast.ListComp):
                score *= 0.95  # Slight penalty for list comprehensions
                
        return max(0.1, score)
    
    def _calculate_reliability(self, code: str) -> float:
        """Calculate code reliability score."""
        tree = ast.parse(code)
        score = 1.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                score *= 1.1  # Reward error handling
            elif isinstance(node, ast.Assert):
                score *= 1.05  # Reward assertions
                
        return min(1.0, score)
    
    def _analyze_security(self, code: str) -> float:
        """Analyze code security."""
        tree = ast.parse(code)
        score = 1.0
        
        security_risks = [
            'eval', 'exec', 'os.system', 'subprocess.call',
            'pickle.loads', 'yaml.load'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in security_risks:
                        score *= 0.5
                        
        return score
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on metrics improvement."""
        if not hasattr(self, '_previous_metrics'):
            self._previous_metrics = metrics
            return 0.0
            
        reward = 0.0
        weights = {
            'complexity': -0.2,
            'maintainability': 0.3,
            'performance': 0.2,
            'reliability': 0.15,
            'security': 0.15
        }
        
        for metric, weight in weights.items():
            improvement = metrics[metric] - self._previous_metrics[metric]
            reward += improvement * weight
            
        self._previous_metrics = metrics
        return reward
    
    def _is_done(self) -> bool:
        """Check if optimization should stop."""
        if len(self.current_state.action_history) >= 10:
            return True
            
        # Check if metrics have converged
        if hasattr(self, '_previous_metrics'):
            diff = sum(abs(self.current_state.metrics[m] - self._previous_metrics[m])
                      for m in self.current_state.metrics)
            if diff < 0.01:
                return True
                
        return False
    
    def _get_observation(self) -> Dict:
        """Convert current state to observation."""
        return {
            'code_embedding': self._embed_code(self.current_state.code),
            'metrics': np.array(list(self.current_state.metrics.values())),
            'action_history': np.array(
                [next((i for i, a in enumerate(self.code_modifier.actions) if a.name == action), 0)
                 for action in self.current_state.action_history]
                + [0] * (5 - len(self.current_state.action_history))
            )
        }
    
    def _embed_code(self, code: str) -> np.ndarray:
        """Convert code to numerical embedding."""
        # Simple embedding based on AST node types
        embedding = np.zeros(512)
        tree = ast.parse(code)
        
        for i, node in enumerate(ast.walk(tree)):
            if i >= 512:
                break
            embedding[i] = hash(node.__class__.__name__) % 100
            
        return embedding
    
    def _find_similar_blocks(self, code: str) -> List[str]:
        """Find similar code blocks for extraction."""
        # First, normalize indentation
        lines = []
        min_indent = float('inf')
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent < float('inf'):
            for line in code.split('\n'):
                if line.strip():
                    # Remove common indentation but preserve relative indentation
                    indent = len(line) - len(line.lstrip())
                    new_indent = ' ' * (indent - min_indent)
                    lines.append(new_indent + line.lstrip())
                else:
                    lines.append(line)
        else:
            lines = code.split('\n')
        
        code = '\n'.join(lines)
        tree = ast.parse(code)
        blocks = []
        
        # Extract function bodies
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function body
                body_lines = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        # Skip assignments as they might have different variable names
                        continue
                    elif isinstance(stmt, ast.Expr):
                        # Handle expressions (like print)
                        body_lines.append(f'    {ast.unparse(stmt.value)}')
                    elif isinstance(stmt, ast.Return):
                        # Handle return statements
                        body_lines.append(f'    return {ast.unparse(stmt.value)}')
                    else:
                        # Handle other statements
                        body_lines.append(f'    {ast.unparse(stmt)}')
                if body_lines:  # Only add non-empty blocks
                    blocks.append('\n'.join(body_lines))
        
        # Find similar blocks
        similar_blocks = []
        for i, block1 in enumerate(blocks):
            for block2 in blocks[i+1:]:
                # Compare blocks line by line
                lines1 = block1.split('\n')
                lines2 = block2.split('\n')
                
                # Find common lines
                common_lines = []
                for line1 in lines1:
                    line1_stripped = line1.strip()
                    for line2 in lines2:
                        if line1_stripped == line2.strip():
                            common_lines.append(line1)
                            break
                
                # If we found enough similar lines, add the block
                if len(common_lines) >= 2:  # At least 2 similar lines
                    similar_block = '\n'.join(common_lines)
                    if similar_block not in similar_blocks:
                        similar_blocks.append(similar_block)
        
        return similar_blocks
    
    def _detect_appropriate_pattern(self, code: str) -> Optional[str]:
        """Detect which design pattern might be appropriate."""
        tree = ast.parse(code)
        
        # Simple pattern detection (can be enhanced)
        factory_indicators = ['create', 'build', 'make']
        observer_indicators = ['notify', 'update', 'subscribe']
        strategy_indicators = ['algorithm', 'policy', 'behavior']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name.lower()
                if any(ind in name for ind in factory_indicators):
                    return 'factory'
                elif any(ind in name for ind in observer_indicators):
                    return 'observer'
                elif any(ind in name for ind in strategy_indicators):
                    return 'strategy'
                    
        return None 