import torch
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import uuid
import pickle

from .quantum_inspired_nn import QuantumInspiredNN
from .web_learning import WebLearningSystem
from .code_analyzer import CodeAnalyzer
from .continuous_learning import ContinuousLearningSystem
from .self_improvement import SelfImprovement
from .emotional_memory import EmotionalMemory
from .logical_reasoning import LogicalReasoning

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class Brainstem:
    """Core component for managing system state and task processing."""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.web_learning = WebLearningSystem(base_dir=self.base_dir)
        self.emotional_memory = EmotionalMemory()
        self.code_analyzer = CodeAnalyzer(base_dir=self.base_dir)
        
        # Initialize quantum-inspired neural network
        self.qinn = QuantumInspiredNN(
            input_size=768,    # Size of input embeddings
            hidden_size=256,   # Size of hidden layer
            output_size=64,    # Size of output layer
            num_qubits=4       # Number of qubits for quantum simulation
        )
        
        # Initialize task management
        self.task_queue = []
        self.current_task = None
        self.task_history = []
        
        # Initialize current state
        self.current_state = {
            "task_queue_size": 0,
            "current_task": None,
            "task_history_size": 0,
            "emotional_state": self.emotional_memory._get_current_state(),
            "memory_count": len(self.emotional_memory.memory_buffer)
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming task.
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            Dictionary containing task results
        """
        task_id = str(uuid.uuid4())
        task['id'] = task_id
        task['status'] = 'pending'
        task['timestamp'] = datetime.now().isoformat()
        
        self.task_queue.append(task)
        self.current_task = task
        
        try:
            if task['type'] == 'learning':
                result = self._handle_learning_task(task)
            elif task['type'] == 'analysis':
                result = self._handle_analysis_task(task)
            elif task['type'] == 'improvement':
                result = self._handle_improvement_task(task)
            elif task['type'] == 'test':
                result = self._handle_test_task(task)
            else:
                return {
                    'success': False,
                    'task_id': task_id,
                    'error': f"Unknown task type: {task['type']}"
                }
                
            task['status'] = 'completed'
            task['result'] = result
            return {
                'success': True,
                'task_id': task_id,
                **result
            }
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            return {
                'success': False,
                'task_id': task_id,
                'error': str(e)
            }
            
        finally:
            self.task_history.append(task)
            self.current_task = None
            
    def _handle_test_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a test task."""
        return {
            'status': 'success',
            'message': 'Test task completed successfully'
        }
        
    def _handle_learning_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a learning task."""
        if 'source' not in task or 'content' not in task:
            raise ValueError("Learning task requires 'source' and 'content' parameters")
            
        result = self.web_learning.learn_from_url(task['source'])
        self.emotional_memory.process_interaction({
            'type': 'learning',
            'success': bool(result),
            'content': task['content']
        })
        return {
            'status': 'success',
            'message': 'Learning task completed successfully'
        }
        
    def _handle_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an analysis task."""
        if 'code' not in task:
            raise ValueError("Analysis task requires 'code' parameter")
            
        result = self.code_analyzer.analyze_code(task['code'])
        
        # Process result through quantum network
        if isinstance(result, dict) and 'embeddings' in result:
            embeddings = torch.tensor(result['embeddings'])
            quantum_result = self.qinn(embeddings)
            result['quantum_analysis'] = quantum_result.detach().numpy().tolist()
            
        return {
            'status': 'success',
            'analysis': result
        }
        
    def _handle_improvement_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an improvement task."""
        if 'code' not in task or 'metrics' not in task:
            raise ValueError("Improvement task requires 'code' and 'metrics' parameters")
            
        result = self.code_analyzer.improve_code(task['code'], task['metrics'])
        self.emotional_memory.process_interaction({
            'type': 'improvement',
            'success': bool(result),
            'content': 'Code improvement completed'
        })
        return {
            'status': 'success',
            'improvements': result
        }
        
    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save the current system state."""
        state = {
            'task_history': self.task_history,
            'emotional_state': self.emotional_memory.get_current_state()
        }
        
        filepath = Path(filepath)
        try:
            with filepath.open('wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"State saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            raise
            
    def load_state(self, filepath: Union[str, Path]) -> None:
        """Load a saved system state."""
        filepath = Path(filepath)
        try:
            
            with filepath.open('rb') as f:
                state = pickle.load(f)
                
            self.task_history = state['task_history']
            self.emotional_memory.update_state(state['emotional_state'])
            self.logger.info(f"State loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            raise
        
    def get_state(self) -> Dict[str, Any]:
        """Get current system state.
        
        Returns:
            Dictionary containing system state
        """
        # Update current state
        self.current_state.update({
            "task_queue_size": len(self.task_queue),
            "current_task": self.current_task,
            "task_history_size": len(self.task_history),
            "emotional_state": self.emotional_memory._get_current_state(),
            "memory_count": len(self.emotional_memory.memory_buffer)
        })
        
        return self.current_state
        
    def close(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'web_learning'):
                self.web_learning.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    def save_state(self, file_path: str):
        """Save current state to file."""
        try:
            state = {
                "current_state": self.current_state,
                "emotional_state": self.emotional_memory._get_current_state(),
                "memory_count": len(self.emotional_memory.memory_buffer)
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f)
                
            self.logger.info(f"State saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise
            
    def load_state(self, file_path: str):
        """Load state from file."""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            self.current_state = state.get("current_state", {})
            self.emotional_memory._set_current_state(state.get("emotional_state", {}))
            
            self.logger.info(f"State loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            raise 