import torch
import numpy as np
from typing import Dict, Any, List
import logging

class QuantumMeasurement:
    """Class for performing quantum measurements on quantum states."""
    
    def __init__(self):
        """Initialize the quantum measurement system."""
        self.logger = logging.getLogger(__name__)
        
    def measure_state(self, state: torch.Tensor, basis: str = 'computational') -> Dict[str, Any]:
        """Measure a quantum state in the specified basis.
        
        Args:
            state: The quantum state to measure
            basis: The measurement basis ('computational' or 'hadamard')
            
        Returns:
            Dictionary containing measurement results
        """
        if basis not in ['computational', 'hadamard']:
            raise ValueError("Basis must be either 'computational' or 'hadamard'")
            
        # Convert state to numpy for easier manipulation
        state_np = state.detach().numpy()
        
        # Calculate probabilities
        probs = np.abs(state_np) ** 2
        
        # Sample from the probability distribution
        outcome = np.random.choice(len(probs), p=probs)
        
        # Prepare measurement result
        result = {
            'outcome': outcome,
            'probability': float(probs[outcome]),
            'basis': basis,
            'state': state_np.tolist()
        }
        
        return result
        
    def measure_multiple(self, states: List[torch.Tensor], basis: str = 'computational') -> List[Dict[str, Any]]:
        """Measure multiple quantum states.
        
        Args:
            states: List of quantum states to measure
            basis: The measurement basis
            
        Returns:
            List of measurement results
        """
        return [self.measure_state(state, basis) for state in states]
        
    def expectation_value(self, state: torch.Tensor, observable: torch.Tensor) -> float:
        """Calculate the expectation value of an observable.
        
        Args:
            state: The quantum state
            observable: The observable operator
            
        Returns:
            The expectation value
        """
        # Convert to numpy for easier manipulation
        state_np = state.detach().numpy()
        obs_np = observable.detach().numpy()
        
        # Calculate expectation value
        exp_val = np.real(np.vdot(state_np, obs_np @ state_np))
        
        return float(exp_val)
        
    def variance(self, state: torch.Tensor, observable: torch.Tensor) -> float:
        """Calculate the variance of an observable.
        
        Args:
            state: The quantum state
            observable: The observable operator
            
        Returns:
            The variance
        """
        exp_val = self.expectation_value(state, observable)
        exp_val_sq = self.expectation_value(state, observable @ observable)
        
        return float(exp_val_sq - exp_val ** 2) 