from typing import Dict, List, Any, Optional
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum

class PhysiologicalParameter(Enum):
    HEART_RATE = "heart_rate"
    CORTISOL = "cortisol"
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ADRENALINE = "adrenaline"
    OXYTOCIN = "oxytocin"
    BLOOD_PRESSURE = "blood_pressure"
    RESPIRATORY_RATE = "respiratory_rate"
    BODY_TEMPERATURE = "body_temperature"
    MUSCLE_TENSION = "muscle_tension"

@dataclass
class HomeostaticRange:
    min_value: float
    max_value: float
    optimal_value: float
    current_value: float
    adaptation_rate: float = 0.1
    
    def is_in_range(self) -> bool:
        return self.min_value <= self.current_value <= self.max_value
        
    def distance_from_optimal(self) -> float:
        return abs(self.current_value - self.optimal_value)
        
    def adapt(self, target: float):
        """Gradually adapt current value towards target"""
        delta = (target - self.current_value) * self.adaptation_rate
        self.current_value += delta
        self.current_value = max(self.min_value, min(self.max_value, self.current_value))

class PhysiologicalSystem:
    def __init__(self):
        # Initialize physiological parameters
        self.parameters: Dict[PhysiologicalParameter, HomeostaticRange] = {
            PhysiologicalParameter.HEART_RATE: HomeostaticRange(60, 100, 70, 70),
            PhysiologicalParameter.CORTISOL: HomeostaticRange(10, 30, 15, 15),
            PhysiologicalParameter.DOPAMINE: HomeostaticRange(30, 70, 50, 50),
            PhysiologicalParameter.SEROTONIN: HomeostaticRange(80, 120, 100, 100),
            PhysiologicalParameter.ADRENALINE: HomeostaticRange(0, 100, 20, 20),
            PhysiologicalParameter.OXYTOCIN: HomeostaticRange(0, 50, 25, 25),
            PhysiologicalParameter.BLOOD_PRESSURE: HomeostaticRange(90, 140, 120, 120),
            PhysiologicalParameter.RESPIRATORY_RATE: HomeostaticRange(12, 20, 16, 16),
            PhysiologicalParameter.BODY_TEMPERATURE: HomeostaticRange(36.5, 37.5, 37, 37),
            PhysiologicalParameter.MUSCLE_TENSION: HomeostaticRange(0, 100, 30, 30)
        }
        
        # Emotional impact matrices
        self.emotional_impacts: Dict[str, Dict[PhysiologicalParameter, float]] = {
            "joy": {
                PhysiologicalParameter.HEART_RATE: 10,
                PhysiologicalParameter.DOPAMINE: 20,
                PhysiologicalParameter.SEROTONIN: 15,
                PhysiologicalParameter.OXYTOCIN: 10
            },
            "fear": {
                PhysiologicalParameter.HEART_RATE: 30,
                PhysiologicalParameter.CORTISOL: 40,
                PhysiologicalParameter.ADRENALINE: 50,
                PhysiologicalParameter.MUSCLE_TENSION: 40
            },
            "anger": {
                PhysiologicalParameter.HEART_RATE: 20,
                PhysiologicalParameter.CORTISOL: 30,
                PhysiologicalParameter.ADRENALINE: 40,
                PhysiologicalParameter.BLOOD_PRESSURE: 20
            },
            "sadness": {
                PhysiologicalParameter.SEROTONIN: -20,
                PhysiologicalParameter.DOPAMINE: -10,
                PhysiologicalParameter.RESPIRATORY_RATE: -2
            },
            "love": {
                PhysiologicalParameter.OXYTOCIN: 30,
                PhysiologicalParameter.DOPAMINE: 15,
                PhysiologicalParameter.SEROTONIN: 10
            }
        }
        
        # State tracking
        self.last_update = time.time()
        self.state_history = []
        
    def update(self, emotional_state: Dict[str, float], delta_time: Optional[float] = None):
        """Update physiological state based on emotional input"""
        current_time = time.time()
        if delta_time is None:
            delta_time = current_time - self.last_update
            
        # Calculate emotional impacts
        impacts = self._calculate_emotional_impacts(emotional_state)
        
        # Apply impacts and natural recovery
        for param, impact in impacts.items():
            # Apply emotional impact
            target = self.parameters[param].current_value + impact
            
            # Natural recovery towards optimal (homeostasis)
            recovery_rate = 0.1 * delta_time
            optimal = self.parameters[param].optimal_value
            target = target * (1 - recovery_rate) + optimal * recovery_rate
            
            # Update parameter
            self.parameters[param].adapt(target)
            
        # Record state
        self._record_state()
        self.last_update = current_time
        
    def get_state(self) -> Dict[str, float]:
        """Get current physiological state"""
        return {
            param.value: range.current_value
            for param, range in self.parameters.items()
        }
        
    def get_state_summary(self) -> Dict[str, str]:
        """Get summary of current physiological state"""
        state = self.get_state()
        summary = {}
        
        # Analyze arousal
        heart_rate = state[PhysiologicalParameter.HEART_RATE.value]
        if heart_rate > 90:
            summary["arousal"] = "high"
        elif heart_rate < 70:
            summary["arousal"] = "low"
        else:
            summary["arousal"] = "normal"
            
        # Analyze stress
        cortisol = state[PhysiologicalParameter.CORTISOL.value]
        adrenaline = state[PhysiologicalParameter.ADRENALINE.value]
        if cortisol > 25 or adrenaline > 60:
            summary["stress"] = "high"
        elif cortisol < 15 and adrenaline < 30:
            summary["stress"] = "low"
        else:
            summary["stress"] = "moderate"
            
        # Analyze mood
        serotonin = state[PhysiologicalParameter.SEROTONIN.value]
        dopamine = state[PhysiologicalParameter.DOPAMINE.value]
        if serotonin > 110 and dopamine > 60:
            summary["mood"] = "very_positive"
        elif serotonin > 100 and dopamine > 50:
            summary["mood"] = "positive"
        elif serotonin < 90 or dopamine < 40:
            summary["mood"] = "negative"
        else:
            summary["mood"] = "neutral"
            
        return summary
        
    def _calculate_emotional_impacts(self, 
                                   emotional_state: Dict[str, float]
                                   ) -> Dict[PhysiologicalParameter, float]:
        """Calculate total physiological impacts from emotional state"""
        total_impacts = {param: 0.0 for param in PhysiologicalParameter}
        
        for emotion, intensity in emotional_state.items():
            if emotion in self.emotional_impacts:
                for param, impact in self.emotional_impacts[emotion].items():
                    total_impacts[param] += impact * intensity
                    
        return total_impacts
        
    def _record_state(self):
        """Record current state in history"""
        self.state_history.append({
            "timestamp": time.time(),
            "state": self.get_state(),
            "summary": self.get_state_summary()
        })
        
        # Keep only last 1000 states
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
            
    def get_stability_metrics(self) -> Dict[str, float]:
        """Calculate physiological stability metrics"""
        metrics = {}
        
        for param in PhysiologicalParameter:
            range = self.parameters[param]
            
            # Calculate stability as inverse of distance from optimal
            distance = range.distance_from_optimal()
            max_distance = max(
                abs(range.max_value - range.optimal_value),
                abs(range.min_value - range.optimal_value)
            )
            stability = 1 - (distance / max_distance)
            
            metrics[param.value] = stability
            
        # Overall stability
        metrics["overall"] = np.mean(list(metrics.values()))
        
        return metrics

    def initialize(self):
        """Initialize physiological system for compatibility with main system"""
        # Reset all parameters to optimal values
        for param in PhysiologicalParameter:
            range = self.parameters[param]
            range.current_value = range.optimal_value
        
        # Clear state history
        self.state_history = []
        self.last_update = time.time()
        
        return True
    
    def get_physiological_state(self) -> Dict[str, Any]:
        """Get current physiological state summary"""
        return {
            "parameters": self.get_state(),
            "summary": self.get_state_summary(),
            "stability": self.get_stability_metrics(),
            "overall_state": self.get_state_summary().get("mood", "neutral")
        }
    def get_physiological_state(self):
        """Get current physiological state summary"""
        return {
            "parameters": self.get_state(),
            "summary": self.get_state_summary(),
            "stability": self.get_stability_metrics(),
            "overall_state": self.get_state_summary().get("mood", "neutral")
        }
