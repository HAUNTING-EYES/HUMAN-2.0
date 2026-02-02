from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field
from enum import Enum

try:
    from src.consciousness.self_awareness import SelfAwarenessSystem
    from src.consciousness.reflection import ReflectionEngine
    from src.consciousness.curiosity import CuriosityEngine
    from src.consciousness.physiology import PhysiologicalSystem
except ImportError:
    from consciousness.self_awareness import SelfAwarenessSystem
    from consciousness.reflection import ReflectionEngine
    from consciousness.curiosity import CuriosityEngine
    from consciousness.physiology import PhysiologicalSystem


class AttentionFocus(Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"
    EMOTIONAL = "emotional"
    PHYSICAL = "physical"
    SOCIAL = "social"
    TASK = "task"


@dataclass
class GlobalWorkspaceState:
    active_thoughts: List[Dict[str, Any]] = field(default_factory=list)
    attention_focus: str = AttentionFocus.EXTERNAL.value
    current_goals: List[Dict[str, Any]] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    physiological_state: Dict[str, float] = field(default_factory=dict)
    consciousness_state: Dict[str, Any] = field(default_factory=dict)


class HUMAN:
    MAX_ACTIVE_THOUGHTS = 10
    TOP_QUESTIONS_LIMIT = 3
    HIGH_PRIORITY_THRESHOLD = 0.8
    
    def __init__(self):
        self.consciousness = SelfAwarenessSystem()
        self.reflection = ReflectionEngine()
        self.curiosity = CuriosityEngine()
        self.physiology = PhysiologicalSystem()
        
        self.global_workspace = GlobalWorkspaceState()
        
        self.is_initialized = False
        self.start_time = None
        self.last_update = None
        self.update_count = 0
        
        self.metrics = {
            "cognitive_load": 0.0,
            "emotional_stability": 0.0,
            "consciousness_clarity": 0.0,
            "physical_stability": 0.0
        }
        
    def initialize(self):
        try:
            self._initialize_all_systems()
            self._set_initial_state()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            return False
    
    def _initialize_all_systems(self):
        self._init_consciousness()
        self._init_reflection()
        self._init_curiosity()
        self._init_physiology()
    
    def _set_initial_state(self):
        current_time = time.time()
        self.start_time = current_time
        self.last_update = current_time
            
    def update(self, delta_time: Optional[float] = None):
        if not self.is_initialized:
            raise RuntimeError("HUMAN must be initialized before updating")
            
        current_time = time.time()
        delta_time = delta_time if delta_time is not None else current_time - self.last_update
            
        try:
            self._update_all_systems(delta_time)
            self._update_global_workspace()
            self._update_metrics()
            
            self.last_update = current_time
            self.update_count += 1
            return True
        except Exception as e:
            print(f"Update failed: {str(e)}")
            return False
    
    def _update_all_systems(self, delta_time: float):
        self._update_consciousness(delta_time)
        self._update_reflection(delta_time)
        self._update_curiosity(delta_time)
        self._update_physiology(delta_time)
            
    def process_input(self, input_data: Dict[str, Any]):
        self._route_input_to_systems(input_data)
        self._integrate_input(input_data)
    
    def _route_input_to_systems(self, input_data: Dict[str, Any]):
        input_handlers = {
            "text": self._process_text_input,
            "emotion": self._process_emotional_input,
            "sensory": self._process_sensory_input
        }
        
        for key, handler in input_handlers.items():
            if key in input_data:
                handler(input_data[key])
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "global_workspace": self._get_workspace_dict(),
            "metrics": self.metrics,
            "consciousness": self.consciousness.get_current_state(),
            "physiology": self.physiology.get_state(),
            "update_count": self.update_count,
            "uptime": self._calculate_uptime()
        }
    
    def _get_workspace_dict(self) -> Dict[str, Any]:
        return {
            "active_thoughts": self.global_workspace.active_thoughts,
            "attention_focus": self.global_workspace.attention_focus,
            "current_goals": self.global_workspace.current_goals,
            "emotional_state": self.global_workspace.emotional_state,
            "physiological_state": self.global_workspace.physiological_state,
            "consciousness_state": self.global_workspace.consciousness_state
        }
    
    def _calculate_uptime(self) -> float:
        return time.time() - self.start_time if self.start_time else 0
        
    def _init_consciousness(self):
        pass
        
    def _init_reflection(self):
        pass
        
    def _init_curiosity(self):
        pass
        
    def _init_physiology(self):
        pass
        
    def _update_consciousness(self, delta_time: float):
        self.consciousness.update_state()
        
    def _update_reflection(self, delta_time: float):
        experience = self._create_experience_data()
        self.reflection.process_experience(experience)
    
    def _create_experience_data(self) -> Dict[str, Any]:
        return {
            "type": "state_update",
            "content": self.get_state(),
            "timestamp": time.time()
        }
        
    def _update_curiosity(self, delta_time: float):
        questions = self.curiosity.generate_curiosity()
        if questions:
            self._add_questions_to_workspace(questions)
    
    def _add_questions_to_workspace(self, questions: List[Any]):
        new_thoughts = [
            {"type": "question", "content": q.content}
            for q in questions[:self.TOP_QUESTIONS_LIMIT]
        ]
        self.global_workspace.active_thoughts.extend(new_thoughts)
            
    def _update_physiology(self, delta_time: float):
        self.physiology.update(
            self.global_workspace.emotional_state,
            delta_time
        )
        
    def _update_global_workspace(self):
        self.global_workspace.consciousness_state = self.consciousness.get_current_state()
        self.global_workspace.physiological_state = self.physiology.get_state()
        self._trim_active_thoughts()
    
    def _trim_active_thoughts(self):
        self.global_workspace.active_thoughts = (
            self.global_workspace.active_thoughts[-self.MAX_ACTIVE_THOUGHTS:]
        )
        
    def _update_metrics(self):
        self._update_cognitive_load()
        self._update_emotional_stability()
        self._update_consciousness_clarity()
        self._update_physical_stability()
    
    def _update_cognitive_load(self):
        thought_count = len(self.global_workspace.active_thoughts)
        self.metrics["cognitive_load"] = thought_count / float(self.MAX_ACTIVE_THOUGHTS)
    
    def _update_emotional_stability(self):
        if self.global_workspace.emotional_state:
            total = sum(self.global_workspace.emotional_state.values())
            count = len(self.global_workspace.emotional_state)
            self.metrics["emotional_stability"] = total / count
    
    def _update_consciousness_clarity(self):
        self.metrics["consciousness_clarity"] = 1.0 - self.metrics["cognitive_load"]
    
    def _update_physical_stability(self):
        phys_metrics = self.physiology.get_stability_metrics()
        self.metrics["physical_stability"] = phys_metrics["overall"]
        
    def _process_text_input(self, text: str):
        pass
        
    def _process_emotional_input(self, emotion: Dict[str, float]):
        self.global_workspace.emotional_state.update(emotion)
        
    def _process_sensory_input(self, sensory: Dict[str, Any]):
        pass
        
    def _integrate_input(self, input_data: Dict[str, Any]):
        self._add_input_to_thoughts(input_data)
        self._update_attention_focus(input_data)
    
    def _add_input_to_thoughts(self, input_data: Dict[str, Any]):
        self.global_workspace.active_thoughts.append({
            "type": "input",
            "content": input_data,
            "timestamp": time.time()
        })
    
    def _update_attention_focus(self, input_data: Dict[str, Any]):
        if self._is_high_priority_input(input_data):
            self.global_workspace.attention_focus = AttentionFocus.EXTERNAL.value
    
    def _is_high_priority_input(self, input_data: Dict[str, Any]) -> bool:
        return (
            "priority" in input_data and 
            input_data["priority"] > self.HIGH_PRIORITY_THRESHOLD
        )