from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from enum import Enum

# Use absolute imports instead of relative imports
try:
    from src.consciousness.self_awareness import SelfAwarenessSystem
    from src.consciousness.reflection import ReflectionEngine
    from src.consciousness.curiosity import CuriosityEngine
    from src.consciousness.physiology import PhysiologicalSystem
except ImportError:
    # Fallback for when running from src directory
    from consciousness.self_awareness import SelfAwarenessSystem
    from consciousness.reflection import ReflectionEngine
    from consciousness.curiosity import CuriosityEngine
    from consciousness.physiology import PhysiologicalSystem

@dataclass
class GlobalWorkspaceState:
    """Current state of the global workspace"""
    active_thoughts: List[Dict[str, Any]]
    attention_focus: str
    current_goals: List[Dict[str, Any]]
    emotional_state: Dict[str, float]
    physiological_state: Dict[str, float]
    consciousness_state: Dict[str, Any]

class AttentionFocus(Enum):
    EXTERNAL = "external"  # Focus on external input/stimuli
    INTERNAL = "internal"  # Focus on internal processes/thoughts
    EMOTIONAL = "emotional"  # Focus on emotional state
    PHYSICAL = "physical"  # Focus on physiological state
    SOCIAL = "social"  # Focus on social interaction
    TASK = "task"  # Focus on current task/goal

class HUMAN:
    def __init__(self):
        # Core Systems
        self.consciousness = SelfAwarenessSystem()
        self.reflection = ReflectionEngine()
        self.curiosity = CuriosityEngine()
        self.physiology = PhysiologicalSystem()
        
        # Integration Components
        self.global_workspace = GlobalWorkspaceState(
            active_thoughts=[],
            attention_focus=AttentionFocus.EXTERNAL.value,
            current_goals=[],
            emotional_state={},
            physiological_state={},
            consciousness_state={}
        )
        
        # System State
        self.is_initialized = False
        self.start_time = None
        self.last_update = None
        self.update_count = 0
        
        # Performance Metrics
        self.metrics = {
            "cognitive_load": 0.0,
            "emotional_stability": 0.0,
            "consciousness_clarity": 0.0,
            "physical_stability": 0.0
        }
        
    def initialize(self):
        """Initialize all systems and establish baseline state"""
        try:
            # Initialize core systems
            self._init_consciousness()
            self._init_reflection()
            self._init_curiosity()
            self._init_physiology()
            
            # Set initial state
            self.start_time = time.time()
            self.last_update = self.start_time
            self.is_initialized = True
            
            return True
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            return False
            
    def update(self, delta_time: Optional[float] = None):
        """Update all systems and maintain integration"""
        if not self.is_initialized:
            raise RuntimeError("HUMAN must be initialized before updating")
            
        current_time = time.time()
        if delta_time is None:
            delta_time = current_time - self.last_update
            
        try:
            # Update core systems
            self._update_consciousness(delta_time)
            self._update_reflection(delta_time)
            self._update_curiosity(delta_time)
            self._update_physiology(delta_time)
            
            # Update global workspace
            self._update_global_workspace()
            
            # Update metrics
            self._update_metrics()
            
            self.last_update = current_time
            self.update_count += 1
            
            return True
        except Exception as e:
            print(f"Update failed: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]):
        """Process external input through all relevant systems"""
        # Classify input type and route to appropriate systems
        if "text" in input_data:
            self._process_text_input(input_data["text"])
        if "emotion" in input_data:
            self._process_emotional_input(input_data["emotion"])
        if "sensory" in input_data:
            self._process_sensory_input(input_data["sensory"])
            
        # Update global workspace with new input
        self._integrate_input(input_data)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of all systems"""
        return {
            "global_workspace": self.global_workspace.__dict__,
            "metrics": self.metrics,
            "consciousness": self.consciousness.get_current_state(),
            "physiology": self.physiology.get_state(),
            "update_count": self.update_count,
            "uptime": time.time() - self.start_time if self.start_time else 0
        }
        
    def _init_consciousness(self):
        """Initialize consciousness system"""
        # TODO: Implement consciousness initialization
        pass
        
    def _init_reflection(self):
        """Initialize reflection system"""
        # TODO: Implement reflection initialization
        pass
        
    def _init_curiosity(self):
        """Initialize curiosity system"""
        # TODO: Implement curiosity initialization
        pass
        
    def _init_physiology(self):
        """Initialize physiological system"""
        # TODO: Implement physiology initialization
        pass
        
    def _update_consciousness(self, delta_time: float):
        """Update consciousness system"""
        self.consciousness.update_state()
        
    def _update_reflection(self, delta_time: float):
        """Update reflection system"""
        # Process recent experiences
        current_state = self.get_state()
        self.reflection.process_experience({
            "type": "state_update",
            "content": current_state,
            "timestamp": time.time()
        })
        
    def _update_curiosity(self, delta_time: float):
        """Update curiosity system"""
        # Generate new questions based on current state
        questions = self.curiosity.generate_curiosity()
        if questions:
            self.global_workspace.active_thoughts.extend([
                {"type": "question", "content": q.content}
                for q in questions[:3]  # Add top 3 questions
            ])
            
    def _update_physiology(self, delta_time: float):
        """Update physiological system"""
        self.physiology.update(
            self.global_workspace.emotional_state,
            delta_time
        )
        
    def _update_global_workspace(self):
        """Update global workspace state"""
        # Update consciousness state
        self.global_workspace.consciousness_state = (
            self.consciousness.get_current_state()
        )
        
        # Update physiological state
        self.global_workspace.physiological_state = (
            self.physiology.get_state()
        )
        
        # Maintain active thoughts (keep only recent/relevant)
        self.global_workspace.active_thoughts = (
            self.global_workspace.active_thoughts[-10:]
        )
        
    def _update_metrics(self):
        """Update performance metrics"""
        # Update cognitive load
        self.metrics["cognitive_load"] = len(
            self.global_workspace.active_thoughts
        ) / 10.0
        
        # Update emotional stability
        if self.global_workspace.emotional_state:
            self.metrics["emotional_stability"] = sum(
                self.global_workspace.emotional_state.values()
            ) / len(self.global_workspace.emotional_state)
            
        # Update consciousness clarity
        self.metrics["consciousness_clarity"] = (
            1.0 - self.metrics["cognitive_load"]
        )
        
        # Update physical stability
        phys_metrics = self.physiology.get_stability_metrics()
        self.metrics["physical_stability"] = phys_metrics["overall"]
        
    def _process_text_input(self, text: str):
        """Process text input"""
        # TODO: Implement text processing
        pass
        
    def _process_emotional_input(self, emotion: Dict[str, float]):
        """Process emotional input"""
        # Update emotional state
        self.global_workspace.emotional_state.update(emotion)
        
    def _process_sensory_input(self, sensory: Dict[str, Any]):
        """Process sensory input"""
        # TODO: Implement sensory processing
        pass
        
    def _integrate_input(self, input_data: Dict[str, Any]):
        """Integrate new input into global workspace"""
        # Add to active thoughts
        self.global_workspace.active_thoughts.append({
            "type": "input",
            "content": input_data,
            "timestamp": time.time()
        })
        
        # Update attention focus based on input
        if "priority" in input_data and input_data["priority"] > 0.8:
            self.global_workspace.attention_focus = (
                AttentionFocus.EXTERNAL.value
            )
