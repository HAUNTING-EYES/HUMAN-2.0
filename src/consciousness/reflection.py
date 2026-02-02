from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from enum import Enum
from collections import deque


class ReflectionTrigger(Enum):
    UNEXPECTED_OUTCOME = "unexpected_outcome"
    DECISION_POINT = "decision_point"
    ERROR_DETECTED = "error_detected"
    SUCCESS_ACHIEVED = "success_achieved"
    PATTERN_DETECTED = "pattern_detected"
    EMOTIONAL_EVENT = "emotional_event"


@dataclass
class Experience:
    timestamp: float
    type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class Pattern:
    pattern_type: str
    frequency: float
    confidence: float
    examples: List[Experience]
    description: str


class ReflectionEngine:
    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.pattern_threshold = 0.7
        self.max_reflection_depth = 3
        self._reset_state()
        
    def _reset_state(self) -> None:
        """Reset internal state"""
        self.short_term_memory = deque(maxlen=self.memory_capacity)
        self.long_term_memory = []
        self.known_patterns: Dict[str, Pattern] = {}
        self.current_reflection: Optional[Dict[str, Any]] = None
        self.reflection_depth = 0
        
    async def process_experience(self, experience: Dict[str, Any]) -> None:
        """Process a new experience and trigger reflection if needed"""
        exp = self._create_experience(experience)
        self.short_term_memory.append(exp)
        
        triggers = self._check_triggers(exp)
        if triggers:
            reflection_result = await self.reflect(triggers)
            self.current_reflection = reflection_result
    
    def _create_experience(self, experience: Dict[str, Any]) -> Experience:
        """Create Experience object from dictionary"""
        return Experience(
            timestamp=time.time(),
            type=experience.get("type", "general"),
            content=experience.get("content", {}),
            metadata={"processed": False}
        )
            
    def _check_triggers(self, experience: Experience) -> List[ReflectionTrigger]:
        """Check if experience should trigger reflection"""
        triggers = []
        
        trigger_map = {
            "unexpected_outcome": ReflectionTrigger.UNEXPECTED_OUTCOME,
            "decision_point": ReflectionTrigger.DECISION_POINT,
        }
        
        if experience.type in trigger_map:
            triggers.append(trigger_map[experience.type])
        
        if self._is_error_experience(experience):
            triggers.append(ReflectionTrigger.ERROR_DETECTED)
            
        if self._is_success_experience(experience):
            triggers.append(ReflectionTrigger.SUCCESS_ACHIEVED)
            
        if self._is_emotional_experience(experience):
            triggers.append(ReflectionTrigger.EMOTIONAL_EVENT)
            triggers.append(ReflectionTrigger.PATTERN_DETECTED)
            
        return triggers
    
    def _is_error_experience(self, experience: Experience) -> bool:
        """Check if experience represents an error"""
        return (experience.type == "error" or 
                (experience.type == "unexpected_outcome" and 
                 experience.content.get("actual") == "failure"))
    
    def _is_success_experience(self, experience: Experience) -> bool:
        """Check if experience represents a success"""
        return (experience.type == "success" or 
                (experience.type == "unexpected_outcome" and 
                 experience.content.get("actual") == "success"))
    
    def _is_emotional_experience(self, experience: Experience) -> bool:
        """Check if experience is emotional"""
        emotional_keys = ["emotion", "joy", "fear", "anger", "sadness"]
        return (experience.type == "emotional_event" or
                any(key in experience.content for key in emotional_keys))
        
    async def reflect(self, triggers: List[ReflectionTrigger]) -> Dict[str, Any]:
        """Perform reflection based on triggers"""
        self.reflection_depth += 1
        
        if self.reflection_depth > self.max_reflection_depth:
            self.reflection_depth -= 1
            return {"status": "max_depth_reached"}
        
        reflection_result = await self._create_reflection_result(triggers)
        self._update_long_term_memory(reflection_result)
        self.reflection_depth -= 1
        
        return reflection_result
    
    async def _create_reflection_result(self, triggers: List[ReflectionTrigger]) -> Dict[str, Any]:
        """Create reflection result dictionary"""
        patterns = await self._analyze_patterns()
        insights = self._generate_insights(triggers)
        actions = self._recommend_actions(triggers)
        
        return {
            "timestamp": time.time(),
            "triggers": triggers,
            "patterns": patterns,
            "insights": insights,
            "actions": actions
        }
        
    async def _analyze_patterns(self) -> List[Pattern]:
        """Analyze patterns in recent experiences"""
        recent_experiences = list(self.short_term_memory)
        
        temporal_patterns = await self._find_temporal_patterns(recent_experiences)
        causal_patterns = await self._find_causal_patterns(recent_experiences)
        
        all_patterns = temporal_patterns + causal_patterns
        self._update_known_patterns(all_patterns)
        
        return all_patterns
        
    def _generate_insights(self, triggers: List[ReflectionTrigger]) -> List[str]:
        """Generate insights based on patterns and triggers"""
        insights = []
        
        insights.extend(self._get_pattern_insights())
        insights.extend(self._get_trigger_insights(triggers))
                
        return insights
    
    def _get_pattern_insights(self) -> List[str]:
        """Generate insights from known patterns"""
        insights = []
        for pattern in self.known_patterns.values():
            if pattern.confidence > self.pattern_threshold:
                insights.append(self._pattern_to_insight(pattern))
        return insights
    
    def _get_trigger_insights(self, triggers: List[ReflectionTrigger]) -> List[str]:
        """Generate insights from triggers"""
        return [self._trigger_to_insight(trigger) for trigger in triggers]
        
    def _recommend_actions(self, triggers: List[ReflectionTrigger]) -> List[Dict[str, Any]]:
        """Recommend actions based on reflection"""
        action_map = {
            ReflectionTrigger.ERROR_DETECTED: {
                "type": "error_correction",
                "priority": "high",
                "description": "Review and correct recent error"
            },
            ReflectionTrigger.PATTERN_DETECTED: {
                "type": "pattern_reinforcement",
                "priority": "medium",
                "description": "Reinforce beneficial pattern"
            }
        }
        
        return [action_map[trigger] for trigger in triggers if trigger in action_map]
        
    def _is_unexpected(self, experience: Experience) -> bool:
        """Check if experience was unexpected"""
        return False
        
    def _is_decision_point(self, experience: Experience) -> bool:
        """Check if experience is a decision point"""
        return False
        
    def _is_error(self, experience: Experience) -> bool:
        """Check if experience contains an error"""
        return False
        
    def _is_success(self, experience: Experience) -> bool:
        """Check if experience represents a success"""
        return False
        
    async def _find_temporal_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Find temporal patterns in experiences"""
        return []
        
    async def _find_causal_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Find causal patterns in experiences"""
        return []
        
    def _update_known_patterns(self, new_patterns: List[Pattern]) -> None:
        """Update known patterns with new observations"""
        pass
        
    def _pattern_to_insight(self, pattern: Pattern) -> str:
        """Convert pattern to insight"""
        return f"Identified pattern: {pattern.description}"
        
    def _trigger_to_insight(self, trigger: ReflectionTrigger) -> str:
        """Convert trigger to insight"""
        return f"Triggered by: {trigger.value}"
        
    def _update_long_term_memory(self, reflection_result: Dict[str, Any]) -> None:
        """Update long-term memory with reflection results"""
        self.long_term_memory.append(reflection_result)

    def initialize(self):
        """Initialize reflection engine for compatibility with main system"""
        self._reset_state()
        return True
    
    def get_recent_insights(self) -> List[str]:
        """Get recent insights from reflection"""
        insights = []
        
        for reflection in self.long_term_memory[-5:]:
            if 'insights' in reflection:
                insights.extend(reflection['insights'])
        
        return insights[-10:]