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
        """Find temporal patterns in experiences (sequences, recurring types)"""
        patterns = []
        if len(experiences) < 3:
            return patterns

        # Find sequence patterns (A -> B -> C)
        type_sequence = [exp.type for exp in experiences]
        sequence_counts = {}
        for i in range(len(type_sequence) - 2):
            seq = tuple(type_sequence[i:i+3])
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        # Create patterns for frequent sequences
        for seq, count in sequence_counts.items():
            if count >= 2:
                frequency = count / (len(experiences) - 2)
                pattern = Pattern(
                    pattern_type="temporal_sequence",
                    frequency=frequency,
                    confidence=min(0.9, frequency * 1.5),
                    examples=[exp for exp in experiences if exp.type in seq][:3],
                    description=f"Sequence: {' -> '.join(seq)} (occurs {count} times)"
                )
                patterns.append(pattern)

        # Find recurring type patterns (experiences clustered by type)
        type_counts = {}
        for exp in experiences:
            type_counts[exp.type] = type_counts.get(exp.type, 0) + 1

        for exp_type, count in type_counts.items():
            if count >= 3:
                frequency = count / len(experiences)
                pattern = Pattern(
                    pattern_type="recurring_type",
                    frequency=frequency,
                    confidence=min(0.9, frequency * 2),
                    examples=[exp for exp in experiences if exp.type == exp_type][:3],
                    description=f"Frequent '{exp_type}' experiences ({count}/{len(experiences)})"
                )
                patterns.append(pattern)

        return patterns
        
    async def _find_causal_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Find causal patterns in experiences (cause-effect relationships)"""
        patterns = []
        if len(experiences) < 2:
            return patterns

        # Look for cause-effect relationships (A followed by B consistently)
        transitions = {}
        for i in range(len(experiences) - 1):
            current = experiences[i]
            next_exp = experiences[i + 1]

            # Get outcome from current and type from next
            current_outcome = current.content.get("outcome", current.type) if isinstance(current.content, dict) else current.type
            next_trigger = next_exp.type

            key = (current_outcome, next_trigger)
            if key not in transitions:
                transitions[key] = {"count": 0, "examples": []}
            transitions[key]["count"] += 1
            transitions[key]["examples"].append((current, next_exp))

        # Create patterns for frequent causal relationships
        total_transitions = len(experiences) - 1
        for (cause, effect), data in transitions.items():
            if data["count"] >= 2:
                frequency = data["count"] / total_transitions
                pattern = Pattern(
                    pattern_type="causal",
                    frequency=frequency,
                    confidence=min(0.85, frequency * 2),
                    examples=[ex[0] for ex in data["examples"][:3]],
                    description=f"'{cause}' often leads to '{effect}' ({data['count']} times)"
                )
                patterns.append(pattern)

        # Look for error -> correction patterns
        for i in range(len(experiences) - 1):
            if self._is_error_experience(experiences[i]) and self._is_success_experience(experiences[i + 1]):
                pattern = Pattern(
                    pattern_type="error_correction",
                    frequency=1.0 / total_transitions,
                    confidence=0.7,
                    examples=[experiences[i], experiences[i + 1]],
                    description="Error followed by successful correction"
                )
                patterns.append(pattern)

        return patterns
        
    def _update_known_patterns(self, new_patterns: List[Pattern]) -> None:
        """Update known patterns with new observations using exponential moving average"""
        for pattern in new_patterns:
            # Create a key from pattern type and description
            key = f"{pattern.pattern_type}:{pattern.description[:50]}"

            if key in self.known_patterns:
                # Update existing pattern with exponential moving average
                existing = self.known_patterns[key]
                alpha = 0.3
                existing.frequency = (1 - alpha) * existing.frequency + alpha * pattern.frequency
                existing.confidence = (1 - alpha) * existing.confidence + alpha * pattern.confidence
                # Add new examples (keep max 10)
                existing.examples.extend(pattern.examples)
                existing.examples = existing.examples[-10:]
            else:
                # Add new pattern
                self.known_patterns[key] = pattern

        # Decay old patterns not seen recently (reduce confidence)
        decay_rate = 0.95
        keys_to_update = list(self.known_patterns.keys())
        new_pattern_keys = {f"{p.pattern_type}:{p.description[:50]}" for p in new_patterns}

        for key in keys_to_update:
            if key not in new_pattern_keys:
                self.known_patterns[key].confidence *= decay_rate

        # Remove patterns with very low confidence
        self.known_patterns = {
            k: v for k, v in self.known_patterns.items()
            if v.confidence > 0.1
        }
        
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