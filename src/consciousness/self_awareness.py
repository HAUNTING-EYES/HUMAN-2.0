from typing import Dict, List, Any
import time
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class AttentionFocus(Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"
    METACOGNITIVE = "metacognitive"

@dataclass
class ProcessState:
    process_id: str
    status: str
    resource_usage: Dict[str, float]
    start_time: float
    last_update: float

class SelfAwarenessSystem:
    def __init__(self):
        # Layer 1: Basic Self-Monitoring
        self.current_goals = []
        self.active_processes: Dict[str, ProcessState] = {}
        self.resource_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "attention": 0.0
        }
        self.attention_focus = AttentionFocus.EXTERNAL
        
        # Layer 2: Self-Reflection
        self.thought_history = []
        self.decision_history = []
        self.behavior_patterns = {}
        self.performance_metrics = {}
        
        # Layer 3: Identity Core
        self.personal_history = []
        self.belief_system = {}
        self.values = {}
        self.personality_traits = {}
        
    def update_state(self):
        """Update internal state tracking"""
        current_time = time.time()
        
        # Update process states
        for process_id, state in self.active_processes.items():
            state.last_update = current_time
            
        # Update resource usage
        self._monitor_resources()
        
        # Update attention focus
        self._update_attention()
        
    def _monitor_resources(self):
        """Monitor system resource usage using psutil if available"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self.resource_usage["cpu"] = process.cpu_percent() / 100.0
                self.resource_usage["memory"] = process.memory_percent() / 100.0
            except Exception as e:
                logger.warning(f"Failed to get system resources: {e}")

        # Calculate attention from active processes
        active_count = sum(
            1 for p in self.active_processes.values()
            if p.status == "running"
        )
        self.resource_usage["attention"] = min(1.0, active_count / 5.0)
        
    def _update_attention(self):
        """Update current attention focus based on active processes, goals, and recent experiences"""
        # Check recent experiences for external stimuli
        if self.personal_history:
            recent = self.personal_history[-1].get("experience", {})
            exp_type = recent.get("type", "")
            if "external" in exp_type.lower() or "stimulus" in exp_type.lower():
                self.attention_focus = AttentionFocus.EXTERNAL
                return

        if not self.active_processes:
            self.attention_focus = AttentionFocus.INTERNAL
            return

        # Check for metacognitive processes (self-reflection, planning)
        metacognitive_keywords = ["reflect", "plan", "analyze_self", "meta", "introspect"]
        for process in self.active_processes.values():
            if any(kw in process.process_id.lower() for kw in metacognitive_keywords):
                self.attention_focus = AttentionFocus.METACOGNITIVE
                return

        # Check for external interaction processes
        external_keywords = ["learn", "fetch", "api", "github", "web", "external"]
        for process in self.active_processes.values():
            if any(kw in process.process_id.lower() for kw in external_keywords):
                self.attention_focus = AttentionFocus.EXTERNAL
                return

        self.attention_focus = AttentionFocus.INTERNAL
        
    def add_experience(self, experience: Dict[str, Any]):
        """Add new experience to personal history"""
        self.personal_history.append({
            "timestamp": time.time(),
            "experience": experience
        })
        
        # Update belief system based on experience
        self._update_beliefs(experience)
        
        # Update personality traits
        self._update_personality(experience)
        
    def _update_beliefs(self, experience: Dict[str, Any]):
        """Update belief system based on new experience using exponential moving average"""
        exp_type = experience.get("type", "unknown")
        content = experience.get("content", {})

        # Extract topic and success from experience
        topic = content.get("topic", exp_type) if isinstance(content, dict) else exp_type
        success = content.get("success", False) if isinstance(content, dict) else False

        # Initialize belief for this topic if not exists
        if topic not in self.belief_system:
            self.belief_system[topic] = {
                "success_rate": 0.5,
                "attempts": 0,
                "last_outcome": None
            }

        belief = self.belief_system[topic]
        belief["attempts"] += 1
        belief["last_outcome"] = "success" if success else "failure"

        # Exponential moving average for success rate
        alpha = 0.1  # Learning rate
        belief["success_rate"] = (1 - alpha) * belief["success_rate"] + alpha * (1.0 if success else 0.0)
        
    def _update_personality(self, experience: Dict[str, Any]):
        """Update personality traits based on new experience"""
        # Initialize default traits if empty
        if not self.personality_traits:
            self.personality_traits = {
                "curiosity": 0.7,
                "persistence": 0.6,
                "caution": 0.5,
                "exploration_vs_exploitation": 0.5
            }

        exp_type = experience.get("type", "")
        content = experience.get("content", {})
        if not isinstance(content, dict):
            content = {}

        # Adjust curiosity based on learning outcomes
        if exp_type == "learning":
            novel_patterns = content.get("novel_patterns", 0)
            if novel_patterns > 0:
                self.personality_traits["curiosity"] = min(
                    1.0, self.personality_traits["curiosity"] + 0.02
                )

        # Adjust persistence based on improvement outcomes
        if exp_type == "improvement":
            retries = content.get("required_retries", 0)
            success = content.get("success", False)
            if retries > 2 and success:
                self.personality_traits["persistence"] = min(
                    1.0, self.personality_traits["persistence"] + 0.01
                )

        # Adjust caution based on errors
        if exp_type == "error":
            self.personality_traits["caution"] = min(
                1.0, self.personality_traits["caution"] + 0.03
            )
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of self-awareness"""
        return {
            "goals": self.current_goals,
            "processes": self.active_processes,
            "resources": self.resource_usage,
            "attention": self.attention_focus.value,
            "recent_thoughts": self.thought_history[-10:],
            "recent_decisions": self.decision_history[-10:],
            "current_beliefs": self.belief_system,
            "current_values": self.values,
            "personality": self.personality_traits
        }
        
    def reflect(self) -> Dict[str, Any]:
        """Perform self-reflection"""
        reflection = {
            "timestamp": time.time(),
            "state": self.get_current_state(),
            "patterns": self._analyze_patterns(),
            "insights": self._generate_insights()
        }
        return reflection
        
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in behavior and thought"""
        patterns = {
            "decision_patterns": {},
            "thought_frequency": {},
            "behavior_trends": []
        }

        # Analyze decision history
        if len(self.decision_history) >= 5:
            recent_decisions = self.decision_history[-20:]
            decision_types = {}
            for decision in recent_decisions:
                dtype = decision.get("type", "unknown") if isinstance(decision, dict) else "unknown"
                decision_types[dtype] = decision_types.get(dtype, 0) + 1
            patterns["decision_patterns"] = decision_types

        # Analyze thought history
        if len(self.thought_history) >= 5:
            recent_thoughts = self.thought_history[-50:]
            for thought in recent_thoughts:
                category = thought.get("category", "general") if isinstance(thought, dict) else "general"
                patterns["thought_frequency"][category] = patterns["thought_frequency"].get(category, 0) + 1

        # Identify trends from decision patterns
        if patterns["decision_patterns"]:
            total = sum(patterns["decision_patterns"].values())
            for dtype, count in patterns["decision_patterns"].items():
                ratio = count / total
                if ratio > 0.4:
                    patterns["behavior_trends"].append(f"Dominant: {dtype} ({ratio:.0%})")

        # Identify trends from beliefs
        low_success_topics = [
            topic for topic, belief in self.belief_system.items()
            if belief.get("success_rate", 0.5) < 0.3 and belief.get("attempts", 0) >= 3
        ]
        if low_success_topics:
            patterns["behavior_trends"].append(f"Struggling with: {', '.join(low_success_topics[:3])}")

        return patterns
        
    def _generate_insights(self) -> List[str]:
        """Generate insights from self-reflection based on patterns, resources, and beliefs"""
        insights = []
        patterns = self._analyze_patterns()

        # Insight from decision patterns
        if patterns.get("decision_patterns"):
            total_decisions = sum(patterns["decision_patterns"].values())
            for dtype, count in patterns["decision_patterns"].items():
                ratio = count / total_decisions
                if ratio > 0.5:
                    insights.append(f"High focus on {dtype} decisions ({ratio:.0%})")

        # Insight from resource usage
        if self.resource_usage.get("attention", 0) > 0.8:
            insights.append("Attention resources heavily utilized - consider prioritization")

        if self.resource_usage.get("cpu", 0) > 0.7:
            insights.append("High CPU usage detected - may need optimization")

        # Insight from beliefs (low confidence topics)
        low_confidence_topics = [
            topic for topic, belief in self.belief_system.items()
            if belief.get("success_rate", 0.5) < 0.4 and belief.get("attempts", 0) > 3
        ]
        if low_confidence_topics:
            insights.append(f"Low success rate in: {', '.join(low_confidence_topics[:3])}")

        # Insight from personality traits
        if self.personality_traits.get("exploration_vs_exploitation", 0.5) < 0.3:
            insights.append("Consider more exploration to discover new approaches")

        if self.personality_traits.get("caution", 0.5) > 0.8:
            insights.append("High caution level - may be overly conservative")

        if self.personality_traits.get("curiosity", 0.7) < 0.4:
            insights.append("Low curiosity - consider exploring new domains")

        # Insight from experience history
        if len(self.personal_history) > 100:
            recent_errors = sum(
                1 for exp in self.personal_history[-20:]
                if exp.get("experience", {}).get("type") == "error"
            )
            if recent_errors > 5:
                insights.append(f"High recent error rate ({recent_errors}/20) - review approach")

        return insights

    def initialize(self):
        """No-op initializer for compatibility with main system."""
        return True 