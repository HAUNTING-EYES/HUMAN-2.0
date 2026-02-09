from typing import Dict, List, Any, Optional
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum

class KnowledgeDomain(Enum):
    PHYSICAL = "physical"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    ABSTRACT = "abstract"

@dataclass
class KnowledgeNode:
    id: str
    domain: KnowledgeDomain
    content: Dict[str, Any]
    confidence: float
    connections: List[str]  # IDs of connected nodes
    last_updated: float

@dataclass
class Question:
    id: str
    domain: KnowledgeDomain
    content: str
    importance: float
    urgency: float
    complexity: float
    related_nodes: List[str]
    status: str = "unanswered"

class CuriosityEngine:
    def __init__(self):
        # Knowledge representation
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.uncertainty_map: Dict[str, float] = {}
        self.interest_areas: Dict[KnowledgeDomain, float] = {
            domain: 0.5 for domain in KnowledgeDomain
        }
        
        # Question management
        self.questions: Dict[str, Question] = {}
        self.question_history: List[str] = []  # Question IDs
        
        # Curiosity parameters
        self.novelty_threshold = 0.3
        self.complexity_preference = 0.5  # 0 = simple, 1 = complex
        self.exploration_rate = 0.7  # 0 = exploit, 1 = explore
        
    def update_knowledge(self, new_knowledge: Dict[str, Any]) -> None:
        """Update knowledge graph with new information"""
        node_id = self._generate_node_id()
        domain = self._classify_domain(new_knowledge)
        
        # Create new knowledge node
        node = KnowledgeNode(
            id=node_id,
            domain=domain,
            content=new_knowledge,
            confidence=self._calculate_confidence(new_knowledge),
            connections=[],
            last_updated=time.time()
        )
        
        # Add to knowledge graph
        self.knowledge_graph[node_id] = node
        
        # Update connections
        self._update_connections(node)
        
        # Update uncertainty map
        self._update_uncertainty(node)
        
        # Generate new questions
        self._generate_questions(node)
        
    def generate_curiosity(self) -> List[Question]:
        """Generate curiosity-driven questions"""
        # Update interest areas based on current state
        self._update_interest_areas()
        
        # Generate questions for each domain
        questions = []
        for domain in KnowledgeDomain:
            if self.interest_areas[domain] > self.novelty_threshold:
                domain_questions = self._generate_domain_questions(domain)
                questions.extend(domain_questions)
                
        # Prioritize questions
        prioritized_questions = self._prioritize_questions(questions)
        
        return prioritized_questions
        
    def _generate_node_id(self) -> str:
        """Generate unique ID for new knowledge node"""
        return f"node_{time.time()}_{np.random.randint(1000)}"
        
    def _classify_domain(self, knowledge: Dict[str, Any]) -> KnowledgeDomain:
        """Classify knowledge into a domain"""
        # TODO: Implement domain classification
        return KnowledgeDomain.COGNITIVE
        
    def _calculate_confidence(self, knowledge: Dict[str, Any]) -> float:
        """Calculate confidence in knowledge"""
        # TODO: Implement confidence calculation
        return 0.5
        
    def _update_connections(self, node: KnowledgeNode) -> None:
        """Update connections between knowledge nodes"""
        for existing_id, existing_node in self.knowledge_graph.items():
            if existing_id != node.id:
                similarity = self._calculate_similarity(node, existing_node)
                if similarity > 0.5:  # Threshold for connection
                    if existing_id not in node.connections:
                        node.connections.append(existing_id)
                    if node.id not in existing_node.connections:
                        existing_node.connections.append(node.id)
                    
    def _calculate_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Calculate similarity between two knowledge nodes"""
        # Basic similarity based on domain
        if node1.domain != node2.domain:
            return 0.0
            
        # Check for explicit relationships
        if "related_to" in node1.content and node1.content["related_to"] == node2.content.get("concept"):
            return 0.8
        if "related_to" in node2.content and node2.content["related_to"] == node1.content.get("concept"):
            return 0.8
            
        # Check for shared concepts
        related_concepts1 = set(node1.content.get("related_concepts", []))
        related_concepts2 = set(node2.content.get("related_concepts", []))
        if related_concepts1 and related_concepts2:
            overlap = len(related_concepts1.intersection(related_concepts2))
            total = len(related_concepts1.union(related_concepts2))
            if total > 0:
                return overlap / total
                
        return 0.0
        
    def _update_uncertainty(self, node: KnowledgeNode) -> None:
        """Update uncertainty map based on new knowledge"""
        # Calculate uncertainty for node
        uncertainty = 1 - node.confidence
        self.uncertainty_map[node.id] = uncertainty
        
        # Update connected nodes
        for connected_id in node.connections:
            if connected_id in self.uncertainty_map:
                self.uncertainty_map[connected_id] *= 0.9  # Reduce uncertainty
                
    def _generate_questions(self, node: KnowledgeNode) -> None:
        """Generate questions based on new knowledge"""
        # Generate questions about uncertainties
        if node.confidence < 0.8:
            question = Question(
                id=f"q_{time.time()}",
                domain=node.domain,
                content=f"Why is {node.content.get('concept', 'this')} uncertain?",
                importance=1 - node.confidence,
                urgency=0.5,
                complexity=0.5,
                related_nodes=[node.id]
            )
            self.questions[question.id] = question
            
        # Generate questions about connections
        if len(node.connections) > 0:
            question = Question(
                id=f"q_{time.time()}_connections",
                domain=node.domain,
                content=f"How do these concepts relate: {node.content.get('concept', 'this')}?",
                importance=0.7,
                urgency=0.3,
                complexity=0.8,
                related_nodes=[node.id] + node.connections[:3]
            )
            self.questions[question.id] = question
            
    def _update_interest_areas(self) -> None:
        """Update interest levels in different domains"""
        for domain in KnowledgeDomain:
            # Calculate average uncertainty in domain
            domain_nodes = [node for node in self.knowledge_graph.values() 
                          if node.domain == domain]
            if domain_nodes:
                avg_uncertainty = np.mean([self.uncertainty_map.get(node.id, 0.5) 
                                         for node in domain_nodes])
                # Update interest based on uncertainty and exploration rate
                self.interest_areas[domain] = (
                    avg_uncertainty * self.exploration_rate +
                    (1 - self.exploration_rate) * self.interest_areas[domain]
                )
                
    def _generate_domain_questions(self, domain: KnowledgeDomain) -> List[Question]:
        """Generate questions for a specific domain"""
        questions = []
        
        # Get domain-specific nodes
        domain_nodes = [node for node in self.knowledge_graph.values() 
                       if node.domain == domain]
        
        # Generate questions based on patterns and gaps
        if domain_nodes:
            # Question about patterns
            questions.append(Question(
                id=f"q_{time.time()}_pattern_{domain.value}",
                domain=domain,
                content=f"What patterns exist in {domain.value} knowledge?",
                importance=self.interest_areas[domain],
                urgency=0.4,
                complexity=0.7,
                related_nodes=[node.id for node in domain_nodes[:5]]
            ))
            
            # Question about gaps
            questions.append(Question(
                id=f"q_{time.time()}_gaps_{domain.value}",
                domain=domain,
                content=f"What are the main gaps in {domain.value} understanding?",
                importance=self.interest_areas[domain],
                urgency=0.6,
                complexity=0.6,
                related_nodes=[node.id for node in domain_nodes[:5]]
            ))
            
            # Questions about relationships between concepts
            for node in domain_nodes:
                if node.connections:
                    questions.append(Question(
                        id=f"q_{time.time()}_relations_{node.id}",
                        domain=domain,
                        content=f"How does {node.content.get('concept', 'this concept')} relate to its connected concepts?",
                        importance=self.interest_areas[domain] * node.confidence,
                        urgency=0.5,
                        complexity=0.8,
                        related_nodes=[node.id] + node.connections[:3]
                    ))
                
                # Questions about low confidence areas
                if node.confidence < 0.7:
                    questions.append(Question(
                        id=f"q_{time.time()}_confidence_{node.id}",
                        domain=domain,
                        content=f"Why is our understanding of {node.content.get('concept', 'this concept')} uncertain?",
                        importance=(1 - node.confidence) * self.interest_areas[domain],
                        urgency=0.7,
                        complexity=0.5,
                        related_nodes=[node.id]
                    ))
            
        return questions
        
    def _prioritize_questions(self, questions: List[Question]) -> List[Question]:
        """Prioritize questions based on importance, urgency, and complexity"""
        if not questions:
            return []
            
        # Calculate priority scores
        priorities = []
        for question in questions:
            priority = (
                question.importance * 0.4 +
                question.urgency * 0.3 +
                (question.complexity * self.complexity_preference) * 0.3
            )
            priorities.append((priority, question))
            
        # Sort by priority value (first element of tuple)
        priorities.sort(key=lambda x: x[0], reverse=True)
        
        return [q for _, q in priorities]
    def initialize(self):
        """Initialize curiosity engine for compatibility with main system"""
        # Reset state to initial conditions
        self.knowledge_graph = {}
        self.uncertainty_map = {}
        self.questions = {}
        self.question_history = []
        
        # Reset interest areas to baseline
        self.interest_areas = {
            domain: 0.5 for domain in KnowledgeDomain
        }
        
        return True
    
    def get_curiosity_state(self):
        """Get current curiosity state"""
        return {
            "knowledge_nodes": len(self.knowledge_graph),
            "active_questions": len(self.questions),
            "interest_areas": self.interest_areas,
            "curiosity_level": 0.5,
            "exploration_rate": self.exploration_rate,
            "complexity_preference": self.complexity_preference
        }

    def reinforce_question(self, question_id: str, reward: float) -> None:
        """
        Reinforce question based on learning outcome (reinforcement learning).

        This implements a simple reinforcement learning mechanism where:
        - Positive reward (0.5 to 1.0): Question led to valuable knowledge
        - Neutral reward (0.0): Question led to mediocre knowledge
        - Negative reward (-0.5 to 0.0): Question led to poor/irrelevant knowledge

        The reward updates:
        1. Question importance (for future similar questions)
        2. Domain interest (affects future question generation in this domain)
        3. Complexity preference (if complex questions work better)

        Args:
            question_id: ID of the question that was explored
            reward: Reward value (-1.0 to 1.0)
        """
        # Find the question
        question = self.questions.get(question_id)

        if not question:
            # Question not in current dict, might be in history
            # Just update domain interest based on question_id pattern
            for domain in KnowledgeDomain:
                if domain.value in question_id:
                    # Update domain interest
                    learning_rate = 0.1
                    self.interest_areas[domain] = np.clip(
                        self.interest_areas[domain] + (reward * learning_rate),
                        0.0,
                        1.0
                    )
                    break
            return

        # Update question importance (for similar future questions)
        learning_rate = 0.15
        question.importance = np.clip(
            question.importance + (reward * learning_rate),
            0.0,
            1.0
        )

        # Update domain interest
        # Positive reward -> increase interest in this domain
        # Negative reward -> decrease interest in this domain
        domain_learning_rate = 0.1
        self.interest_areas[question.domain] = np.clip(
            self.interest_areas[question.domain] + (reward * domain_learning_rate),
            0.2,  # Minimum interest (don't completely abandon domains)
            1.0
        )

        # Update complexity preference based on question complexity and reward
        # If complex questions get good rewards, increase complexity preference
        # If simple questions get good rewards, decrease complexity preference
        if reward > 0.3:  # Only learn from positive experiences
            complexity_influence = (question.complexity - 0.5) * reward * 0.05
            self.complexity_preference = np.clip(
                self.complexity_preference + complexity_influence,
                0.0,
                1.0
            )

        # Update exploration rate
        # Good rewards from exploring -> maintain high exploration
        # Bad rewards -> maybe exploit more (lower exploration)
        if reward > 0.5:
            self.exploration_rate = min(self.exploration_rate + 0.02, 0.9)
        elif reward < 0.0:
            self.exploration_rate = max(self.exploration_rate - 0.02, 0.3)

        # Mark question as answered
        question.status = "answered"

        # Add to history
        if question_id not in self.question_history:
            self.question_history.append(question_id)
