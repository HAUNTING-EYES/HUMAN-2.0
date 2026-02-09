from typing import Dict, List, Any, Optional
import time
import numpy as np
from dataclasses import dataclass, field
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
    connections: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

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
    CONNECTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.7
    UNCERTAINTY_REDUCTION_FACTOR = 0.9
    MIN_DOMAIN_INTEREST = 0.2
    MAX_DOMAIN_INTEREST = 1.0
    
    def __init__(self):
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.uncertainty_map: Dict[str, float] = {}
        self.interest_areas: Dict[KnowledgeDomain, float] = {
            domain: 0.5 for domain in KnowledgeDomain
        }
        self.questions: Dict[str, Question] = {}
        self.question_history: List[str] = []
        self.novelty_threshold = 0.3
        self.complexity_preference = 0.5
        self.exploration_rate = 0.7
        
    def update_knowledge(self, new_knowledge: Dict[str, Any]) -> None:
        node = self._create_knowledge_node(new_knowledge)
        self.knowledge_graph[node.id] = node
        self._update_connections(node)
        self._update_uncertainty(node)
        self._generate_questions(node)
        
    def generate_curiosity(self) -> List[Question]:
        self._update_interest_areas()
        questions = self._collect_domain_questions()
        return self._prioritize_questions(questions)
    
    def initialize(self):
        self.knowledge_graph = {}
        self.uncertainty_map = {}
        self.questions = {}
        self.question_history = []
        self.interest_areas = {domain: 0.5 for domain in KnowledgeDomain}
        return True
    
    def get_curiosity_state(self):
        return {
            "knowledge_nodes": len(self.knowledge_graph),
            "active_questions": len(self.questions),
            "interest_areas": self.interest_areas,
            "curiosity_level": 0.5,
            "exploration_rate": self.exploration_rate,
            "complexity_preference": self.complexity_preference
        }

    def reinforce_question(self, question_id: str, reward: float) -> None:
        question = self.questions.get(question_id)
        
        if not question:
            self._update_domain_interest_by_id(question_id, reward)
            return

        self._update_question_importance(question, reward)
        self._update_domain_interest(question.domain, reward)
        self._update_complexity_preference(question.complexity, reward)
        self._update_exploration_rate(reward)
        self._mark_question_answered(question_id, question)

    def _create_knowledge_node(self, new_knowledge: Dict[str, Any]) -> KnowledgeNode:
        return KnowledgeNode(
            id=self._generate_node_id(),
            domain=self._classify_domain(new_knowledge),
            content=new_knowledge,
            confidence=self._calculate_confidence(new_knowledge),
            connections=[],
            last_updated=time.time()
        )
        
    def _generate_node_id(self) -> str:
        return f"node_{time.time()}_{np.random.randint(1000)}"
        
    def _classify_domain(self, knowledge: Dict[str, Any]) -> KnowledgeDomain:
        return KnowledgeDomain.COGNITIVE
        
    def _calculate_confidence(self, knowledge: Dict[str, Any]) -> float:
        return 0.5
        
    def _update_connections(self, node: KnowledgeNode) -> None:
        for existing_id, existing_node in self.knowledge_graph.items():
            if existing_id != node.id:
                self._create_connection_if_similar(node, existing_node, existing_id)
                    
    def _create_connection_if_similar(self, node: KnowledgeNode, existing_node: KnowledgeNode, existing_id: str) -> None:
        similarity = self._calculate_similarity(node, existing_node)
        if similarity > self.CONNECTION_THRESHOLD:
            if existing_id not in node.connections:
                node.connections.append(existing_id)
            if node.id not in existing_node.connections:
                existing_node.connections.append(node.id)
                    
    def _calculate_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        if node1.domain != node2.domain:
            return 0.0
        
        explicit_similarity = self._check_explicit_relationships(node1, node2)
        if explicit_similarity > 0:
            return explicit_similarity
        
        return self._calculate_concept_overlap(node1, node2)
    
    def _check_explicit_relationships(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        if "related_to" in node1.content and node1.content["related_to"] == node2.content.get("concept"):
            return 0.8
        if "related_to" in node2.content and node2.content["related_to"] == node1.content.get("concept"):
            return 0.8
        return 0.0
    
    def _calculate_concept_overlap(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        related_concepts1 = set(node1.content.get("related_concepts", []))
        related_concepts2 = set(node2.content.get("related_concepts", []))
        
        if not (related_concepts1 and related_concepts2):
            return 0.0
        
        overlap = len(related_concepts1.intersection(related_concepts2))
        total = len(related_concepts1.union(related_concepts2))
        return overlap / total if total > 0 else 0.0
        
    def _update_uncertainty(self, node: KnowledgeNode) -> None:
        self.uncertainty_map[node.id] = 1 - node.confidence
        
        for connected_id in node.connections:
            if connected_id in self.uncertainty_map:
                self.uncertainty_map[connected_id] *= self.UNCERTAINTY_REDUCTION_FACTOR
                
    def _generate_questions(self, node: KnowledgeNode) -> None:
        if node.confidence < self.CONFIDENCE_THRESHOLD:
            self._add_uncertainty_question(node)
            
        if node.connections:
            self._add_connection_question(node)
            
    def _add_uncertainty_question(self, node: KnowledgeNode) -> None:
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
    
    def _add_connection_question(self, node: KnowledgeNode) -> None:
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
        for domain in KnowledgeDomain:
            domain_nodes = self._get_domain_nodes(domain)
            if domain_nodes:
                avg_uncertainty = self._calculate_average_uncertainty(domain_nodes)
                self.interest_areas[domain] = (
                    avg_uncertainty * self.exploration_rate +
                    (1 - self.exploration_rate) * self.interest_areas[domain]
                )
    
    def _get_domain_nodes(self, domain: KnowledgeDomain) -> List[KnowledgeNode]:
        return [node for node in self.knowledge_graph.values() if node.domain == domain]
    
    def _calculate_average_uncertainty(self, nodes: List[KnowledgeNode]) -> float:
        return np.mean([self.uncertainty_map.get(node.id, 0.5) for node in nodes])
    
    def _collect_domain_questions(self) -> List[Question]:
        questions = []
        for domain in KnowledgeDomain:
            if self.interest_areas[domain] > self.novelty_threshold:
                questions.extend(self._generate_domain_questions(domain))
        return questions
                
    def _generate_domain_questions(self, domain: KnowledgeDomain) -> List[Question]:
        domain_nodes = self._get_domain_nodes(domain)
        if not domain_nodes:
            return []
        
        questions = []
        questions.extend(self._create_pattern_and_gap_questions(domain, domain_nodes))
        questions.extend(self._create_node_specific_questions(domain, domain_nodes))
        return questions
    
    def _create_pattern_and_gap_questions(self, domain: KnowledgeDomain, nodes: List[KnowledgeNode]) -> List[Question]:
        node_ids = [node.id for node in nodes[:5]]
        
        pattern_question = Question(
            id=f"q_{time.time()}_pattern_{domain.value}",
            domain=domain,
            content=f"What patterns exist in {domain.value} knowledge?",
            importance=self.interest_areas[domain],
            urgency=0.4,
            complexity=0.7,
            related_nodes=node_ids
        )
        
        gap_question = Question(
            id=f"q_{time.time()}_gaps_{domain.value}",
            domain=domain,
            content=f"What are the main gaps in {domain.value} understanding?",
            importance=self.interest_areas[domain],
            urgency=0.6,
            complexity=0.6,
            related_nodes=node_ids
        )
        
        return [pattern_question, gap_question]
    
    def _create_node_specific_questions(self, domain: KnowledgeDomain, nodes: List[KnowledgeNode]) -> List[Question]:
        questions = []
        for node in nodes:
            if node.connections:
                questions.append(self._create_relation_question(domain, node))
            if node.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                questions.append(self._create_confidence_question(domain, node))
        return questions
    
    def _create_relation_question(self, domain: KnowledgeDomain, node: KnowledgeNode) -> Question:
        return Question(
            id=f"q_{time.time()}_relations_{node.id}",
            domain=domain,
            content=f"How does {node.content.get('concept', 'this concept')} relate to its connected concepts?",
            importance=self.interest_areas[domain] * node.confidence,
            urgency=0.5,
            complexity=0.8,
            related_nodes=[node.id] + node.connections[:3]
        )
    
    def _create_confidence_question(self, domain: KnowledgeDomain, node: KnowledgeNode) -> Question:
        return Question(
            id=f"q_{time.time()}_confidence_{node.id}",
            domain=domain,
            content=f"Why is our understanding of {node.content.get('concept', 'this concept')} uncertain?",
            importance=(1 - node.confidence) * self.interest_areas[domain],
            urgency=0.7,
            complexity=0.5,
            related_nodes=[node.id]
        )
        
    def _prioritize_questions(self, questions: List[Question]) -> List[Question]:
        if not questions:
            return []
        
        priorities = [
            (self._calculate_priority(q), q) for q in questions
        ]
        priorities.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in priorities]
    
    def _calculate_priority(self, question: Question) -> float:
        return (
            question.importance * 0.4 +
            question.urgency * 0.3 +
            (question.complexity * self.complexity_preference) * 0.3
        )
    
    def _update_domain_interest_by_id(self, question_id: str, reward: float) -> None:
        for domain in KnowledgeDomain:
            if domain.value in question_id:
                learning_rate = 0.1
                self.interest_areas[domain] = np.clip(
                    self.interest_areas[domain] + (reward * learning_rate),
                    0.0,
                    1.0
                )
                break
    
    def _update_question_importance(self, question: Question, reward: float) -> None:
        learning_rate = 0.15
        question.importance = np.clip(
            question.importance + (reward * learning_rate),
            0.0,
            1.0
        )
    
    def _update_domain_interest(self, domain: KnowledgeDomain, reward: float) -> None:
        domain_learning_rate = 0.1
        self.interest_areas[domain] = np.clip(
            self.interest_areas[domain] + (reward * domain_learning_rate),
            self.MIN_DOMAIN_INTEREST,
            self.MAX_DOMAIN_INTEREST
        )
    
    def _update_complexity_preference(self, complexity: float, reward: float) -> None:
        if reward > 0.3:
            complexity_influence = (complexity - 0.5) * reward * 0.05
            self.complexity_preference = np.clip(
                self.complexity_preference + complexity_influence,
                0.0,
                1.0
            )
    
    def _update_exploration_rate(self, reward: float) -> None:
        if reward > 0.5:
            self.exploration_rate = min(self.exploration_rate + 0.02, 0.9)
        elif reward < 0.0:
            self.exploration_rate = max(self.exploration_rate - 0.02, 0.3)
    
    def _mark_question_answered(self, question_id: str, question: Question) -> None:
        question.status = "answered"
        if question_id not in self.question_history:
            self.question_history.append(question_id)