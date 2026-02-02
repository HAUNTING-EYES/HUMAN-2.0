import logging
from typing import Dict, Any, List, Optional, Callable
import networkx as nx
from datetime import datetime
import re
from transformers import pipeline

class LogicalReasoning:
    """System for logical reasoning and inference."""
    
    def __init__(self):
        """Initialize the logical reasoning system."""
        self.logger = logging.getLogger(__name__)
        self.causal_graph = {}
        self.symbolic_rules = {}
        self.confidence_threshold = 0.7
        self.rules = []
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.concept_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|(?:(?<=\s)|^)[a-z]+(?:\s+[a-z]+)*(?=\s|$)'
        self.relationship_patterns = {
            'is_a': r'(?:is a|are a|is an|are an)',
            'has_a': r'(?:has|have|contains?|includes?)',
            'causes': r'(?:causes?|leads? to|results? in)',
            'implies': r'(?:implies?|suggests?|indicates?)'
        }
        
    def reason(self, context: dict, query: str) -> dict:
        """
        Perform logical reasoning based on context and query.
        
        Args:
            context (dict): Context information including task and parameters
            query (str): The query to reason about
            
        Returns:
            dict: Result containing reasoning chain, confidence, and applied rules
        """
        try:
            # Initialize result structure
            confidence = 1.0
            applied_rules = []
            reasoning_chain = []
            
            # Check for unknowns in context that reduce confidence
            if any(v == "unknown" for v in context.get("parameters", {}).values()):
                confidence *= 0.5
                
            # Apply symbolic rules
            for rule_id, rule in self.symbolic_rules.items():
                if all(cond in str(context).lower() for cond in rule["conditions"]):
                    applied_rules.append(rule_id)
                    reasoning_chain.append(f"Applied rule: {rule['description']}")
                    
            # Apply causal reasoning
            for cause, effects in self.causal_graph.items():
                if cause in str(context).lower():
                    for effect, prob in effects.items():
                        confidence *= prob
                        reasoning_chain.append(f"Causal inference: {cause} leads to {effect}")
                        
            # Determine status based on confidence
            status = "certain" if confidence >= 0.7 else "uncertain"
            
            return {
                "result": {
                    "status": status,
                    "applied_rules": applied_rules
                },
                "confidence": confidence,
                "reasoning_chain": reasoning_chain
            }
            
        except Exception as e:
            return {
                "result": {
                    "status": "error",
                    "applied_rules": []
                },
                "confidence": 0.0,
                "reasoning_chain": [],
                "error": str(e)
            }
            
    def add_symbolic_rule(self, rule: dict) -> None:
        """
        Add a symbolic rule to the reasoning system.
        
        Args:
            rule (dict): Rule containing id, description, conditions, and conclusion
        """
        self.symbolic_rules[rule["id"]] = rule
        
    def update_causal_graph(self, cause: str, effect: str, probability: float) -> None:
        """
        Update the causal graph with a new relationship.
        
        Args:
            cause (str): The cause event/condition
            effect (str): The effect event/condition
            probability (float): Probability of the causal relationship
        """
        if cause not in self.causal_graph:
            self.causal_graph[cause] = {}
        self.causal_graph[cause][effect] = probability
        
    def _generate_reasoning_chain(self, premises: list, conclusion: str, valid_steps: list) -> list:
        """Generate a chain of reasoning steps."""
        chain = []
        
        # Add premises as initial steps
        for i, premise in enumerate(premises, 1):
            chain.append({
                "step": i,
                "type": "premise",
                "content": premise
            })
        
        # Add reasoning steps
        for i, step in enumerate(valid_steps, len(premises) + 1):
            chain.append({
                "step": i,
                "type": "reasoning",
                "content": step
            })
        
        # Add conclusion
        chain.append({
            "step": len(chain) + 1,
            "type": "conclusion",
            "content": conclusion
        })
        
        return chain
        
    def _extract_premises(self, context: dict) -> list:
        """Extract premises from the context."""
        premises = []
        
        # Extract from task
        if "task" in context:
            premises.append(f"Task is {context['task']}")
        
        # Extract from parameters
        if "parameters" in context:
            for key, value in context["parameters"].items():
                premises.append(f"{key} is {value}")
        
        # Add causal relationships as premises
        for cause, effects in self.causal_graph.items():
            for effect, prob in effects.items():
                if prob > 0.5:
                    premises.append(f"{cause} likely leads to {effect}")
        
        return premises

    def _extract_conclusion(self, query: str) -> str:
        """Extract potential conclusion from the query."""
        # Use zero-shot classification to determine if query is asking for approach/solution
        result = self.zero_shot_classifier(
            query,
            candidate_labels=["asking_for_approach", "asking_for_solution", "other"]
        )
        
        if result["labels"][0] in ["asking_for_approach", "asking_for_solution"]:
            return "Recommend best practices based on context"
        return query

    def _check_modus_ponens(self, premises: list, conclusion: str) -> bool:
        """Check if modus ponens rule applies."""
        for premise in premises:
            if "leads to" in premise and premise.split("leads to")[1].strip() in conclusion:
                return True
        return False

    def _check_modus_tollens(self, premises: list, conclusion: str) -> bool:
        """Check if modus tollens rule applies."""
        for premise in premises:
            if "leads to" in premise and "not" in conclusion:
                return True
        return False

    def _check_hypothetical_syllogism(self, premises: list, conclusion: str) -> bool:
        """Check if hypothetical syllogism rule applies."""
        # Look for chain of implications
        implications = [p for p in premises if "leads to" in p]
        if len(implications) >= 2:
            return True
        return False

    def add_rule(self, name: str, rule_func: Callable) -> None:
        """Add a new reasoning rule.
        
        Args:
            name: Name of the rule
            rule_func: Function implementing the rule
        """
        self.rules.append(name)

    def _derive_conclusion(self, reasoning_chain: List[str]) -> str:
        """Derive conclusion from reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            
        Returns:
            Conclusion string
        """
        if not reasoning_chain:
            return "Insufficient information to draw a conclusion"
        
        # Combine reasoning steps into conclusion
        conclusion = "Based on the following reasoning:\n"
        for i, step in enumerate(reasoning_chain, 1):
            conclusion += f"{i}. {step}\n"
        return conclusion 

    def process_reasoning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input text for logical reasoning."""
        try:
            if not isinstance(input_data, dict) or 'text' not in input_data:
                return {
                    'success': False,
                    'error': 'Invalid input format - expected dict with text key',
                    'confidence': 0.0
                }

            text = input_data['text']
            context = input_data.get('context', '')
            
            # Extract logical components
            premises = self._extract_premises(context)
            conclusion = self._extract_conclusion(text)
            
            # Apply logical rules
            applied_rules = []
            valid_steps = []
            
            # Check for valid logical steps
            if premises and conclusion:
                # Check for modus ponens
                if self._check_modus_ponens(premises, conclusion):
                    valid_steps.append('modus_ponens')
                    applied_rules.append('If P then Q, P therefore Q')
                
                # Check for modus tollens
                if self._check_modus_tollens(premises, conclusion):
                    valid_steps.append('modus_tollens')
                    applied_rules.append('If P then Q, not Q therefore not P')
                
                # Check for hypothetical syllogism
                if self._check_hypothetical_syllogism(premises, conclusion):
                    valid_steps.append('hypothetical_syllogism')
                    applied_rules.append('If P then Q, if Q then R, therefore if P then R')
            
            # Calculate confidence based on valid steps and context
            confidence = self._calculate_confidence(valid_steps, context)
            
            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(premises, conclusion, valid_steps)
            
            return {
                'success': True,
                'premises': premises,
                'conclusion': conclusion,
                'confidence': confidence,
                'reasoning_chain': reasoning_chain,
                'applied_rules': applied_rules
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'applied_rules': []
            }

    def _calculate_confidence(self, valid_steps: List[str], context: str) -> float:
        """Calculate confidence score based on valid reasoning steps and context."""
        if not valid_steps:
            return 0.0
            
        # Base confidence from valid logical steps
        step_confidence = len(valid_steps) * 0.3
        
        # Context relevance boost
        context_boost = 0.2 if context else 0.0
        
        # Calculate final confidence
        confidence = min(step_confidence + context_boost, 1.0)
        
        return confidence

    def _calculate_confidence(self, valid_steps: List[str], context: str) -> float:
        """Calculate confidence score based on valid reasoning steps and context."""
        if not valid_steps:
            return 0.0
            
        # Base confidence from valid logical steps
        step_confidence = len(valid_steps) * 0.3
        
        # Context relevance boost
        context_boost = 0.2 if context else 0.0
        
        # Calculate final confidence
        confidence = min(step_confidence + context_boost, 1.0)
        
        return confidence

    def _extract_premises(self, text: str) -> List[str]:
        """Extract premises from a text."""
        # Split text into sentences
        sentences = text.split('.')
        premises = []
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            # Look for premise indicators
            if any(indicator in sentence for indicator in ['given that', 'since', 'because', 'as']):
                premises.append(sentence)
            # Look for if-then statements
            elif 'if' in sentence and 'then' in sentence:
                premises.append(sentence)
            # Look for statements before 'therefore' or similar
            elif any(indicator in sentence for indicator in ['thus', 'hence', 'so']):
                # Add the part before the indicator
                parts = sentence.split(indicator)
                if len(parts) > 1:
                    premises.append(parts[0].strip())
        
        return premises

    def _extract_conclusion(self, text: str) -> str:
        """Extract conclusion from a text."""
        text = text.lower()
        conclusion = ""
        
        # Look for conclusion indicators
        indicators = ['therefore', 'thus', 'hence', 'consequently', 'so']
        for indicator in indicators:
            if indicator in text:
                parts = text.split(indicator)
                if len(parts) > 1:
                    conclusion = parts[1].strip()
                    break
        
        # If no explicit indicator, look for the final statement after premises
        if not conclusion and 'if' in text and 'then' in text:
            parts = text.split('then')
            if len(parts) > 1:
                conclusion = parts[-1].strip()
        
        return conclusion

    def _check_modus_ponens(self, premises: List[str], conclusion: str) -> bool:
        """Check if the text follows modus ponens (If P then Q, P therefore Q)."""
        if not premises or not conclusion:
            return False
            
        # Look for if-then statement and its antecedent
        has_conditional = False
        antecedent = None
        consequent = None
        
        for premise in premises:
            if 'if' in premise and 'then' in premise:
                has_conditional = True
                parts = premise.split('if')[1].split('then')
                if len(parts) == 2:
                    antecedent = parts[0].strip()
                    consequent = parts[1].strip()
                    break
        
        # Check if another premise matches the antecedent
        has_antecedent = False
        if antecedent:
            for premise in premises:
                if antecedent in premise and 'if' not in premise:
                    has_antecedent = True
                    break
        
        # Check if conclusion matches consequent
        matches_consequent = consequent and consequent in conclusion
        
        return has_conditional and has_antecedent and matches_consequent

    def _check_modus_tollens(self, premises: List[str], conclusion: str) -> bool:
        """Check if the text follows modus tollens (If P then Q, not Q therefore not P)."""
        if not premises or not conclusion:
            return False
            
        # Look for if-then statement
        has_conditional = False
        antecedent = None
        consequent = None
        
        for premise in premises:
            if 'if' in premise and 'then' in premise:
                has_conditional = True
                parts = premise.split('if')[1].split('then')
                if len(parts) == 2:
                    antecedent = parts[0].strip()
                    consequent = parts[1].strip()
                    break
        
        # Check if another premise negates the consequent
        has_negated_consequent = False
        if consequent:
            for premise in premises:
                if ('not ' + consequent) in premise or ('no ' + consequent) in premise:
                    has_negated_consequent = True
                    break
        
        # Check if conclusion negates the antecedent
        negates_antecedent = antecedent and (('not ' + antecedent) in conclusion or ('no ' + antecedent) in conclusion)
        
        return has_conditional and has_negated_consequent and negates_antecedent

    def _check_hypothetical_syllogism(self, premises: List[str], conclusion: str) -> bool:
        """Check if the text follows hypothetical syllogism (If P then Q, if Q then R, therefore if P then R)."""
        if not premises or not conclusion:
            return False
            
        # Look for two if-then statements
        conditionals = []
        for premise in premises:
            if 'if' in premise and 'then' in premise:
                parts = premise.split('if')[1].split('then')
                if len(parts) == 2:
                    antecedent = parts[0].strip()
                    consequent = parts[1].strip()
                    conditionals.append((antecedent, consequent))
        
        # Need exactly two conditional statements
        if len(conditionals) != 2:
            return False
            
        # Check if consequent of first matches antecedent of second
        if conditionals[0][1] == conditionals[1][0]:
            # Check if conclusion links first antecedent to second consequent
            expected = f"if {conditionals[0][0]} then {conditionals[1][1]}"
            return expected.lower() in conclusion.lower()
            
        return False

    def _generate_reasoning_chain(self, premises: List[str], conclusion: str, valid_steps: List[str]) -> List[Dict[str, Any]]:
        """Generate a chain of reasoning steps."""
        chain = []
        
        # Add premises
        for i, premise in enumerate(premises, 1):
            chain.append({
                'step': i,
                'type': 'premise',
                'text': premise
            })
        
        # Add logical steps
        step_num = len(premises) + 1
        for step in valid_steps:
            chain.append({
                'step': step_num,
                'type': 'logical_rule',
                'text': f'Applied {step}'
            })
            step_num += 1
        
        # Add conclusion
        if conclusion:
            chain.append({
                'step': step_num,
                'type': 'conclusion',
                'text': conclusion
            })
        
        return chain 