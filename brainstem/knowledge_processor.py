from typing import Dict, List, Any
import logging

class KnowledgeProcessor:
    def __init__(self):
        """Initialize the knowledge processor."""
        self.knowledge_base: Dict[str, Any] = {}
        
    def process_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Process and store new knowledge.
        
        Args:
            knowledge: Dictionary containing knowledge data
        """
        try:
            # Merge new knowledge with existing knowledge base
            self._merge_knowledge(knowledge)
            logging.info("Knowledge processed successfully")
        except Exception as e:
            logging.error(f"Error processing knowledge: {e}")
            
    def _merge_knowledge(self, new_knowledge: Dict[str, Any]) -> None:
        """Merge new knowledge with existing knowledge base.
        
        Args:
            new_knowledge: New knowledge to merge
        """
        for key, value in new_knowledge.items():
            if key in self.knowledge_base:
                if isinstance(value, list):
                    self.knowledge_base[key].extend(value)
                elif isinstance(value, dict):
                    self.knowledge_base[key].update(value)
                else:
                    self.knowledge_base[key] = value
            else:
                self.knowledge_base[key] = value
                
    def get_knowledge(self, key: str) -> Any:
        """Retrieve knowledge by key.
        
        Args:
            key: Key to retrieve knowledge for
            
        Returns:
            Knowledge value for the given key
        """
        return self.knowledge_base.get(key)
        
    def get_all_knowledge(self) -> Dict[str, Any]:
        """Retrieve all knowledge.
        
        Returns:
            Complete knowledge base
        """
        return self.knowledge_base.copy()
