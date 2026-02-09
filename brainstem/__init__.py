# brainstem/__init__.py

# Import core AI modules
from .brainstem_agent import deepseek_r1, llama_70b
from .memory_manager import MemoryManager

# Import knowledge processor (ensure the file exists)
from .knowledge_processor import KnowledgeProcessor

# Initialize modules
memory_manager = MemoryManager()
knowledge_processor = KnowledgeProcessor()

# Print for debugging
print("Brainstem package initialized successfully.")
