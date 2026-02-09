import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

class MemoryManager:
    def __init__(self, memory_dir: Optional[Path] = None):
        """Initialize the memory manager.
        
        Args:
            memory_dir: Directory to store memories. If None, uses default 'memories' directory.
        """
        self.memory_dir = memory_dir or Path("memories")
        self.memory_dir.mkdir(exist_ok=True)
        self.memories: List[Dict[str, Any]] = []
        self._load_memories()
        
    def _load_memories(self) -> None:
        """Load existing memories from disk."""
        loaded_count = 0
        failed_count = 0
        try:
            for memory_file in sorted(self.memory_dir.glob("memory_*.json")):
                try:
                    with open(memory_file, 'r') as f:
                        memory = json.load(f)
                        self.memories.append(memory)
                        loaded_count += 1
                except json.JSONDecodeError as e:
                    failed_count += 1
                    logging.error(f"Error loading memory file {memory_file}: {e}")

            if loaded_count > 0 or failed_count > 0:
                logging.info(f"Loaded {loaded_count} memories, {failed_count} failed")
        except Exception as e:
            logging.error(f"Error loading memories: {e}")
            
    def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add a new memory to the memory store.
        
        Args:
            memory: Dictionary containing memory data
        """
        self.memories.append(memory)
        self._save_memory(memory)
        
    def _save_memory(self, memory: Dict[str, Any]) -> None:
        """Save a memory to disk.
        
        Args:
            memory: Memory to save
        """
        try:
            memory_file = self.memory_dir / f"memory_{len(self.memories)}.json"
            with open(memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
            
    def get_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve memories, optionally limited to a specific number.
        
        Args:
            limit: Maximum number of memories to return. If None, returns all memories.
            
        Returns:
            List of memory dictionaries
        """
        if limit is None:
            return self.memories
        return self.memories[-limit:]
