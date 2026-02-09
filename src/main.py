#!/usr/bin/env python3
"""
HUMAN 2.0 - Main Entry Point

This is the main entry point for the HUMAN 2.0 artificial general intelligence system.
It initializes all core components and provides a unified interface for interaction.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.human import HUMAN
from components.multimodal_emotion_processor import MultimodalEmotionProcessor
from consciousness.self_awareness import SelfAwarenessSystem
from components.emotional_memory import EmotionalMemory
from components.emotional_learning import EmotionalLearningSystem

class HUMAN2System:
    """Main HUMAN 2.0 system class that integrates all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the HUMAN 2.0 system.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.logger.info("Initializing HUMAN 2.0 System...")
        
        # Initialize core HUMAN class
        self.human = HUMAN()
        
        # Initialize emotional intelligence components
        self.emotion_processor = MultimodalEmotionProcessor()
        self.emotional_memory = EmotionalMemory()
        self.emotional_learning = EmotionalLearningSystem()
        
        # Initialize consciousness components
        self.self_awareness = SelfAwarenessSystem()
        
        # System state
        self.is_running = False
        self.start_time = None
        self.interaction_count = 0
        
        self.logger.info("HUMAN 2.0 System initialized successfully")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('HUMAN2')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('human2.log')
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def start(self) -> bool:
        """Start the HUMAN 2.0 system."""
        try:
            self.logger.info("Starting HUMAN 2.0 System...")
            
            # Initialize core HUMAN system
            if not self.human.initialize():
                self.logger.error("Failed to initialize core HUMAN system")
                return False
                
            # Initialize emotional systems
            self.emotional_memory.initialize()
            self.emotional_learning.initialize()
            
            # Initialize consciousness systems
            self.self_awareness.initialize()
            
            # Start system
            self.is_running = True
            self.start_time = time.time()
            
            self.logger.info("HUMAN 2.0 System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start HUMAN 2.0 System: {str(e)}")
            return False
            
    def stop(self):
        """Stop the HUMAN 2.0 system."""
        self.logger.info("Stopping HUMAN 2.0 System...")
        self.is_running = False
        
        # Cleanup
        self.emotional_memory.cleanup()
        self.emotional_learning.cleanup()
        
        self.logger.info("HUMAN 2.0 System stopped")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through all systems.
        
        Args:
            input_data: Dictionary containing input data
                - text: Text input
                - audio: Audio input (optional)
                - visual: Visual input (optional)
                - context: Additional context (optional)
                
        Returns:
            Dictionary containing system response
        """
        if not self.is_running:
            return {"error": "System not running"}
            
        try:
            self.interaction_count += 1
            self.logger.info(f"Processing interaction #{self.interaction_count}")
            
            # Process through emotional intelligence
            emotion_result = self.emotion_processor.process_multimodal_input(**input_data)
            
            # Update emotional memory
            self.emotional_memory.add_experience({
                "input": input_data,
                "emotion": emotion_result,
                "timestamp": time.time()
            })
            
            # Update consciousness
            self.self_awareness.add_experience({
                "type": "input_processed",
                "content": input_data,
                "emotion": emotion_result.dominant_emotion,
                "confidence": emotion_result.confidence
            })
            
            # Update core HUMAN system
            self.human.process_input(input_data)
            
            # Generate response
            response = self._generate_response(input_data, emotion_result)
            
            return {
                "success": True,
                "response": response,
                "emotion": {
                    "dominant": emotion_result.dominant_emotion,
                    "confidence": emotion_result.confidence,
                    "intensity": emotion_result.emotional_intensity
                },
                "consciousness_state": self.self_awareness.get_current_state(),
                "interaction_id": self.interaction_count
            }
            
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return {"error": str(e)}
            
    def _generate_response(self, input_data: Dict[str, Any], emotion_result) -> str:
        """Generate appropriate response based on input and emotion."""
        # Simple response generation - can be enhanced
        text_input = input_data.get("text", "")
        
        if emotion_result.dominant_emotion in ["joy", "excitement", "happiness"]:
            return f"I sense your {emotion_result.dominant_emotion}! That's wonderful. Tell me more about what's making you feel this way."
        elif emotion_result.dominant_emotion in ["sadness", "disappointment", "grief"]:
            return f"I understand you're feeling {emotion_result.dominant_emotion}. I'm here to listen and support you."
        elif emotion_result.dominant_emotion in ["anger", "frustration", "irritation"]:
            return f"I can see you're feeling {emotion_result.dominant_emotion}. What's been happening that's causing this?"
        elif emotion_result.dominant_emotion in ["fear", "anxiety", "worry"]:
            return f"I notice you're experiencing {emotion_result.dominant_emotion}. Let's talk about what's concerning you."
        else:
            return f"I understand you said: '{text_input}'. I'm processing this with my emotional intelligence systems."
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "interaction_count": self.interaction_count,
            "human_state": self.human.get_state(),
            "consciousness_state": self.self_awareness.get_current_state(),
            "emotional_memory_size": len(self.emotional_memory.memories),
            "emotional_learning_patterns": len(self.emotional_learning.patterns)
        }
        
    def run_interactive_mode(self):
        """Run HUMAN 2.0 in interactive mode."""
        print("🤖 HUMAN 2.0 - Interactive Mode")
        print("=" * 50)
        print("Type 'quit' to exit, 'status' for system status")
        print()
        
        while self.is_running:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\nSystem Status:")
                    print(f"- Running: {status['is_running']}")
                    print(f"- Uptime: {status['uptime']:.1f} seconds")
                    print(f"- Interactions: {status['interaction_count']}")
                    print(f"- Dominant Emotion: {status['human_state']['global_workspace']['emotional_state']}")
                    print()
                    continue
                    
                # Process input
                result = self.process_input({"text": user_input})
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"🤖 HUMAN 2.0: {result['response']}")
                    print(f"   Emotion: {result['emotion']['dominant']} (confidence: {result['emotion']['confidence']:.2f})")
                    
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                
        self.stop()

def main():
    """Main entry point."""
    print("🚀 Starting HUMAN 2.0...")
    
    # Initialize system
    human2 = HUMAN2System()
    
    # Start system
    if not human2.start():
        print("❌ Failed to start HUMAN 2.0")
        return 1
        
    try:
        # Run interactive mode
        human2.run_interactive_mode()
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        return 1
    finally:
        human2.stop()
        
    print("👋 HUMAN 2.0 shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())