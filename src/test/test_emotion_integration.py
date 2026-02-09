#!/usr/bin/env python3
"""Integration test script for the emotional layer of HUMAN 2.0."""

import os
import sys
import json
from pathlib import Path
from components.emotional_memory import EmotionalMemory
import re
from datetime import datetime
from typing import Dict, Any, Tuple

class EmotionTestInterface:
    def __init__(self):
        print("Initializing EmotionalMemory component...")
        self.emotional_memory = EmotionalMemory()
        
    def format_emotional_state(self, state: Dict[str, Any]) -> str:
        """Format emotional state for display."""
        output = "\nCurrent Emotional State:\n"
        output += "=" * 30 + "\n"
        
        # Core emotions
        output += "Core Emotions:\n"
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise']:
            value = state.get(emotion, 0)
            bar = "█" * int(value * 20)
            output += f"{emotion.capitalize():10} [{bar:<20}] {value:.2f}\n"
            
        # Dimensional emotions
        output += "\nDimensional Emotions:\n"
        for dim in ['valence', 'arousal', 'dominance']:
            value = state.get(dim, 0)
            bar = "█" * int((value + 1) * 10)  # Scale from -1 to 1
            output += f"{dim.capitalize():10} [{bar:<20}] {value:.2f}\n"
            
        output += f"\nTimestamp: {state.get('timestamp', 'N/A')}\n"
        return output
        
    def parse_natural_language(self, text: str) -> Tuple[str, str]:
        """Parse natural language input to extract emotion and intensity."""
        text = text.lower()
        
        # Expanded emotion keywords mapping
        emotion_map = {
            'happy': 'joy',
            'joy': 'joy',
            'glad': 'joy',
            'excited': 'joy',
            'sad': 'sadness',
            'sadness': 'sadness',
            'unhappy': 'sadness',
            'down': 'sadness',
            'angry': 'anger',
            'anger': 'anger',
            'mad': 'anger',
            'frustrated': 'anger',
            'scared': 'fear',
            'fear': 'fear',
            'afraid': 'fear',
            'worried': 'fear',
            'surprised': 'surprise',
            'surprise': 'surprise',
            'amazed': 'surprise',
            'shocked': 'surprise'
        }
        
        # Expanded intensity keywords
        intensity_map = {
            'very': 'high',
            'really': 'high',
            'extremely': 'high',
            'totally': 'high',
            'completely': 'high',
            'so': 'high',
            'slightly': 'low',
            'a bit': 'low',
            'a little': 'low',
            'kind of': 'low',
            'somewhat': 'medium',
            'quite': 'medium',
            'pretty': 'medium',
            'moderately': 'medium'
        }
        
        # Default values
        emotion = None
        intensity = 'medium'
        
        # Find emotion with word boundary check
        for keyword, mapped_emotion in emotion_map.items():
            if f" {keyword} " in f" {text} ":
                emotion = mapped_emotion
                break
                
        # Find intensity with word boundary check
        for keyword, mapped_intensity in intensity_map.items():
            if f" {keyword} " in f" {text} ":
                intensity = mapped_intensity
                break
                
        return emotion, intensity
        
    def show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("  express <emotion> <intensity> - Express an emotion")
        print("  respond <emotion>             - Respond to an emotion")
        print("  learn <emotion> <intensity>   - Learn from an emotional experience")
        print("  state                         - Show current emotional state")
        print("  emergence                     - Show emotional emergence patterns")
        print("  personality                   - Show personality traits")
        print("  history                       - Show interaction history")
        print("  help                          - Show this help message")
        print("  exit                          - Exit the interface")
        print("\nYou can also use natural language like:")
        print("  'I am feeling very happy'")
        print("  'I am somewhat sad'")
        print("  'I am really angry'")
        print("  'I feel a bit scared'")
        print("  'I'm totally surprised'")
        print("\nEmotions: joy/happy/glad, sadness/sad/unhappy, anger/angry/mad,")
        print("          fear/scared/afraid, surprise/surprised/amazed")
        print("Intensities: slightly/a bit/kind of (low),")
        print("            somewhat/quite/pretty (medium),")
        print("            very/really/extremely (high)")
        
    def run(self):
        """Run the interactive test interface."""
        print("\n=== HUMAN 2.0 Emotional Layer Test ===")
        print("Type 'help' for available commands or 'exit' to quit\n")
        
        while True:
            try:
                cmd = input("EMOTION> ").strip()
                
                if not cmd:
                    continue
                    
                if cmd == 'exit':
                    break
                    
                if cmd == 'help':
                    self.show_help()
                    continue
                    
                # Handle natural language input
                if cmd.startswith(('i am', 'i feel', 'i\'m feeling', 'i\'m')):
                    emotion, intensity = self.parse_natural_language(cmd)
                    if emotion:
                        print(f"Detected emotion: {emotion} with intensity: {intensity}")
                        self.emotional_memory.update_emotional_state(emotion, intensity)
                        current_state = self.emotional_memory.emotional_state
                        print(self.format_emotional_state(current_state))
                        response = self.emotional_memory.generate_response(emotion)
                        print(f"\nResponse: {response}")
                        continue
                    else:
                        print("I couldn't detect a clear emotion in that statement.")
                        continue
                
                # Parse command
                parts = cmd.split()
                if not parts:
                    continue
                    
                cmd = parts[0].lower()
                
                # Handle express command
                if cmd == 'express':
                    if len(parts) < 2:
                        print("Please specify an emotion and optionally an intensity.")
                        continue
                        
                    emotion = parts[1]
                    intensity = parts[2] if len(parts) > 2 else 'medium'
                    
                    self.emotional_memory.update_emotional_state(emotion, intensity)
                    current_state = self.emotional_memory.emotional_state
                    print(self.format_emotional_state(current_state))
                    response = self.emotional_memory.generate_response(emotion)
                    print(f"\nResponse: {response}")
                    continue
                    
                # Handle respond command
                if cmd == 'respond':
                    if len(parts) < 2:
                        print("Please specify an emotion to respond to.")
                        continue
                        
                    emotion = parts[1]
                    response = self.emotional_memory.generate_response(emotion)
                    print(f"Response: {response}")
                    continue
                    
                # Handle learn command
                if cmd == 'learn':
                    if len(parts) < 2:
                        print("Please specify an emotion and optionally an intensity.")
                        continue
                        
                    emotion = parts[1]
                    intensity = parts[2] if len(parts) > 2 else 'medium'
                    
                    self.emotional_memory.learn_from_experience(emotion, intensity)
                    print("Learned from the emotional experience.")
                    continue
                    
                # Handle state command
                if cmd == 'state':
                    current_state = self.emotional_memory.emotional_state
                    print(self.format_emotional_state(current_state))
                    continue
                    
                # Handle emergence command
                if cmd == 'emergence':
                    patterns = self.emotional_memory.get_emergence_patterns()
                    print("\nEmotional Emergence Patterns:")
                    print("=" * 30)
                    for pattern in patterns:
                        print(f"- {pattern}")
                    continue
                    
                # Handle personality command
                if cmd == 'personality':
                    traits = self.emotional_memory.get_personality_traits()
                    print("\nPersonality Traits:")
                    print("=" * 30)
                    for trait, value in traits.items():
                        print(f"{trait.capitalize():15}: {value:.2f}")
                    continue
                    
                # Handle history command
                if cmd == 'history':
                    history = self.emotional_memory.get_interaction_history()
                    print("\nInteraction History:")
                    print("=" * 30)
                    for entry in history:
                        print(f"Time: {entry['timestamp']}")
                        print(f"Emotion: {entry['emotion']}")
                        print(f"Intensity: {entry['intensity']}")
                        print(f"Response: {entry['response']}")
                        print("-" * 30)
                    continue
                    
                print(f"Unknown command '{cmd}'. Type 'help' for available commands.")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

if __name__ == "__main__":
    interface = EmotionTestInterface()
    interface.run() 