#!/usr/bin/env python3
"""Direct test script for the emotional layer of HUMAN 2.0."""

import os
import sys
from pathlib import Path
from components.emotional_memory import EmotionalMemory

def main():
    """Test the emotional layer directly."""
    # Set up data directory
    base_dir = Path.home() / '.human2'
    data_dir = base_dir / 'data' / 'emotional_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing EmotionalMemory component...")
    emotional_memory = EmotionalMemory(str(data_dir))
    
    # Test express emotion
    print("\n=== Testing Express Emotion ===")
    result = emotional_memory.process_interaction("Expressing happy emotion with high intensity")
    print(f"Emotional state: {result['emotional_state']}")
    print(f"Response: {result['emotional_response']}")
    
    # Test respond to emotion
    print("\n=== Testing Respond to Emotion ===")
    result = emotional_memory.simulate_empathy("sad")
    print(f"Empathetic response: {result}")
    
    # Test learn from emotion
    print("\n=== Testing Learn from Emotion ===")
    result = emotional_memory.process_interaction("Learning about fear emotion with medium intensity")
    print(f"Learned from experience: Learning about fear emotion with medium intensity")
    print(f"Updated emotional state: {result['emotional_state']}")
    
    # Test emotional emergence
    print("\n=== Testing Emotional Emergence ===")
    emergence = emotional_memory.get_emotional_emergence()
    if emergence['emergence_detected']:
        print(f"Emergence type: {emergence['emergence_type']}")
        print(f"Stability: {emergence['emotional_stability']:.2f}")
        print(f"Adaptation: {emergence['emotional_adaptation']:.2f}")
        print(f"Resilience: {emergence['emotional_resilience']:.2f}")
    else:
        print("Not enough emotional history to detect emergence patterns.")
    
    # Test personality traits
    print("\n=== Testing Personality Traits ===")
    print(f"Personality traits: {emotional_memory.personality_traits}")
    print(f"Empathy level: {emotional_memory._calculate_empathy_level():.2f}")
    
    print("\nEmotional layer test completed successfully!")

if __name__ == '__main__':
    main() 