#!/usr/bin/env python3
"""Test script for the emotional layer of HUMAN 2.0."""

import os
import sys
import argparse
from pathlib import Path
from components.emotional_memory import EmotionalMemory

def test_emotional_memory(base_dir=None):
    """Test the EmotionalMemory component."""
    if base_dir is None:
        base_dir = Path.home() / '.human2'
    else:
        base_dir = Path(base_dir)
        
    data_dir = base_dir / 'data' / 'emotional_memory'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing EmotionalMemory component...")
    emotional_memory = EmotionalMemory(str(data_dir))
    
    # Test basic emotion processing
    print("\n=== Testing Basic Emotion Processing ===")
    test_inputs = [
        "I am feeling very happy today!",
        "This makes me sad and disappointed.",
        "I'm really angry about this situation.",
        "I'm scared of what might happen next.",
        "I'm surprised by this unexpected turn of events."
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\nTest {i}: '{text}'")
        result = emotional_memory.process_interaction(text)
        print(f"Emotional state: {result['emotional_state']}")
        print(f"Response: {result['emotional_response']}")
        
    # Test empathy simulation
    print("\n=== Testing Empathy Simulation ===")
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise']
    for emotion in emotions:
        response = emotional_memory.simulate_empathy(emotion)
        print(f"\nEmpathy for {emotion}: {response}")
        
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

def main():
    """Run the emotional layer test."""
    parser = argparse.ArgumentParser(description='Test the emotional layer of HUMAN 2.0')
    parser.add_argument('--base-dir', help='Base directory for data storage')
    args = parser.parse_args()
    
    try:
        test_emotional_memory(args.base_dir)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 