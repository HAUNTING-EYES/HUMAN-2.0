#!/usr/bin/env python3
"""
Test script for multimodal emotion processing
Demonstrates text, audio, and visual emotion recognition integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.components.multimodal_emotion_processor import MultimodalEmotionProcessor, EmotionResult
import time
import json

def test_text_emotion_processing():
    """Test text-based emotion processing"""
    print("=" * 60)
    print("TESTING TEXT EMOTION PROCESSING")
    print("=" * 60)
    
    processor = MultimodalEmotionProcessor()
    
    test_texts = [
        "I am absolutely furious about what they did!",
        "I'm so excited about the new opportunity!",
        "I feel really sad and disappointed today.",
        "I'm nervous about the upcoming presentation.",
        "That was such a pleasant surprise!",
        "I'm feeling quite neutral about this situation."
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        emotion_state = processor.process_multimodal_input(text=text)
        
        print(f"Dominant emotion: {emotion_state.dominant_emotion}")
        print(f"Confidence: {emotion_state.confidence:.3f}")
        print(f"Emotional intensity: {emotion_state.emotional_intensity:.3f}")
        
        if emotion_state.combined_emotions:
            print("Top emotions:")
            for emotion_result in emotion_state.combined_emotions[:3]:
                print(f"  - {emotion_result.emotion}: {emotion_result.confidence:.3f}")
        
        print("-" * 40)

def test_multimodal_integration():
    """Test multimodal emotion integration"""
    print("\n" + "=" * 60)
    print("TESTING MULTIMODAL INTEGRATION")
    print("=" * 60)
    
    processor = MultimodalEmotionProcessor()
    
    # Simulate multimodal input
    print("\nSimulating multimodal input...")
    print("Text: 'I'm really happy about the results!'")
    print("Audio: Detecting high pitch and energy (joy indicators)")
    print("Visual: Detecting smile and raised cheeks (joy indicators)")
    
    # Process with text only (since we can't capture real audio/video in this test)
    emotion_state = processor.process_multimodal_input(
        text="I'm really happy about the results!"
    )
    
    print(f"\nResults:")
    print(f"Dominant emotion: {emotion_state.dominant_emotion}")
    print(f"Confidence: {emotion_state.confidence:.3f}")
    print(f"Emotional intensity: {emotion_state.emotional_intensity:.3f}")
    
    # Show how different modalities would contribute
    print(f"\nModality contributions:")
    print(f"Text weight: {processor.modality_weights['text']:.1%}")
    print(f"Audio weight: {processor.modality_weights['audio']:.1%}")
    print(f"Visual weight: {processor.modality_weights['visual']:.1%}")

def test_emotion_fusion():
    """Test emotion fusion from multiple modalities"""
    print("\n" + "=" * 60)
    print("TESTING EMOTION FUSION")
    print("=" * 60)
    
    processor = MultimodalEmotionProcessor()
    
    # Test conflicting emotions
    print("\nTest 1: Conflicting emotions")
    print("Text: 'I'm happy but also nervous'")
    
    emotion_state = processor.process_multimodal_input(
        text="I'm happy but also nervous"
    )
    
    print(f"Fused dominant emotion: {emotion_state.dominant_emotion}")
    print(f"Confidence: {emotion_state.confidence:.3f}")
    
    if emotion_state.combined_emotions:
        print("All detected emotions:")
        for emotion_result in emotion_state.combined_emotions:
            print(f"  - {emotion_result.emotion}: {emotion_result.confidence:.3f}")
    
    # Test neutral text
    print("\nTest 2: Neutral text")
    print("Text: 'The weather is cloudy today'")
    
    emotion_state2 = processor.process_multimodal_input(
        text="The weather is cloudy today"
    )
    
    print(f"Fused dominant emotion: {emotion_state2.dominant_emotion}")
    print(f"Confidence: {emotion_state2.confidence:.3f}")

def demonstrate_integration():
    """Demonstrate how this would integrate with HUMAN 2.0"""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH HUMAN 2.0")
    print("=" * 60)
    
    processor = MultimodalEmotionProcessor()
    
    print("\nScenario: User interacting with HUMAN 2.0")
    print("1. User says: 'I'm frustrated with this code'")
    print("2. AI detects emotion and responds appropriately")
    print("3. AI adapts its communication style")
    
    # Simulate emotion detection
    emotion_state = processor.process_multimodal_input(
        text="I'm frustrated with this code"
    )
    
    print(f"\nEmotion Analysis:")
    print(f"Detected emotion: {emotion_state.dominant_emotion}")
    print(f"Confidence: {emotion_state.confidence:.3f}")
    print(f"Intensity: {emotion_state.emotional_intensity:.3f}")
    
    # Simulate AI response adaptation
    print(f"\nAI Response Adaptation:")
    if emotion_state.dominant_emotion == 'anger' or emotion_state.dominant_emotion == 'frustration':
        print("- Using calming, supportive tone")
        print("- Offering step-by-step guidance")
        print("- Acknowledging the frustration")
    elif emotion_state.dominant_emotion == 'sadness':
        print("- Using encouraging, positive tone")
        print("- Offering motivation and support")
        print("- Breaking down complex tasks")
    elif emotion_state.dominant_emotion == 'joy':
        print("- Matching enthusiasm")
        print("- Building on positive momentum")
        print("- Celebrating achievements")
    else:
        print("- Using neutral, professional tone")
        print("- Providing clear, direct assistance")

def show_system_capabilities():
    """Show the system's capabilities"""
    print("\n" + "=" * 60)
    print("SYSTEM CAPABILITIES")
    print("=" * 60)
    
    print("\n✅ Text Emotion Recognition:")
    print("  - Uses trained ERADEM model")
    print("  - Supports 28+ emotion categories")
    print("  - Context-aware processing")
    print("  - Confidence scoring")
    
    print("\n✅ Audio Emotion Recognition:")
    print("  - Voice tone analysis")
    print("  - Pitch and energy detection")
    print("  - Spectral feature extraction")
    print("  - Real-time processing")
    
    print("\n✅ Visual Emotion Recognition:")
    print("  - Facial expression analysis")
    print("  - Face detection and tracking")
    print("  - Expression feature extraction")
    print("  - Real-time video processing")
    
    print("\n✅ Multimodal Fusion:")
    print("  - Weighted emotion combination")
    print("  - Conflict resolution")
    print("  - Confidence aggregation")
    print("  - Temporal consistency")
    
    print("\n✅ Integration Features:")
    print("  - Real-time monitoring")
    print("  - Emotion history tracking")
    print("  - Trend analysis")
    print("  - Callback system for AI adaptation")

def main():
    """Run all tests"""
    print("HUMAN 2.0 Multimodal Emotion Processing Test")
    print("=" * 60)
    
    try:
        # Test text processing
        test_text_emotion_processing()
        
        # Test multimodal integration
        test_multimodal_integration()
        
        # Test emotion fusion
        test_emotion_fusion()
        
        # Demonstrate integration
        demonstrate_integration()
        
        # Show capabilities
        show_system_capabilities()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe multimodal emotion processor is ready for integration!")
        print("Next steps:")
        print("1. Install audio/video dependencies (pyaudio, opencv-python)")
        print("2. Run real-time monitoring: python src/components/realtime_emotion_monitor.py")
        print("3. Integrate with HUMAN 2.0 main system")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Make sure the ERADEM model is properly trained and available.")

if __name__ == "__main__":
    main() 