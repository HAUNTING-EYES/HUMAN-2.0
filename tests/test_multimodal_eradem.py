#!/usr/bin/env python3
"""
Test script for Multimodal ERADEM emotional model
"""

import sys
import os
sys.path.append('src')

from components.multimodal_emotion_processor import MultimodalEmotionProcessor
import numpy as np
import time

def test_multimodal_eradem():
    """Test the Multimodal ERADEM model with sample inputs"""
    
    print("Loading Multimodal ERADEM model...")
    
    try:
        # Initialize the multimodal processor
        processor = MultimodalEmotionProcessor()
        print(f"✓ Multimodal ERADEM loaded successfully")
        print(f"✓ Using device: {processor.device}")
        
        # Test inputs
        test_texts = [
            "I am so happy today!",
            "This makes me really angry.",
            "I feel sad and disappointed.",
            "That's absolutely disgusting!",
            "I'm feeling anxious about the test.",
            "What a beautiful surprise!",
            "I'm so frustrated with this situation.",
            "This is really exciting news!"
        ]
        
        print("\n" + "="*60)
        print("TESTING MULTIMODAL ERADEM MODEL")
        print("="*60)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Input: '{text}'")
            
            # Process text emotion
            text_emotions = processor.process_text_emotion(text)
            
            print("   Text Emotion Results:")
            for emotion_result in text_emotions:
                print(f"     - {emotion_result.emotion}: {emotion_result.confidence:.3f}")
            
            # Test multimodal processing (text only for now)
            multimodal_state = processor.process_multimodal_input(text=text)
            
            print(f"   Dominant emotion: {multimodal_state.dominant_emotion}")
            print(f"   Emotional intensity: {multimodal_state.emotional_intensity:.3f}")
            print(f"   Overall confidence: {multimodal_state.confidence:.3f}")
        
        print("\n" + "="*60)
        print("MODEL INFO")
        print("="*60)
        
        # Check text processor info
        if processor.text_processor:
            text_model = processor.text_processor['model']
            id2label = processor.text_processor['id2label']
            print(f"Text model type: {type(text_model).__name__}")
            print(f"Number of emotions: {len(id2label)}")
            print(f"Available emotions: {list(id2label.values())}")
        else:
            print("❌ Text processor not loaded")
        
        # Check audio processor
        print(f"Audio processor: {'✓ Loaded' if processor.audio_processor else '❌ Not loaded'}")
        
        # Check visual processor
        print(f"Visual processor: {'✓ Loaded' if processor.visual_processor else '❌ Not loaded'}")
        
        # Test with dummy audio and visual data
        print("\n" + "="*60)
        print("TESTING WITH DUMMY AUDIO/VISUAL DATA")
        print("="*60)
        
        # Create dummy audio data (1 second of silence at 16kHz)
        dummy_audio = np.zeros(16000)  # 1 second of silence
        
        # Create dummy visual data (black image)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test multimodal processing
        multimodal_state = processor.process_multimodal_input(
            text="I'm feeling great today!",
            audio_data=dummy_audio,
            frame=dummy_frame
        )
        
        print(f"Multimodal result:")
        print(f"  Dominant emotion: {multimodal_state.dominant_emotion}")
        print(f"  Emotional intensity: {multimodal_state.emotional_intensity:.3f}")
        print(f"  Confidence: {multimodal_state.confidence:.3f}")
        print(f"  Text emotions: {len(multimodal_state.text_emotions)}")
        print(f"  Audio emotions: {len(multimodal_state.audio_emotions)}")
        print(f"  Visual emotions: {len(multimodal_state.visual_emotions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multimodal ERADEM: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multimodal_eradem()
    if success:
        print("\n✅ Multimodal ERADEM test completed successfully!")
    else:
        print("\n❌ Multimodal ERADEM test failed!") 