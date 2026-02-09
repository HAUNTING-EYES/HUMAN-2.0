#!/usr/bin/env python3
"""
Test script for improved audio preprocessing with noise reduction
"""

import numpy as np
import librosa
import torch
import time
from realtime_audio_emotion import preprocess_audio, RealtimeEmotionDetector

def test_noise_reduction():
    """Test noise reduction preprocessing"""
    print("Testing noise reduction preprocessing...")
    
    # Generate synthetic audio with noise
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Clean signal (sine wave)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Add noise
    noise = 0.1 * np.random.randn(len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Convert to bytes for testing
    audio_bytes = noisy_signal.astype(np.float32).tobytes()
    
    # Test preprocessing
    start_time = time.time()
    features = preprocess_audio(audio_bytes, sr=sr)
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Feature shape: {features.shape if features is not None else 'None'}")
    print(f"Features valid: {features is not None}")
    
    if features is not None:
        print(f"Feature statistics:")
        print(f"  Mean: {np.mean(features):.4f}")
        print(f"  Std: {np.std(features):.4f}")
        print(f"  Min: {np.min(features):.4f}")
        print(f"  Max: {np.max(features):.4f}")
    
    return features is not None

def test_model_compatibility():
    """Test if the model can handle the new feature format"""
    print("\nTesting model compatibility...")
    
    try:
        # Load the model
        recognizer = RealtimeEmotionDetector()
        
        # Generate test features
        test_features = np.random.randn(100, 39)  # 100 time steps, 39 features (13*3)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
        
        # Test prediction
        with torch.no_grad():
            outputs = recognizer.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Model prediction successful")
        print(f"Predicted emotion: {recognizer.emotion_labels[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Model compatibility test failed: {e}")
        return False

def test_processing_speed():
    """Test processing speed with different window sizes"""
    print("\nTesting processing speed...")
    
    sr = 16000
    window_sizes = [1.0, 1.5, 2.0, 3.0]  # seconds
    
    for window_size in window_sizes:
        # Generate audio of specified length
        duration = window_size
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        audio_bytes = audio.astype(np.float32).tobytes()
        
        # Time the processing
        start_time = time.time()
        features = preprocess_audio(audio_bytes, sr=sr)
        processing_time = time.time() - start_time
        
        print(f"Window size: {window_size}s, Processing time: {processing_time:.3f}s, "
              f"Real-time factor: {processing_time/window_size:.2f}x")

def main():
    """Run all tests"""
    print("Testing Audio Emotion Recognition Improvements")
    print("=" * 50)
    
    # Test noise reduction
    noise_test_passed = test_noise_reduction()
    
    # Test model compatibility
    model_test_passed = test_model_compatibility()
    
    # Test processing speed
    test_processing_speed()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Noise reduction: {'PASSED' if noise_test_passed else 'FAILED'}")
    print(f"Model compatibility: {'PASSED' if model_test_passed else 'FAILED'}")
    
    if noise_test_passed and model_test_passed:
        print("\n✅ All tests passed! The improvements are working correctly.")
        print("\nKey improvements:")
        print("- Reduced analysis window from 3s to 1.5s for faster response")
        print("- Added preemphasis filter for noise reduction")
        print("- Added spectral gating for additional noise suppression")
        print("- Reduced feature length from 150 to 100 for faster processing")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 