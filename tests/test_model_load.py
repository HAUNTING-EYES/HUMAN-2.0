import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from models.emotion_recognition import EmotionRecognitionSystem
import json

def test_model_loading():
    print("Testing model loading...")
    try:
        # Load the checkpoint first to inspect its structure
        checkpoint = torch.load("models/hierarchical_eradem_final/model.pt")
        print("\nCheckpoint structure:")
        if isinstance(checkpoint, dict):
            print("Keys:", checkpoint.keys())
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"\n{key} keys:", checkpoint[key].keys())
        else:
            print("Checkpoint is not a dictionary")
            print("Type:", type(checkpoint))
            
        # Initialize the emotion recognition system
        model = EmotionRecognitionSystem(
            model_name="roberta-base",
            num_labels=28,
            learning_rate=2e-5,
            weight_decay=0.01,
            emotion_threshold=0.3
        )
        
        # Try loading the checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("\nModel loaded successfully!")
        
        # Test basic inference
        test_text = "I am feeling happy and excited today!"
        emotions = model.predict_emotions(test_text)
        print("\nTest inference successful!")
        print(f"Input text: {test_text}")
        print("Predicted emotions:")
        for emotion, score in emotions.items():
            print(f"- {emotion}: {score:.3f}")
            
    except Exception as e:
        print(f"Error during model loading/inference: {str(e)}")

if __name__ == "__main__":
    test_model_loading() 