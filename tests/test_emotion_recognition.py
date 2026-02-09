import torch
from transformers import AutoTokenizer
from src.train_hierarchical_eradem import HierarchicalEmotionModel
import json
import numpy as np

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    # Load hierarchy info
    with open("data/processed/hierarchy_info.json", "r") as f:
        hierarchy_info = json.load(f)
    
    # Initialize model with correct dimensions
    input_size = 768  # RoBERTa base hidden size
    hidden_size = 256
    num_emotions = len(hierarchy_info['emotion_hierarchy']['basic']['emotion_hierarchy'])
    num_emotion_groups = len(hierarchy_info['emotion_hierarchy'].keys())
    
    model = HierarchicalEmotionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_emotions=num_emotions,
        num_emotion_groups=num_emotion_groups
    )
    
    # Load model weights
    model.load_state_dict(torch.load('hierarchical_emotion_model.pth'))
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    return model, tokenizer, hierarchy_info

def test_emotion_recognition():
    """Test the emotion recognition model with example sentences"""
    model, tokenizer, hierarchy_info = load_model_and_tokenizer()
    
    # Set model to evaluation mode
    model.eval()
    
    # Test sentences with natural emotional context
    test_sentences = [
        # Complex emotional scenarios
        "The room fell silent as I opened the envelope. My hands were trembling, and I had to read the letter three times before the words finally sank in.",
        
        "Walking through my old neighborhood, the familiar scent of jasmine brought back memories of summer evenings on the porch with my grandmother.",
        
        "After months of preparation, I stood there watching as people started filling the auditorium. My presentation notes felt heavy in my pocket.",
        
        "The notification light blinked on my phone. Another message from them, probably. I've been staring at it for the past hour.",
        
        "The sun was setting as I finally reached the summit. Everything looked so small from up here - all those problems that had seemed so overwhelming this morning.",
        
        "The last box is now unpacked, but this new apartment still doesn't feel like home. The echo of my footsteps in empty rooms is deafening.",
    ]
    
    # Process each sentence
    for sentence in test_sentences:
        print("\nAnalyzing:", sentence)
        
        # Tokenize and prepare input
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get model's base embeddings
        with torch.no_grad():
            # Get predictions
            final_probs, group_probs, emotion_logits = model(inputs['input_ids'].float())
            
            # Get emotion names for each level
            emotions = list(hierarchy_info['emotion_hierarchy']['basic']['emotion_hierarchy'].keys())
            groups = list(hierarchy_info['emotion_hierarchy'].keys())
            
            # Print group predictions
            print("\nEmotion Group Predictions:")
            for group_idx, (group, prob) in enumerate(zip(groups, group_probs[0])):
                print(f"  - {group}: {prob:.2f}")
            
            # Print emotion predictions
            print("\nEmotion Predictions:")
            for emotion_idx, (emotion, prob) in enumerate(zip(emotions, final_probs[0])):
                if prob > 0.5:  # Only show predictions above threshold
                    print(f"  - {emotion}: {prob:.2f}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_emotion_recognition() 