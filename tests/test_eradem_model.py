#!/usr/bin/env python3
"""
Test script for ERADEM emotional model
"""

import torch
import json
import os
from transformers import RobertaTokenizer
from src.models.hierarchical_eradem import HierarchicalERADEM

def test_eradem_model():
    """Test the ERADEM model with sample inputs"""
    
    # Model path
    model_path = "models/hierarchical_eradem_final"
    
    print("Loading ERADEM model...")
    
    try:
        # Load tokenizer (use standard RoBERTa tokenizer)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print(f"✓ Tokenizer loaded successfully")
        
        # Load model
        model = HierarchicalERADEM.from_pretrained(model_path)
        model.eval()
        print(f"✓ Model loaded successfully")
        
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
        
        print("\n" + "="*50)
        print("TESTING ERADEM MODEL")
        print("="*50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Input: '{text}'")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                
                # Get top emotion
                top_prob, top_idx = torch.max(predictions, dim=1)
                emotion_name = model.id2label.get(top_idx.item(), f"emotion_{top_idx.item()}")
                
                print(f"   Predicted emotion: {emotion_name} (confidence: {top_prob.item():.3f})")
                
                # Show top 3 emotions
                top3_probs, top3_indices = torch.topk(predictions, 3, dim=1)
                print("   Top 3 emotions:")
                for j in range(3):
                    emotion = model.id2label.get(top3_indices[0][j].item(), f"emotion_{top3_indices[0][j].item()}")
                    prob = top3_probs[0][j].item()
                    print(f"     - {emotion}: {prob:.3f}")
        
        print("\n" + "="*50)
        print("MODEL INFO")
        print("="*50)
        print(f"Model type: {type(model).__name__}")
        print(f"Number of emotions: {len(model.id2label)}")
        print(f"Available emotions: {list(model.id2label.values())}")
        
        # Check if emotion groups are available
        if hasattr(model, 'emotion_groups') and model.emotion_groups:
            print(f"Emotion groups: {list(model.emotion_groups.keys())}")
            for group, emotions in model.emotion_groups.items():
                print(f"  {group}: {emotions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eradem_model()
    if success:
        print("\n✅ ERADEM model test completed successfully!")
    else:
        print("\n❌ ERADEM model test failed!") 