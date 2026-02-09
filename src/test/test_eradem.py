import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class ERADEM:
    def __init__(self, model_path: str = "models/final_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Initialize model with config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load emotion labels
        with open("data/emotion_labels.json", "r") as f:
            self.emotion_labels = json.load(f)
            
    def predict_emotions(self, text: str, threshold: float = 0.65) -> List[Tuple[str, float]]:
        """Predict emotions in text with confidence scores"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
            
        # Get emotions with confidence > threshold
        probs = probabilities[0].cpu().numpy()
        emotion_scores = [(label, float(prob)) for label, prob in zip(self.emotion_labels, probs) if prob > threshold]
        
        # Sort by confidence
        return sorted(emotion_scores, key=lambda x: x[1], reverse=True)

def main():
    # Initialize ERADEM
    print("Initializing ERADEM...")
    eradem = ERADEM()
    print("ERADEM initialized successfully!")
    
    # Test cases covering various emotional scenarios
    test_inputs = [
        # Joy and Achievement
        "I just got accepted into my dream university! I can't believe it, I'm so excited but also nervous!",
        
        # Anger and Disappointment
        "That movie was absolutely terrible. I want my money back.",
        
        # Fear and Anxiety
        "I'm really worried about the upcoming exam. I haven't studied enough.",
        
        # Gratitude
        "Thank you so much for your help! I really appreciate it.",
        
        # Anger and Disapproval
        "I can't believe they would do something like that. It's completely unacceptable!",
        
        # Sarcasm and Disappointment
        "Oh great, another meeting that could have been an email. Just what I needed.",
        
        # Mixed Emotions (Pride and Nervousness)
        "I'm presenting my research at the conference tomorrow. I'm proud of my work but terrified of public speaking.",
        
        # Subtle Emotions
        "The way she smiled at me made my heart skip a beat. I think I might be falling in love.",
        
        # Complex Emotional State
        "After years of hard work, I finally got the promotion. But now I'm worried about living up to expectations.",
        
        # Cultural Reference
        "That plot twist was mind-blowing! I never saw it coming.",
        
        # Professional Context
        "The quarterly results exceeded our expectations. The team's dedication has really paid off.",
        
        # Personal Growth
        "Looking back at my journey, I realize how much I've grown. The challenges made me stronger.",
        
        # Social Interaction
        "I feel so awkward at these networking events. Everyone seems to know each other except me.",
        
        # Future Uncertainty
        "I'm not sure if I should take this job offer. It's a great opportunity but means moving to a new city.",
        
        # Past Regret
        "I wish I had spent more time with my grandparents when they were still here."
    ]
    
    print("\nTesting ERADEM Emotion Recognition (Confidence Threshold: 65%):")
    print("=" * 70)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {text}")
        emotions = eradem.predict_emotions(text)
        print("Detected Emotions:")
        if emotions:
            for emotion, confidence in emotions:
                print(f"- {emotion.upper()}: {confidence:.1%}")
        else:
            print("No strong emotions detected (confidence > 65%)")
        print("-" * 70)

if __name__ == "__main__":
    main() 