import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from typing import Dict, List, Tuple

def load_label_mapping():
    """Load label mapping"""
    with open("data/processed/label_mapping.json", "r") as f:
        return json.load(f)

def get_emotion_groups() -> Dict[str, List[str]]:
    """Define emotion groups for context-aware detection"""
    return {
        "positive": ["joy", "excitement", "pride", "gratitude", "relief", "love", "optimism"],
        "negative": ["anger", "sadness", "fear", "disappointment", "disgust", "grief", "remorse"],
        "social": ["admiration", "amusement", "approval", "caring", "desire"],
        "cognitive": ["confusion", "curiosity", "realization", "surprise"],
        "complex": ["nervousness", "embarrassment", "annoyance", "disapproval"]
    }

def get_emotion_thresholds() -> Dict[str, float]:
    """Define emotion-specific thresholds"""
    return {
        # Very high threshold for commonly over-predicted emotions
        "admiration": 0.85,
        "amusement": 0.85,
        
        # High threshold for strong emotions
        "anger": 0.6,
        "joy": 0.5,
        "sadness": 0.5,
        "fear": 0.5,
        "surprise": 0.5,
        "love": 0.5,
        
        # Medium threshold for secondary emotions
        "disappointment": 0.4,
        "anxiety": 0.4,
        "confusion": 0.4,
        "relief": 0.4,
        "pride": 0.4,
        "gratitude": 0.4,
        
        # Lower threshold for subtle emotions
        "realization": 0.3,
        "curiosity": 0.3,
        "caring": 0.3,
        "optimism": 0.3,
        
        # Default threshold
        "default": 0.45
    }

def adjust_threshold_by_context(emotion: str, score: float, all_scores: Dict[str, float]) -> bool:
    """Adjust threshold based on emotion context"""
    groups = get_emotion_groups()
    base_thresholds = get_emotion_thresholds()
    threshold = base_thresholds.get(emotion, base_thresholds["default"])
    
    # Find which group this emotion belongs to
    emotion_group = None
    for group, emotions in groups.items():
        if emotion in emotions:
            emotion_group = group
            break
    
    # Context-based adjustments
    if emotion_group:
        # Check for conflicting emotions
        for group, emotions in groups.items():
            if group != emotion_group:
                conflicting_scores = [all_scores.get(e, 0) for e in emotions]
                if any(s > threshold for s in conflicting_scores):
                    # Increase threshold if conflicting emotions are strong
                    threshold *= 1.2
    
    # Special cases
    if emotion in ["admiration", "amusement"]:
        # Check if any negative emotions are present
        negative_scores = [all_scores.get(e, 0) for e in groups["negative"]]
        if any(s > 0.3 for s in negative_scores):
            threshold = 0.95  # Make it very hard to trigger in negative contexts
    
    return score > threshold

def predict_emotions(text: str, model, tokenizer, id2label: Dict[str, str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Predict emotions with context-aware thresholds"""
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
    
    # Create score dictionary
    all_scores = {id2label[str(idx)]: score.item() for idx, score in enumerate(predictions[0])}
    
    # Process emotions with context-aware thresholds
    primary_emotions = []
    secondary_emotions = []
    
    for emotion, score in all_scores.items():
        if adjust_threshold_by_context(emotion, score, all_scores):
            primary_emotions.append((emotion, score))
        elif score > get_emotion_thresholds().get(emotion, 0.45) * 0.6:
            secondary_emotions.append((emotion, score))
    
    # Sort both lists by confidence
    primary_emotions.sort(key=lambda x: x[1], reverse=True)
    secondary_emotions.sort(key=lambda x: x[1], reverse=True)
    
    return primary_emotions, secondary_emotions

def main():
    # Load model and tokenizer
    print("Loading model...")
    model_path = "models/final_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load label mapping
    label_mapping = load_label_mapping()
    id2label = label_mapping["id2label"]
    
    # Test scenarios
    test_texts = [
        # Strong negative emotions
        "I am absolutely furious about what they did!",
        
        # Mixed emotions with context
        "While I'm proud of my achievement, I'm also nervous about the next challenge.",
        
        # Negative context
        "Their constant criticism and mockery made me feel worthless.",
        
        # Positive context
        "The team's support and encouragement helped me overcome my fears.",
        
        # Complex emotional transition
        "What started as anger slowly turned into understanding and acceptance.",
        
        # Subtle positive emotions
        "There's a quiet satisfaction in completing this project.",
        
        # Mixed intensity
        "I'm incredibly excited about the opportunity but also feeling some mild anxiety.",
        
        # Social context
        "Seeing my friend succeed fills me with genuine happiness and pride."
    ]
    
    # Test each text
    print("\nTesting model with context-aware thresholds:\n")
    for text in test_texts:
        print(f"Text: {text}")
        primary_emotions, secondary_emotions = predict_emotions(text, model, tokenizer, id2label)
        
        if primary_emotions:
            print("\nPrimary Emotions:")
            for emotion, score in primary_emotions:
                print(f"- {emotion}: {score:.4f}")
        else:
            print("\nNo primary emotions detected")
            
        if secondary_emotions:
            print("\nSecondary Emotions:")
            for emotion, score in secondary_emotions:
                print(f"- {emotion}: {score:.4f}")
        else:
            print("\nNo secondary emotions detected")
            
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 