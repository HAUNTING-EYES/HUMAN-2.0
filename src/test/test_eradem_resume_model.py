import torch
from transformers import AutoTokenizer, AutoModel
import json
from typing import Dict, List, Tuple
import torch.nn as nn
import os

# Model definition (must match training)
class ImprovedEmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_emotions):
        super(ImprovedEmotionModel, self).__init__()
        self.hidden_size = hidden_size
        self.transformer = AutoModel.from_pretrained('roberta-base')
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_size // 2, num_emotions)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        features = self.feature_extractor(pooled_output)
        logits = self.classifier(features)
        return logits

def load_label_mapping():
    with open("data/processed/label_mapping.json", "r") as f:
        return json.load(f)

def get_emotion_groups() -> Dict[str, List[str]]:
    return {
        "positive": ["joy", "excitement", "pride", "gratitude", "relief", "love", "optimism"],
        "negative": ["anger", "sadness", "fear", "disappointment", "disgust", "grief", "remorse"],
        "social": ["admiration", "amusement", "approval", "caring", "desire"],
        "cognitive": ["confusion", "curiosity", "realization", "surprise"],
        "complex": ["nervousness", "embarrassment", "annoyance", "disapproval"]
    }

def get_emotion_thresholds() -> Dict[str, float]:
    return {
        "admiration": 0.85,
        "amusement": 0.85,
        "anger": 0.6,
        "joy": 0.5,
        "sadness": 0.5,
        "fear": 0.5,
        "surprise": 0.5,
        "love": 0.5,
        "disappointment": 0.4,
        "anxiety": 0.4,
        "confusion": 0.4,
        "relief": 0.4,
        "pride": 0.4,
        "gratitude": 0.4,
        "realization": 0.3,
        "curiosity": 0.3,
        "caring": 0.3,
        "optimism": 0.3,
        "default": 0.45
    }

def adjust_threshold_by_context(emotion: str, score: float, all_scores: Dict[str, float]) -> bool:
    groups = get_emotion_groups()
    base_thresholds = get_emotion_thresholds()
    threshold = base_thresholds.get(emotion, base_thresholds["default"])
    emotion_group = None
    for group, emotions in groups.items():
        if emotion in emotions:
            emotion_group = group
            break
    if emotion_group:
        for group, emotions in groups.items():
            if group != emotion_group:
                conflicting_scores = [all_scores.get(e, 0) for e in emotions]
                if any(s > threshold for s in conflicting_scores):
                    threshold *= 1.2
    if emotion in ["admiration", "amusement"]:
        negative_scores = [all_scores.get(e, 0) for e in groups["negative"]]
        if any(s > 0.3 for s in negative_scores):
            threshold = 0.95
    return score > threshold

def predict_emotions(text: str, model, tokenizer, id2label: Dict[str, str], device) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = torch.sigmoid(logits)
    all_scores = {id2label[str(idx)]: score.item() for idx, score in enumerate(predictions[0])}
    primary_emotions = []
    secondary_emotions = []
    for emotion, score in all_scores.items():
        if adjust_threshold_by_context(emotion, score, all_scores):
            primary_emotions.append((emotion, score))
        elif score > get_emotion_thresholds().get(emotion, 0.45) * 0.6:
            secondary_emotions.append((emotion, score))
    primary_emotions.sort(key=lambda x: x[1], reverse=True)
    secondary_emotions.sort(key=lambda x: x[1], reverse=True)
    return primary_emotions, secondary_emotions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    label_mapping = load_label_mapping()
    id2label = label_mapping["id2label"]
    num_emotions = len(id2label)
    model = ImprovedEmotionModel(768, 256, num_emotions)
    model.load_state_dict(torch.load("models/best_emotion_model_resume.pt", map_location=device))
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    test_texts = [
        "I am absolutely furious about what they did!",
        "While I'm proud of my achievement, I'm also nervous about the next challenge.",
        "Their constant criticism and mockery made me feel worthless.",
        "The team's support and encouragement helped me overcome my fears.",
        "What started as anger slowly turned into understanding and acceptance.",
        "There's a quiet satisfaction in completing this project.",
        "I'm incredibly excited about the opportunity but also feeling some mild anxiety.",
        "Seeing my friend succeed fills me with genuine happiness and pride."
    ]
    print("\nTesting model with context-aware thresholds:\n")
    for text in test_texts:
        print(f"Text: {text}")
        primary_emotions, secondary_emotions = predict_emotions(text, model, tokenizer, id2label, device)
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