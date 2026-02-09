import torch
from transformers import AutoTokenizer, AutoModel
import json
import torch.nn as nn

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load label mapping
    with open("data/processed/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    id2label = label_mapping["id2label"]
    num_emotions = len(id2label)
    
    # Load model
    model = ImprovedEmotionModel(768, 256, num_emotions)
    model.load_state_dict(torch.load("models/best_emotion_model_resume.pt", map_location=device))
    model.eval()
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Test sentence
    text = "i have recieved an offer letter from a company they are paying more than my expectations"
    print(f"\nTesting sentence: '{text}'")
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = torch.sigmoid(logits)
    
    # Get all emotion scores
    all_scores = {id2label[str(idx)]: score.item() for idx, score in enumerate(predictions[0])}
    
    # Sort by confidence
    sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAll emotion scores (sorted by confidence):")
    print("-" * 40)
    for emotion, score in sorted_emotions:
        if score > 0.1:  # Only show emotions with >10% confidence
            print(f"{emotion:15} : {score:.4f}")
    
    print("\nTop 5 emotions:")
    print("-" * 20)
    for i, (emotion, score) in enumerate(sorted_emotions[:5]):
        print(f"{i+1}. {emotion:15} : {score:.4f}")

if __name__ == "__main__":
    main() 