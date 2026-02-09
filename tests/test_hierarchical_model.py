import torch
from src.models.hierarchical_eradem import HierarchicalEmotionModel

def test_model():
    print("Loading hierarchical emotion model...")
    model = HierarchicalEmotionModel()
    model.load_checkpoint("models/hierarchical_eradem_final/model.pt")
    model.eval()
    
    # Test sentences
    test_texts = [
        "I am feeling really happy and excited about the future!",
        "I'm so angry about what happened yesterday.",
        "I feel proud of my team's accomplishment.",
        "I'm worried about the upcoming exam."
    ]
    
    print("\nMaking predictions...")
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Tokenize
        inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
        
        # Print predictions for each emotion type
        for emotion_type in ["basic", "social", "cognitive"]:
            probs = torch.sigmoid(outputs[emotion_type]["emotions"])[0]
            threshold = torch.sigmoid(outputs[emotion_type]["threshold"])[0]
            confidence = torch.sigmoid(outputs[emotion_type]["confidence"])[0]
            
            print(f"\n{emotion_type.capitalize()} emotions:")
            print(f"Probabilities: {probs.tolist()}")
            print(f"Threshold: {threshold.item():.3f}")
            print(f"Confidence: {confidence.item():.3f}")

if __name__ == "__main__":
    test_model()
