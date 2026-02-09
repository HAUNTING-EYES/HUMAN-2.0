import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class HierarchicalEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.basic_classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 5)
        )
        self.social_classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 3)
        )
        self.cognitive_classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        basic_logits = self.basic_classifier(pooled_output)
        social_logits = self.social_classifier(pooled_output)
        cognitive_logits = self.cognitive_classifier(pooled_output)
        return {
            "basic": {"emotions": basic_logits},
            "social": {"emotions": social_logits},
            "cognitive": {"emotions": cognitive_logits}
        }

def test_model():
    print("Loading model...")
    model = HierarchicalEmotionModel()
    checkpoint = torch.load("models/hierarchical_eradem_final/model.pt", map_location=torch.device("cpu"))
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"Key: {key}")
        if isinstance(checkpoint[key], dict):
            print("Subkeys:", list(checkpoint[key].keys()))
    model.roberta.load_state_dict(checkpoint["base_model_state_dict"])
    classifier_state_dict = {}
    for key, value in checkpoint["classifiers_state_dict"].items():
        parts = key.split(".")
        classifier_type = parts[0]
        layer_num = parts[1]
        param_type = parts[2]
        if classifier_type == "basic":
            classifier_state_dict[f"basic_classifier.{layer_num}.{param_type}"] = value
        elif classifier_type == "social":
            classifier_state_dict[f"social_classifier.{layer_num}.{param_type}"] = value
        elif classifier_type == "cognitive":
            classifier_state_dict[f"cognitive_classifier.{layer_num}.{param_type}"] = value
    model.load_state_dict(classifier_state_dict, strict=False)
    model.eval()
    hierarchy_info = checkpoint["hierarchy_info"]
    emotion_groups = hierarchy_info["emotion_groups"]
    print("\nEmotion groups in checkpoint:")
    for k, v in emotion_groups.items():
        print(f"{k}: {v}")
    test_texts = [
        "I am feeling really happy and excited about the future!",
        "I'm so angry about what happened yesterday.",
        "I feel proud of my team's accomplishment.",
        "I'm worried about the upcoming exam."
    ]
    print("\nMaking predictions...")
    for text in test_texts:
        print(f"\nText: {text}")
        inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            for emotion_type in ["basic", "social", "cognitive"]:
                probs = torch.sigmoid(outputs[emotion_type]["emotions"])[0]
                print(f"\n{emotion_type.capitalize()} emotions:")
                emotions = emotion_groups[emotion_type]
                for emotion, prob in zip(emotions, probs):
                    print(f"{emotion}: {prob.item():.3f}")

if __name__ == "__main__":
    test_model()