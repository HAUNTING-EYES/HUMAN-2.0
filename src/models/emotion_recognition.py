import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from datasets import load_dataset
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import wandb
from huggingface_hub import HfApi, Repository
import shutil
import os
import torchmetrics

class EmotionDataset(Dataset):
    """Dataset class for emotion recognition"""
    
    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("roberta-base")
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            
        return item

class EmotionRecognitionSystem(pl.LightningModule):
    """Main class for emotion recognition system"""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 28,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        emotion_threshold: float = 0.3,
        use_wandb: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Define emotion labels
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise'
        ]
        
        # Load model and tokenizer
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(self.emotion_labels)},
            label2id={label: str(i) for i, label in enumerate(self.emotion_labels)}
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)
        
        # Additional metrics for validation
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels)
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_labels)
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log("train_loss", loss)
        preds = torch.sigmoid(logits)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.sigmoid(logits)
        
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log("test_loss", loss)
        preds = torch.sigmoid(logits)
        self.test_acc(preds, labels)
        self.log("test_acc", self.test_acc)
        
        return loss
        
    def configure_optimizers(self):
        # Implement differential learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=1e-8
        )
        
        return optimizer
        
    def predict_emotions(self, text: Union[str, List[str]]) -> Dict[str, float]:
        """Predict emotions for given text"""
        self.eval()
        with torch.no_grad():
            # Prepare input
            if isinstance(text, str):
                text = [text]
                
            # Tokenize
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            logits = self(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(logits)
            
            # Convert to emotions dictionary
            emotions = {}
            for idx, prob in enumerate(probs[0]):
                if prob > self.hparams.emotion_threshold:
                    emotions[self.emotion_labels[idx]] = float(prob)
                    
            return emotions
            
    def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding"""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.transformer(**inputs)
            return outputs[1].cpu().numpy()  # Return pooled output
            
    def save_to_hub(self, hub_token: str, hub_model_id: str):
        """Save model to Hugging Face Hub"""
        api = HfApi()
        api.set_access_token(hub_token)
        
        # Create repo if it doesn't exist
        try:
            repo_url = api.create_repo(hub_model_id, private=True)
        except Exception:
            repo_url = f"https://huggingface.co/{hub_model_id}"
            
        # Clone repo
        repo = Repository(
            local_dir=f"models/{hub_model_id}",
            clone_from=repo_url,
            use_auth_token=hub_token
        )
        
        # Save model
        self.transformer.save_pretrained(f"models/{hub_model_id}")
        self.tokenizer.save_pretrained(f"models/{hub_model_id}")
        
        # Save config
        config = {
            "model_name": self.hparams.model_name,
            "num_labels": self.hparams.num_labels,
            "emotion_labels": self.emotion_labels,
            "emotion_threshold": self.hparams.emotion_threshold
        }
        with open(f"models/{hub_model_id}/config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        # Push to hub
        repo.push_to_hub()

    def save_pretrained(self, save_path: Union[str, Path]):
        """Save the model to a directory"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformer model
        self.transformer.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save classifier
        torch.save(self.classifier.state_dict(), save_path / "classifier.pt")
        
        # Save config
        config = {
            "model_name": self.hparams.model_name,
            "num_labels": self.hparams.num_labels,
            "emotion_labels": self.emotion_labels,
            "emotion_threshold": self.hparams.emotion_threshold,
            "learning_rate": self.hparams.learning_rate,
            "weight_decay": self.hparams.weight_decay
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]):
        """Load a pretrained model"""
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            
        # Create model instance
        model = cls(
            model_name=str(model_path),  # Load from local path
            num_labels=config["num_labels"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            emotion_threshold=config["emotion_threshold"]
        )
        
        # Load classifier
        classifier_path = model_path / "classifier.pt"
        if classifier_path.exists():
            model.classifier.load_state_dict(torch.load(classifier_path))
            
        return model 