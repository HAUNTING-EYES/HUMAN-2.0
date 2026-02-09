#!/usr/bin/env python3
"""
Simplified Multimodal Emotion Recognition Training
Focuses on text and audio processing without visual dependencies
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MELDTextDataset(Dataset):
    """Dataset for MELD text emotion recognition"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Load label mappings
        mappings_path = Path(data_path).parent / "label_mappings.json"
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        self.emotion_map = mappings['emotion_map']
        self.num_emotions = len(self.emotion_map)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Number of emotions: {self.num_emotions}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sample['utterance'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels
        emotion_id = sample['emotion_id']
        sentiment_id = sample['sentiment_id']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'emotion_label': torch.tensor(emotion_id, dtype=torch.long),
            'sentiment_label': torch.tensor(sentiment_id, dtype=torch.long),
            'utterance': sample['utterance'],
            'speaker': sample['speaker']
        }

class TextEmotionModel(nn.Module):
    """Text-based emotion recognition model using BERT"""
    
    def __init__(self, model_name='bert-base-uncased', num_emotions=7, num_sentiments=3):
        super(TextEmotionModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Emotion classifier
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)
        
        # Sentiment classifier
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiments)
        
        # Freeze BERT layers (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions
        emotion_logits = self.emotion_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        return emotion_logits, sentiment_logits

class AudioEmotionModel(nn.Module):
    """Audio-based emotion recognition model using CNN + LSTM"""
    
    def __init__(self, num_emotions=7, num_sentiments=3, input_size=128, hidden_size=256):
        super(AudioEmotionModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        
        # LSTM layer
        self.lstm = nn.LSTM(256, hidden_size, batch_first=True, bidirectional=True)
        
        # Classifiers
        self.emotion_classifier = nn.Linear(hidden_size * 2, num_emotions)
        self.sentiment_classifier = nn.Linear(hidden_size * 2, num_sentiments)
    
    def forward(self, x):
        # CNN feature extraction
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)
        
        # LSTM processing
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.squeeze(1)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Get predictions
        emotion_logits = self.emotion_classifier(lstm_out)
        sentiment_logits = self.sentiment_classifier(lstm_out)
        
        return emotion_logits, sentiment_logits

class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model combining text and audio features"""
    
    def __init__(self, text_model, audio_model, num_emotions=7, num_sentiments=3, fusion_dim=512):
        super(MultimodalFusionModel, self).__init__()
        
        self.text_model = text_model
        self.audio_model = audio_model
        
        # Fusion layers
        self.fusion_layer = nn.Linear(768 + 512, fusion_dim)  # 768 from BERT, 512 from audio
        self.dropout = nn.Dropout(0.3)
        
        # Classifiers
        self.emotion_classifier = nn.Linear(fusion_dim, num_emotions)
        self.sentiment_classifier = nn.Linear(fusion_dim, num_sentiments)
    
    def forward(self, text_inputs, audio_inputs):
        # Get text features
        text_emotion_logits, text_sentiment_logits = self.text_model(**text_inputs)
        text_features = self.text_model.bert(**text_inputs).pooler_output
        
        # Get audio features
        audio_emotion_logits, audio_sentiment_logits = self.audio_model(audio_inputs)
        
        # Simple feature fusion (concatenation)
        fused_features = torch.cat([text_features, audio_emotion_logits], dim=1)
        fused_features = torch.relu(self.fusion_layer(fused_features))
        fused_features = self.dropout(fused_features)
        
        # Final predictions
        emotion_logits = self.emotion_classifier(fused_features)
        sentiment_logits = self.sentiment_classifier(fused_features)
        
        return emotion_logits, sentiment_logits

class Trainer:
    """Training class for text emotion recognition"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project="human2-multimodal-emotion",
                config=config,
                name=f"text-emotion-{config['text_model_name']}"
            )
        
        # Load data
        self.load_data()
        
        # Initialize models
        self.initialize_models()
        
        # Setup training
        self.setup_training()
    
    def load_data(self):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['text_model_name'])
        
        # Create datasets
        data_dir = Path(self.config['data_dir'])
        
        self.train_dataset = MELDTextDataset(
            data_dir / "train_split.json",
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        self.val_dataset = MELDTextDataset(
            data_dir / "val_split.json", 
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        self.test_dataset = MELDTextDataset(
            data_dir / "test_split.json",
            self.tokenizer, 
            max_length=self.config['max_length']
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def initialize_models(self):
        """Initialize the models"""
        logger.info("Initializing text emotion model...")
        
        self.model = TextEmotionModel(
            model_name=self.config['text_model_name'],
            num_emotions=self.train_dataset.num_emotions,
            num_sentiments=3
        ).to(self.device)
        
        logger.info("Text emotion model initialized")
    
    def setup_training(self):
        """Setup training components"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['scheduler_step'],
            gamma=self.config['scheduler_gamma']
        )
        
        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_emotions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)
            sentiment_labels = batch['sentiment_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            emotion_logits, sentiment_logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            emotion_loss = self.criterion(emotion_logits, emotion_labels)
            sentiment_loss = self.criterion(sentiment_logits, sentiment_labels)
            total_batch_loss = emotion_loss + sentiment_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            
            # Calculate accuracy
            emotion_preds = torch.argmax(emotion_logits, dim=1)
            correct_emotions += (emotion_preds == emotion_labels).sum().item()
            total_samples += emotion_labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_batch_loss.item():.4f}",
                'acc': f"{correct_emotions/total_samples:.4f}"
            })
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_emotions / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, loader, split_name="Validation"):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_emotions = 0
        total_samples = 0
        
        all_emotion_preds = []
        all_emotion_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=split_name):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_label'].to(self.device)
                sentiment_labels = batch['sentiment_label'].to(self.device)
                
                # Forward pass
                emotion_logits, sentiment_logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                emotion_loss = self.criterion(emotion_logits, emotion_labels)
                sentiment_loss = self.criterion(sentiment_logits, sentiment_labels)
                total_batch_loss = emotion_loss + sentiment_loss
                
                # Update metrics
                total_loss += total_batch_loss.item()
                
                # Calculate accuracy
                emotion_preds = torch.argmax(emotion_logits, dim=1)
                correct_emotions += (emotion_preds == emotion_labels).sum().item()
                total_samples += emotion_labels.size(0)
                
                # Store predictions for detailed analysis
                all_emotion_preds.extend(emotion_preds.cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = correct_emotions / total_samples
        
        return avg_loss, accuracy, all_emotion_preds, all_emotion_labels
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(self.val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if self.config.get('use_wandb', True):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_text_model.pth')
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        logger.info("Training completed!")
    
    def test(self):
        """Test the model"""
        logger.info("Testing model...")
        
        # Load best model
        self.load_model('best_text_model.pth')
        
        # Test
        test_loss, test_acc, test_preds, test_labels = self.validate(self.test_loader, "Testing")
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        # Generate classification report
        emotion_names = list(self.train_dataset.emotion_map.keys())
        report = classification_report(test_labels, test_preds, target_names=emotion_names)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save results
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': report,
            'predictions': [int(p) for p in test_preds],
            'labels': [int(l) for l in test_labels]
        }
        
        with open('text_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Test results saved to text_test_results.json")
    
    def save_model(self, filename):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }, filename)
    
    def load_model(self, filename):
        """Load model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']

def main():
    """Main function"""
    # Configuration
    config = {
        'text_model_name': 'bert-base-uncased',
        'data_dir': 'data/processed/meld',
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'scheduler_step': 3,
        'scheduler_gamma': 0.5,
        'use_wandb': True
    }
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()
    
    # Test
    trainer.test()

if __name__ == "__main__":
    main() 