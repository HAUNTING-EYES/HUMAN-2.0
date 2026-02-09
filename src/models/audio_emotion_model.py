#!/usr/bin/env python3
"""
Audio Emotion Recognition Model
==============================

This module implements audio emotion recognition models for the MELD dataset.
Uses CNN+LSTM architecture to process mel spectrograms and classify emotions.

Models:
- AudioEmotionModel: CNN+LSTM architecture
- AudioTransformerModel: Transformer-based architecture
- AudioAttentionModel: CNN+Attention architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from tqdm import tqdm
import os
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """Configuration for audio emotion models"""
    num_emotions: int = 7
    input_channels: int = 1
    n_mels: int = 128
    hidden_size: int = 256
    dropout: float = 0.5


@dataclass
class LSTMModelConfig(ModelConfig):
    """Configuration for LSTM-based model"""
    num_lstm_layers: int = 2


@dataclass
class TransformerModelConfig(ModelConfig):
    """Configuration for Transformer-based model"""
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1


@dataclass
class AttentionModelConfig(ModelConfig):
    """Configuration for Attention-based model"""
    num_heads: int = 8


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    num_epochs: int = 50
    learning_rate: float = 0.001
    device: str = 'cuda'
    model_name: str = 'audio_emotion_model'
    weight_decay: float = 0.01
    patience: int = 5
    lr_factor: float = 0.5


class CNNFeatureExtractor(nn.Module):
    """Reusable CNN feature extraction module"""
    
    def __init__(self, input_channels: int, dropout: float = 0.5):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Extract features from input"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        return x


class WeightInitializer:
    """Utility class for weight initialization"""
    
    @staticmethod
    def initialize_weights(module):
        """Initialize model weights"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AudioEmotionModel(nn.Module):
    """
    CNN+LSTM Audio Emotion Recognition Model
    
    Architecture:
    - CNN layers for feature extraction from mel spectrograms
    - LSTM layers for temporal modeling
    - Fully connected layers for classification
    """
    
    def __init__(self, config: LSTMModelConfig = None):
        super(AudioEmotionModel, self).__init__()
        
        if config is None:
            config = LSTMModelConfig()
        
        self.config = config
        self.num_emotions = config.num_emotions
        self.hidden_size = config.hidden_size
        self.num_lstm_layers = config.num_lstm_layers
        
        self.feature_extractor = CNNFeatureExtractor(config.input_channels, config.dropout)
        
        cnn_output_size = 256 * (config.n_mels // 16) * 8
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = self._build_classifier(config.hidden_size * 2, config.num_emotions, config.dropout)
        
        WeightInitializer.initialize_weights(self)
    
    def _build_classifier(self, input_size: int, num_classes: int, dropout: float):
        """Build classification head"""
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram (batch, channels, n_mels, time_steps)
        
        Returns:
            logits: Emotion classification logits (batch, num_emotions)
        """
        batch_size = x.size(0)
        
        x = self.feature_extractor(x)
        
        x = x.view(batch_size, x.size(1), -1).transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        x = torch.mean(lstm_out, dim=1)
        
        logits = self.classifier(x)
        
        return logits


class AudioTransformerModel(nn.Module):
    """
    Transformer-based Audio Emotion Recognition Model
    
    Architecture:
    - CNN feature extractor
    - Transformer encoder layers
    - Classification head
    """
    
    def __init__(self, config: TransformerModelConfig = None):
        super(AudioTransformerModel, self).__init__()
        
        if config is None:
            config = TransformerModelConfig()
        
        self.config = config
        self.num_emotions = config.num_emotions
        self.d_model = config.d_model
        
        self.feature_extractor = self._build_feature_extractor(config)
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, config.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_emotions)
        )
        
        WeightInitializer.initialize_weights(self)
    
    def _build_feature_extractor(self, config: TransformerModelConfig):
        """Build CNN feature extractor"""
        return nn.Sequential(
            nn.Conv2d(config.input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(config.dropout),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(config.dropout),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(config.dropout),
            
            nn.AdaptiveAvgPool2d((1, config.d_model))
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram (batch, channels, n_mels, time_steps)
        
        Returns:
            logits: Emotion classification logits (batch, num_emotions)
        """
        features = self.feature_extractor(x)
        features = features.squeeze(1)
        
        seq_len = features.size(1)
        pos_encoding = self.pos_encoder[:, :seq_len, :]
        features = features + pos_encoding
        
        transformer_out = self.transformer(features)
        
        pooled = torch.mean(transformer_out, dim=1)
        
        logits = self.classifier(pooled)
        
        return logits


class AudioAttentionModel(nn.Module):
    """
    CNN+Attention Audio Emotion Recognition Model
    
    Architecture:
    - CNN layers for feature extraction
    - Self-attention mechanism
    - Classification head
    """
    
    def __init__(self, config: AttentionModelConfig = None):
        super(AudioAttentionModel, self).__init__()
        
        if config is None:
            config = AttentionModelConfig()
        
        self.config = config
        self.num_emotions = config.num_emotions
        self.hidden_size = config.hidden_size
        
        self.conv1 = nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(config.dropout)
        
        cnn_output_size = 128 * (config.n_mels // 8) * 16
        
        self.projection = nn.Linear(cnn_output_size, config.hidden_size)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_emotions)
        )
        
        WeightInitializer.initialize_weights(self)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input mel spectrogram (batch, channels, n_mels, time_steps)
        
        Returns:
            logits: Emotion classification logits (batch, num_emotions)
        """
        batch_size = x.size(0)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = x.view(batch_size, x.size(1), -1).transpose(1, 2)
        x = self.projection(x)
        
        attended, _ = self.attention(x, x, x)
        attended = self.layer_norm(attended + x)
        
        x = torch.mean(attended, dim=1)
        
        logits = self.classifier(x)
        
        return logits


class ModelTrainer:
    """Handles model training logic"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.model = self.model.to(config.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=config.patience, 
            factor=config.lr_factor
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]")
        for batch in train_pbar:
            audio = batch['audio'].to(self.config.device)
            emotions = batch['emotion'].to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            loss = self.criterion(outputs, emotions)