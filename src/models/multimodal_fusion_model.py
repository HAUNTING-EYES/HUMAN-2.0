#!/usr/bin/env python3
"""
Multimodal Fusion Model for Emotion Recognition
===============================================

This module implements multimodal fusion models that combine audio, visual, and text
emotion recognition for the MELD dataset.

Models:
- MultimodalFusionModel: Attention-based fusion
- MultimodalTransformerModel: Transformer-based fusion
- MultimodalLateFusionModel: Late fusion approach
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

class MultimodalFusionModel(nn.Module):
    """
    Attention-based Multimodal Fusion Model
    
    Architecture:
    - Individual modality models (audio, visual, text)
    - Attention mechanism for modality fusion
    - Final classification head
    """
    
    def __init__(self, num_emotions=7, audio_model=None, visual_model=None, text_model=None, 
                 fusion_dim=256, dropout=0.5):
        super(MultimodalFusionModel, self).__init__()
        
        self.num_emotions = num_emotions
        self.fusion_dim = fusion_dim
        
        # Individual modality models
        self.audio_model = audio_model
        self.visual_model = visual_model
        self.text_model = text_model
        
        # Modality-specific projections
        self.audio_projection = nn.Linear(num_emotions, fusion_dim)
        self.visual_projection = nn.Linear(num_emotions, fusion_dim)
        self.text_projection = nn.Linear(num_emotions, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Modality attention weights
        self.modality_attention = nn.Parameter(torch.ones(3))
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_emotions)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, audio_input=None, visual_input=None, text_input=None):
        """
        Forward pass
        
        Args:
            audio_input: Audio features (batch, audio_features)
            visual_input: Visual features (batch, visual_features)
            text_input: Text features (batch, text_features)
        
        Returns:
            logits: Emotion classification logits (batch, num_emotions)
        """
        batch_size = audio_input.size(0) if audio_input is not None else visual_input.size(0)
        
        # Get predictions from individual modalities
        audio_pred = torch.zeros(batch_size, self.num_emotions, device=audio_input.device if audio_input is not None else visual_input.device)
        visual_pred = torch.zeros(batch_size, self.num_emotions, device=visual_input.device if visual_input is not None else audio_input.device)
        text_pred = torch.zeros(batch_size, self.num_emotions, device=text_input.device if text_input is not None else audio_input.device)
        
        if self.audio_model is not None and audio_input is not None:
            audio_pred = self.audio_model(audio_input)
        
        if self.visual_model is not None and visual_input is not None:
            visual_pred = self.visual_model(visual_input)
        
        if self.text_model is not None and text_input is not None:
            text_pred = self.text_model(text_input)
        
        # Project to fusion dimension
        audio_features = self.audio_projection(audio_pred)
        visual_features = self.visual_projection(visual_pred)
        text_features = self.text_projection(text_pred)
        
        # Stack features for cross-modal attention
        features = torch.stack([audio_features, visual_features, text_features], dim=1)  # (batch, 3, fusion_dim)
        
        # Apply cross-modal attention
        attended, _ = self.cross_attention(features, features, features)
        attended = self.layer_norm(attended + features)  # Residual connection
        
        # Apply modality attention weights
        attention_weights = F.softmax(self.modality_attention, dim=0)
        weighted_features = attended * attention_weights.unsqueeze(0).unsqueeze(-1)
        
        # Flatten and fuse
        fused_features = weighted_features.view(batch_size, -1)
        
        # Final classification
        logits = self.fusion_layer(fused_features)
        
        return logits

class MultimodalTransformerModel(nn.Module):
    """
    Transformer-based Multimodal Fusion Model
    
    Architecture:
    - Individual modality encoders
    - Transformer encoder for fusion
    - Classification head
    """
    
    def __init__(self, num_emotions=7, audio_model=None, visual_model=None, text_model=None,
                 d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(MultimodalTransformerModel, self).__init__()
        
        self.num_emotions = num_emotions
        self.d_model = d_model
        
        # Individual modality models
        self.audio_model = audio_model
        self.visual_model = visual_model
        self.text_model = text_model
        
        # Modality projections
        self.audio_projection = nn.Linear(num_emotions, d_model)
        self.visual_projection = nn.Linear(num_emotions, d_model)
        self.text_projection = nn.Linear(num_emotions, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 3, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, audio_input=None, visual_input=None, text_input=None):
        """
        Forward pass
        
        Args:
            audio_input: Audio features
            visual_input: Visual features
            text_input: Text features
        
        Returns:
            logits: Emotion classification logits
        """
        batch_size = audio_input.size(0) if audio_input is not None else visual_input.size(0)
        device = audio_input.device if audio_input is not None else visual_input.device
        
        # Get predictions from individual modalities
        audio_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        visual_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        text_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        
        if self.audio_model is not None and audio_input is not None:
            audio_pred = self.audio_model(audio_input)
        
        if self.visual_model is not None and visual_input is not None:
            visual_pred = self.visual_model(visual_input)
        
        if self.text_model is not None and text_input is not None:
            text_pred = self.text_model(text_input)
        
        # Project to transformer dimension
        audio_features = self.audio_projection(audio_pred)
        visual_features = self.visual_projection(visual_pred)
        text_features = self.text_projection(text_pred)
        
        # Stack features
        features = torch.stack([audio_features, visual_features, text_features], dim=1)
        
        # Add positional encoding
        features = features + self.pos_encoder
        
        # Transformer processing
        transformer_out = self.transformer(features)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

class MultimodalLateFusionModel(nn.Module):
    """
    Late Fusion Multimodal Model
    
    Architecture:
    - Individual modality models
    - Simple concatenation or weighted averaging
    - Final classification
    """
    
    def __init__(self, num_emotions=7, audio_model=None, visual_model=None, text_model=None,
                 fusion_method='weighted', dropout=0.5):
        super(MultimodalLateFusionModel, self).__init__()
        
        self.num_emotions = num_emotions
        self.fusion_method = fusion_method
        
        # Individual modality models
        self.audio_model = audio_model
        self.visual_model = visual_model
        self.text_model = text_model
        
        if fusion_method == 'weighted':
            # Learnable weights for each modality
            self.audio_weight = nn.Parameter(torch.tensor(1.0))
            self.visual_weight = nn.Parameter(torch.tensor(1.0))
            self.text_weight = nn.Parameter(torch.tensor(1.0))
            
            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_emotions, num_emotions // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_emotions // 2, num_emotions)
            )
        
        elif fusion_method == 'concatenation':
            # Concatenate predictions and classify
            self.classifier = nn.Sequential(
                nn.Linear(num_emotions * 3, num_emotions * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_emotions * 2, num_emotions),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_emotions, num_emotions)
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, audio_input=None, visual_input=None, text_input=None):
        """
        Forward pass
        
        Args:
            audio_input: Audio features
            visual_input: Visual features
            text_input: Text features
        
        Returns:
            logits: Emotion classification logits
        """
        batch_size = audio_input.size(0) if audio_input is not None else visual_input.size(0)
        device = audio_input.device if audio_input is not None else visual_input.device
        
        # Get predictions from individual modalities
        audio_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        visual_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        text_pred = torch.zeros(batch_size, self.num_emotions, device=device)
        
        if self.audio_model is not None and audio_input is not None:
            audio_pred = self.audio_model(audio_input)
        
        if self.visual_model is not None and visual_input is not None:
            visual_pred = self.visual_model(visual_input)
        
        if self.text_model is not None and text_input is not None:
            text_pred = self.text_model(text_input)
        
        if self.fusion_method == 'weighted':
            # Weighted combination
            weights = F.softmax(torch.stack([self.audio_weight, self.visual_weight, self.text_weight]), dim=0)
            fused_pred = (weights[0] * audio_pred + weights[1] * visual_pred + weights[2] * text_pred)
            logits = self.classifier(fused_pred)
        
        elif self.fusion_method == 'concatenation':
            # Concatenate predictions
            fused_pred = torch.cat([audio_pred, visual_pred, text_pred], dim=1)
            logits = self.classifier(fused_pred)
        
        return logits

def train_multimodal_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                          device='cuda', model_name='multimodal_fusion_model'):
    """
    Train multimodal fusion model
    
    Args:
        model: Multimodal fusion model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        model_name: Name for saving the model
    """
    
    # Initialize wandb
    wandb.init(
        project="human2-multimodal-emotion",
        name=model_name,
        config={
            "model_type": model.__class__.__name__,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "device": device
        }
    )
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    
    print(f"Training {model.__class__.__name__} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            emotions = batch['emotion'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(audio_input=audio, visual_input=visual, text_input=text)
            loss = criterion(outputs, emotions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += emotions.size(0)
            train_correct += (predicted == emotions).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100 * train_correct / train_total:.2f}%"
            })
        
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_emotions = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                text = batch['text'].to(device)
                emotions = batch['emotion'].to(device)
                
                outputs = model(audio_input=audio, visual_input=visual, text_input=text)
                loss = criterion(outputs, emotions)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += emotions.size(0)
                val_correct += (predicted == emotions).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_emotions.extend(emotions.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100 * val_correct / val_total:.2f}%"
                })
        
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1]
            }, f'models/{model_name}_best.pth')
            print(f"Saved best model with validation accuracy: {val_accuracy:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'train_accuracy': train_accuracy,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1]
    }, f'models/{model_name}_final.pth')
    
    wandb.finish()
    
    return model

def test_multimodal_model(model, test_loader, device='cuda', model_name='multimodal_fusion_model'):
    """Test multimodal fusion model"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_emotions = []
    
    print("Testing multimodal fusion model...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            emotions = batch['emotion'].to(device)
            
            outputs = model(audio_input=audio, visual_input=visual, text_input=text)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += emotions.size(0)
            test_correct += (predicted == emotions).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_emotions.extend(emotions.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy

if __name__ == "__main__":
    # Example usage
    print("Multimodal Fusion Model")
    print("=" * 40)
    
    # Create model
    model = MultimodalFusionModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_audio = torch.randn(2, 1, 128, 128)
    dummy_visual = torch.randn(2, 3, 224, 224)
    dummy_text = torch.randn(2, 768)
    
    with torch.no_grad():
        output = model(audio_input=dummy_audio, visual_input=dummy_visual, text_input=dummy_text)
    print(f"Audio input shape: {dummy_audio.shape}")
    print(f"Visual input shape: {dummy_visual.shape}")
    print(f"Text input shape: {dummy_text.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}") 