#!/usr/bin/env python3
"""
Train Robust Audio Emotion Recognition Model
Uses augmented dataset with improved regularization and training techniques
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

# Try to use DirectML if available
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Config
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert to tensor and add channel dimension
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return feature_tensor, label_tensor

class RobustAudioCNN(nn.Module):
    def __init__(self, n_mfcc, n_classes, n_frames=None):
        super(RobustAudioCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        # Calculate the size after convolutions and pooling
        if n_frames is None:
            conv_output_size = (n_mfcc // 16) * 256
        else:
            conv_output_size = (n_frames // 16) * (n_mfcc // 16) * 256
        
        # Fully connected layers with regularization
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)
        self.layer_norm3 = nn.LayerNorm(128)
        
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        self.n_frames = n_frames
    
    def forward(self, x):
        # Convolutional layers with batch normalization and residual connections
        x1 = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x2 = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x1))))
        x3 = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x2))))
        x4 = self.pool(self.leaky_relu(self.batch_norm4(self.conv4(x3))))
        
        # Flatten
        x = x4.view(x4.size(0), -1)
        
        # Fully connected layers with regularization
        x = self.dropout(self.layer_norm1(self.leaky_relu(self.fc1(x))))
        x = self.dropout(self.layer_norm2(self.leaky_relu(self.fc2(x))))
        x = self.dropout(self.layer_norm3(self.leaky_relu(self.fc3(x))))
        x = self.fc4(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True

def load_balanced_dataset():
    """Load the balanced augmented dataset"""
    dataset_path = 'data/audio_features/balanced_augmented_dataset.pkl'
    
    if not os.path.exists(dataset_path):
        print(f"Balanced dataset not found at {dataset_path}")
        print("Please run the augmentation script first: python src/augment_audio_data.py")
        return None, None, None
    
    print(f"Loading balanced dataset from {dataset_path}")
    df = pd.read_pickle(dataset_path)
    
    # Find the maximum dimensions for padding
    max_frames = 0
    max_mfcc = 0
    
    for mfcc in df['mfcc'].values:
        if mfcc is not None:
            max_frames = max(max_frames, mfcc.shape[0])
            max_mfcc = max(max_mfcc, mfcc.shape[1])
    
    print(f"Maximum dimensions: frames={max_frames}, mfcc={max_mfcc}")
    
    # Pad all MFCC features to the same size
    padded_features = []
    for mfcc in df['mfcc'].values:
        if mfcc is not None:
            # Pad or truncate to max dimensions
            if mfcc.shape[0] < max_frames:
                padding = np.zeros((max_frames - mfcc.shape[0], mfcc.shape[1]))
                mfcc = np.concatenate([mfcc, padding], axis=0)
            else:
                mfcc = mfcc[:max_frames, :]
            
            if mfcc.shape[1] < max_mfcc:
                padding = np.zeros((mfcc.shape[0], max_mfcc - mfcc.shape[1]))
                mfcc = np.concatenate([mfcc, padding], axis=1)
            else:
                mfcc = mfcc[:, :max_mfcc]
            
            padded_features.append(mfcc)
        else:
            # Create zero array for None values
            padded_features.append(np.zeros((max_frames, max_mfcc)))
    
    # Convert to numpy array
    mfcc_features = np.array(padded_features)
    labels = df['label'].values
    
    print(f"Dataset shape: {mfcc_features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check class distribution
    emotion_counts = Counter(labels)
    print("Class distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
    
    return mfcc_features, labels, df

def create_data_loaders(features, labels, test_size=0.2, val_size=0.2):
    """Create train/validation/test data loaders"""
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {le.classes_}")
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train_enc)
    val_dataset = AudioDataset(X_val, y_val_enc)
    test_dataset = AudioDataset(X_test, y_test_enc)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, le

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs):
    """Train the model with early stopping and learning rate scheduling"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        if early_stopping.early_stop:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate the model on test set"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_predictions = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*test_correct/test_total:.2f}%'
            })
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * test_correct / test_total
    
    print(f'\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Classification report
    target_names = label_encoder.classes_
    print('\nClassification Report:')
    print(classification_report(all_targets, all_predictions, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_robust.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_accuracy, all_predictions, all_targets

def main():
    """Main training function"""
    print("🎯 Training Robust Audio Emotion Recognition Model")
    print("=" * 60)
    
    # Initialize wandb
    try:
        wandb.init(
            project="audio-emotion-robust",
            config={
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE
            }
        )
        print("✅ WANDB initialized")
    except Exception as e:
        print(f"⚠️ WANDB initialization failed: {e}")
        wandb = None
    
    # Load dataset
    features, labels, df = load_balanced_dataset()
    if features is None:
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, label_encoder = create_data_loaders(features, labels)
    
    # Initialize model
    n_classes = len(label_encoder.classes_)
    model = RobustAudioCNN(
        n_mfcc=features.shape[2],
        n_classes=n_classes,
        n_frames=features.shape[1]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, 
        min_delta=MIN_DELTA
    )
    
    # Train model
    print("\n🚀 Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        early_stopping, EPOCHS
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_accuracy, predictions, targets = evaluate_model(model, test_loader, label_encoder)
    
    # Save model and config
    model_path = 'models/robust_audio_emotion_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save label encoder
    encoder_path = 'models/robust_label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to {encoder_path}")
    
    # Save model config
    config = {
        'n_mfcc': features.shape[2],
        'n_frames': features.shape[1],
        'n_classes': n_classes,
        'test_accuracy': test_accuracy
    }
    config_path = 'models/robust_model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Model config saved to {config_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_curves_robust.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 Training complete!")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main() 