#!/usr/bin/env python3
"""
Improved Real-time Audio Emotion Recognition
- Exact preprocessing matching with training pipeline
- Voice Activity Detection (VAD)
- Silence/noise filtering
- Confidence calibration
- Robust error handling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyaudio
import threading
import time
import json
import pickle
from collections import deque
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox
import librosa
import queue
import logging
import sounddevice as sd
from scipy.signal import butter, filtfilt
import webrtcvad
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

@dataclass
class GUIUpdateData:
    emotion: str
    confidence: float
    tone_info: dict
    is_voice: bool
    energy: float
    spectral: float

class AudioConfig:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.hop_length = 512
        self.n_mels = 128
        self.n_fft = 2048
        self.energy_threshold = 0.01
        self.spectral_threshold = 0.1

class BaseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_leaky_relu=True):
        super(BaseConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1) if use_leaky_relu else nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class BaseFCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.4):
        super(BaseFCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class BalancedAudioCNN(nn.Module):
    def __init__(self, n_mfcc=40, n_frames=105, n_classes=8, dropout_rate=0.5):
        super(BalancedAudioCNN, self).__init__()
        
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.n_classes = n_classes
        
        self.conv_blocks = nn.ModuleList([
            BaseConvBlock(1, 32),
            BaseConvBlock(32, 64),
            BaseConvBlock(64, 128),
            BaseConvBlock(128, 256)
        ])
        
        conv_output_size = (n_frames // 16) * (n_mfcc // 16) * 256
        
        self.fc_blocks = nn.ModuleList([
            BaseFCBlock(conv_output_size, 512, dropout_rate),
            BaseFCBlock(512, 256, dropout_rate),
            BaseFCBlock(256, 128, dropout_rate)
        ])
        
        self.fc_output = nn.Linear(128, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.view(x.size(0), -1)
        
        for fc_block in self.fc_blocks:
            x = fc_block(x)
        
        x = self.fc_output(x)
        return x

class RobustAudioCNN(nn.Module):
    def __init__(self, n_mfcc, n_classes, n_frames=None):
        super(RobustAudioCNN, self).__init__()
        
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        self.n_frames = n_frames
        
        self.conv_blocks = nn.ModuleList([
            BaseConvBlock(1, 32),
            BaseConvBlock(32, 64),
            BaseConvBlock(64, 128),
            BaseConvBlock(128, 256)
        ])
        
        conv_output_size = self._calculate_conv_output_size(n_mfcc, n_frames)
        
        self.fc_blocks = nn.ModuleList([
            BaseFCBlock(conv_output_size, 512),
            BaseFCBlock(512, 256),
            BaseFCBlock(256, 128)
        ])
        
        self.fc_output = nn.Linear(128, n_classes)
    
    def _calculate_conv_output_size(self, n_mfcc, n_frames):
        if n_frames is None:
            return (n_mfcc // 16) * 256
        return (n_frames // 16) * (n_mfcc // 16) * 256
    
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.view(x.size(0), -1)
        
        for fc_block in self.fc_blocks:
            x = fc_block(x)
        
        x = self.fc_output(x)
        return x

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration=0.03, energy_threshold=0.01, spectral_threshold=0.1):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.energy_threshold = energy_threshold
        self.spectral_threshold = spectral_threshold
    
    def _calculate_energy(self, audio_data):
        return np.mean(audio_data**2)
    
    def _calculate_spectral_centroid(self, audio_data):
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate
        ).mean()
        return spectral_centroid / (self.sample_rate / 2)
    
    def _is_voice_detected(self, energy, spectral_centroid_norm):
        return (energy > self.energy_threshold and 
                spectral_centroid_norm > self.spectral_threshold)
    
    def detect_voice_activity(self, audio_data):
        if len(audio_data) < self.frame_size:
            return False, 0.0, 0.0
        
        energy = self._calculate_energy(audio_data)
        spectral_centroid_norm = self._calculate_spectral_centroid(audio_data)
        
        print(f"[VAD] Energy: {energy:.5f}, Spectral: {spectral_centroid_norm:.3f}, "
              f"Thresholds: E>{self.energy_threshold}, S>{self.spectral_threshold}")
        
        has_voice = self._is_voice_detected(energy, spectral_centroid_norm)
        
        return has_voice, energy, spectral_centroid_norm

class ConfidenceCalibrator:
    def __init__(self, temperature=1.0, min_confidence=0.3):
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.confidence_history = deque(maxlen=100)
    
    def _scale_probabilities(self, probabilities):
        scaled_probs = np.power(probabilities, 1/self.temperature)
        return scaled_probs / np.sum(scaled_probs)
    
    def _is_confident(self, max_prob):
        return max_prob >= self.min_confidence
    
    def calibrate_confidence(self, probabilities):
        scaled_probs = self._scale_probabilities(probabilities)
        max_prob = np.max(scaled_probs)
        is_confident = self._is_confident(max_prob)
        
        return scaled_probs, is_confident

class RobustEmotionModel(nn.Module):
    def __init__(self, num_classes=7, input_size=128):
        super(RobustEmotionModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class EmotionStabilizer:
    def __init__(self, buffer_size=30, min_confidence=0.4, persistence_frames=15):
        self.emotion_buffer = []
        self.tone_buffer = []
        self.buffer_size = buffer_size
        self.min_confidence_threshold = min_confidence
        self.emotion_persistence_frames = persistence_frames
        self.last_stable_emotion = "neutral"
        self.last_stable_confidence = 0.0
        self.emotion_stability_counter = 0
    
    def add_emotion(self, emotion, confidence):
        self.emotion_buffer.append((emotion, confidence))
        if len(self.emotion_buffer) > self.buffer_size:
            self.emotion_buffer = self.emotion_buffer[-self.buffer_size:]
    
    def add_tone(self, tone_info):
        self.tone_buffer.append(tone_info)
    
    def _calculate_emotion_stats(self):
        emotion_counts = {}
        
        for emotion, confidence in self.emotion_buffer:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'confidence': 0}
            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['confidence'] += confidence
        
        return emotion_counts
    
    def _get_most_frequent_emotion(self, emotion_counts):
        sorted_emotions = sorted(
            emotion_counts.items(), 
            key=lambda x: (x[1]['count'], x[1]['confidence']/x[1]['count']), 
            reverse=True
        )
        return sorted_emotions[0]
    
    def _should_update_stable_emotion(self, most_frequent_emotion, stats):
        min_samples = max(5, self.buffer_size // 6)
        avg_confidence = stats['confidence'] / stats['count']
        
        if stats['count'] < min_samples or avg_confidence < self.min_confidence_threshold:
            return False
        
        if most_frequent_emotion == self.last_stable_emotion:
            return True
        
        return stats['count'] >= len(self.emotion_buffer) * 0.4
    
    def _update_emotion_state(self, most_frequent_emotion, stats):
        avg_confidence = stats['confidence'] / stats['count']
        
        if most_frequent_emotion != self.last_stable_emotion:
            self.last_stable_emotion = most_frequent_emotion
            self.last_stable_confidence = avg_confidence
            self.emotion_stability_counter = 0
        else:
            self.last_stable_confidence = avg_confidence
            self.emotion_stability_counter = 0
    
    def _reset_to_neutral(self):
        self.last_stable_emotion = "neutral"
        self.last_stable_confidence = 0.0
        self.emotion_stability_counter = 0
    
    def get_stable_emotion(self):
        if not self.emotion_buffer:
            return self.last_stable_emotion, self.last_stable_confidence
        
        emotion_counts = self._calculate_emotion_stats()
        
        if not emotion_counts:
            return self.last_stable_emotion, self.last_stable_confidence
        
        most_frequent_emotion, stats = self._get_most_frequent_emotion(emotion_counts)
        
        if self._should_update_stable_emotion(most_frequent_emotion, stats):
            self._update_emotion_state(most_frequent_emotion, stats)
        else:
            self.emotion_stability_counter += 1
        
        if self.emotion_stability_counter > self.emotion_persistence_frames:
            self._reset_to_neutral()
        
        return self.last_stable_emotion, self.last_stable_confidence

class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.config = None
        self.label_encoder = None
        self.emotion_labels = None
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _prepare_config(self, config):
        n_mfcc = config.get('n_mfcc', 40)
        n_frames = config.get('n_frames', 105)
        config['input_size'] = n_mfcc * n_frames + 15
        return config
    
    def load_fixed_model(self):
        self.config = self._load_config('models/fixed_model_config.json')
        self.config = self._prepare_config(self.config)
        num_classes = self.config['num_classes']