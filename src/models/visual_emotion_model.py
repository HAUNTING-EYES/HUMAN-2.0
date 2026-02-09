#!/usr/bin/env python3
"""
Visual Emotion Recognition Model
CNN + LSTM architecture for video-based emotion recognition
Optimized to match the successful audio emotion recognition system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for Visual Emotion Model"""
    num_classes: int = 7
    max_frames: int = 30
    feature_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    pretrained: bool = True


class VisualEmotionModel(nn.Module):
    """Visual emotion recognition model using CNN + LSTM architecture"""
    
    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig object containing model parameters
        """
        super(VisualEmotionModel, self).__init__()
        
        self.num_classes = config.num_classes
        self.max_frames = config.max_frames
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        self.cnn_backbone = self._build_cnn_backbone(config.pretrained)
        self.lstm = self._build_lstm(config)
        self.frame_attention = self._build_attention(config)
        self.classifier = self._build_classifier(config)
        self.frame_classifier = self._build_frame_classifier(config)
        
        self._initialize_weights()
        
        logger.info(f"Initialized VisualEmotionModel with {config.num_classes} classes, {config.max_frames} max frames")
    
    def _build_cnn_backbone(self, pretrained: bool) -> nn.Module:
        """Build CNN backbone"""
        backbone = models.resnet18(pretrained=pretrained)
        return nn.Sequential(*list(backbone.children())[:-1])
    
    def _build_lstm(self, config: ModelConfig) -> nn.LSTM:
        """Build LSTM module"""
        return nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
    
    def _build_attention(self, config: ModelConfig) -> nn.MultiheadAttention:
        """Build attention module"""
        return nn.MultiheadAttention(
            embed_dim=config.hidden_dim * 2,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
    
    def _build_classifier(self, config: ModelConfig) -> nn.Module:
        """Build main classifier"""
        return nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    
    def _build_frame_classifier(self, config: ModelConfig) -> nn.Module:
        """Build frame-level classifier"""
        return nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract features from frames using CNN"""
        batch_size, num_frames, channels, height, width = frames.shape
        frame_features = []
        
        for i in range(num_frames):
            frame = frames[:, i, :, :, :]
            features = self.cnn_backbone(frame)
            features = features.squeeze(-1).squeeze(-1)
            frame_features.append(features)
        
        return torch.stack(frame_features, dim=1)
    
    def _apply_temporal_modeling(self, frame_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LSTM and attention for temporal modeling"""
        lstm_out, _ = self.lstm(frame_features)
        attended_features, attention_weights = self.frame_attention(lstm_out, lstm_out, lstm_out)
        return attended_features, attention_weights
    
    def _compute_predictions(self, attended_features: torch.Tensor, frame_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute final predictions"""
        pooled_features = torch.mean(attended_features, dim=1)
        
        logits = self.classifier(pooled_features)
        probabilities = F.softmax(logits, dim=1)
        
        frame_logits = self.frame_classifier(frame_features)
        frame_probs = F.softmax(frame_logits, dim=2)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'frame_logits': frame_logits,
            'frame_probabilities': frame_probs,
            'features': pooled_features
        }
    
    def forward(self, frames: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            frames: Input frames tensor (batch_size, max_frames, channels, height, width)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and optional attention weights
        """
        frame_features = self._extract_frame_features(frames)
        attended_features, attention_weights = self._apply_temporal_modeling(frame_features)
        output = self._compute_predictions(attended_features, frame_features)
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def predict_emotion(self, frames: torch.Tensor) -> Tuple[int, float, Dict[str, float]]:
        """
        Predict emotion from video frames
        
        Args:
            frames: Input frames tensor
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(frames)
            probabilities = output['probabilities']
            
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            class_probs = {
                f"class_{i}": probabilities[0, i].item()
                for i in range(self.num_classes)
            }
            
            return predicted_class.item(), confidence.item(), class_probs
    
    def get_attention_weights(self, frames: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization"""
        self.eval()
        with torch.no_grad():
            output = self.forward(frames, return_attention=True)
            return output['attention_weights']


class DeviceManager:
    """Manages device selection with AMD GPU support"""
    
    @staticmethod
    def get_device(device: str) -> torch.device:
        """Get appropriate device with AMD GPU support"""
        if device != 'auto':
            return torch.device(device)
        
        device_obj = DeviceManager._try_directml()
        if device_obj:
            return device_obj
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        
        return torch.device('cpu')
    
    @staticmethod
    def _try_directml() -> Optional[torch.device]:
        """Try to use DirectML for AMD GPU"""
        try:
            import torch_directml
            dml_device = torch_directml.device()
            logger.info(f"Using DirectML device: {dml_device}")
            return dml_device
        except ImportError:
            logger.info("DirectML not available, trying other devices...")
            return None


class FramePreprocessor:
    """Handles frame preprocessing operations"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), max_frames: int = 30):
        self.target_size = target_size
        self.max_frames = max_frames
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create preprocessing transform"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
        """Preprocess video frames for model input"""
        processed_frames = [self._process_single_frame(frame) for frame in frames]
        processed_frames = self._pad_or_truncate(processed_frames)
        
        frames_tensor = torch.stack(processed_frames, dim=0)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor.to(device)
    
    def _process_single_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame"""
        if frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        return self.transform(frame)
    
    def _pad_or_truncate(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pad or truncate frames to max_frames"""
        if len(frames) < self.max_frames:
            last_frame = frames[-1] if frames else torch.zeros(3, *self.target_size)
            frames.extend([last_frame] * (self.max_frames - len(frames)))
        else:
            frames = frames[:self.max_frames]
        return frames


class FaceDetector:
    """Handles face detection in frames"""
    
    def __init__(self):
        import cv2
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_and_crop(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Detect and crop faces from frames"""
        return [self._process_frame(frame) for frame in frames]
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for face detection"""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return self._crop_largest_face(frame, faces)
        return frame
    
    def _crop_largest_face(self, frame: np.ndarray, faces) -> np.ndarray:
        """Crop the largest detected face"""
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        return frame[y:y+h, x:x+w]


class VisualEmotionProcessor:
    """Processor for visual emotion recognition"""
    
    EMOTION_MAP = {
        0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
        4: 'fear', 5: 'surprise', 6: 'disgust'
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize visual emotion processor
        
        Args:
            model_path: Path to trained model
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = DeviceManager.get_device(device)
        self.model = None
        self.emotion_map_reverse = {v: k for k, v in self.EMOTION_MAP.items()}
        self.preprocessor = FramePreprocessor()
        self.face_detector = FaceDetector()
        
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"Initialized VisualEmotionProcessor on device: {self.device}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            config = ModelConfig(
                num_classes=checkpoint.get('num_classes', 7),
                max_frames=checkpoint.get('max_frames', 30),
                feature_dim=checkpoint.get('feature_dim', 512),
                hidden_dim=checkpoint.get('hidden_dim', 256),
                num_layers=checkpoint.get('num_layers', 2),
                dropout=checkpoint.get('dropout', 0.5)
            )
            
            self.model = VisualEmotionModel(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded visual emotion model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def predict_emotion(self, frames: List[np.ndarray], detect_faces: bool = True) -> Dict[str, any]:
        """
        Predict emotion from video frames
        
        Args:
            frames: List of video frames
            detect_faces: Whether to detect and crop faces
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if detect_faces:
            frames = self.face_detector.detect_and_crop(frames)
        
        frames_tensor = self.preprocessor.preprocess(frames, self.device)
        predicted_class, confidence, class_probs = self.model.predict_emotion(frames_tensor)
        emotion_label = self.EMOTION_MAP[predicted_class]
        attention_weights = self.model.get_attention_weights(frames_tensor)