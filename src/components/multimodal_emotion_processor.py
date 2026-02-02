#!/usr/bin/env python3
"""
Multimodal Emotion Processor for HUMAN 2.0
Integrates text, audio, and visual emotion recognition
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import cv2
import librosa
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Structured emotion detection result"""
    emotion: str
    confidence: float
    modality: str  # 'text', 'audio', 'visual', 'multimodal'
    timestamp: float
    metadata: Dict = None

@dataclass
class MultimodalEmotionState:
    """Complete emotional state from all modalities"""
    text_emotions: List[EmotionResult]
    audio_emotions: List[EmotionResult]
    visual_emotions: List[EmotionResult]
    combined_emotions: List[EmotionResult]
    dominant_emotion: str
    emotional_intensity: float
    confidence: float
    timestamp: float

class ImprovedEmotionModel(nn.Module):
    """Text-based emotion model (ERADEM)"""
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

class AudioEmotionProcessor:
    """Audio-based emotion recognition using voice features"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.feature_names = [
            'mfcc_mean', 'mfcc_std', 'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std', 'zero_crossing_rate_mean',
            'zero_crossing_rate_std', 'energy_mean', 'energy_std', 'pitch_mean', 'pitch_std'
        ]
        
        # Simple rule-based audio emotion mapping
        self.audio_emotion_rules = {
            'joy': {'pitch_high': True, 'energy_high': True, 'mfcc_variance': 'high'},
            'sadness': {'pitch_low': True, 'energy_low': True, 'mfcc_variance': 'low'},
            'anger': {'pitch_high': True, 'energy_high': True, 'spectral_centroid_high': True},
            'fear': {'pitch_variable': True, 'energy_low': True, 'zero_crossing_high': True},
            'surprise': {'pitch_high': True, 'energy_high': True, 'spectral_centroid_high': True},
            'disgust': {'pitch_low': True, 'energy_low': True, 'spectral_centroid_low': True},
            'neutral': {'pitch_medium': True, 'energy_medium': True, 'mfcc_variance': 'medium'}
        }
    
    def extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract audio features for emotion recognition"""
        try:
            # Ensure audio is mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if len(audio_data) > 0:
                audio_data = librosa.resample(audio_data, orig_sr=len(audio_data), target_sr=self.sample_rate)
            
            features = {}
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc)
            features['mfcc_std'] = np.std(mfcc)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zero_crossing_rate_mean'] = np.mean(zcr)
            features['zero_crossing_rate_std'] = np.std(zcr)
            
            # Energy
            energy = np.sum(audio_data**2)
            features['energy_mean'] = energy
            features['energy_std'] = np.std(audio_data**2)
            
            # Pitch (using autocorrelation)
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
                pitch_values = pitches[magnitudes > np.percentile(magnitudes, 90)]
                features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
                features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else 0
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def classify_audio_emotion(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Classify emotions based on audio features"""
        try:
            emotion_scores = {}
            
            # Simple rule-based classification
            for emotion, rules in self.audio_emotion_rules.items():
                score = 0.0
                total_rules = len(rules)
                
                for rule, condition in rules.items():
                    if rule == 'pitch_high' and features['pitch_mean'] > 200:
                        score += 1.0
                    elif rule == 'pitch_low' and features['pitch_mean'] < 150:
                        score += 1.0
                    elif rule == 'pitch_medium' and 150 <= features['pitch_mean'] <= 200:
                        score += 1.0
                    elif rule == 'energy_high' and features['energy_mean'] > 0.1:
                        score += 1.0
                    elif rule == 'energy_low' and features['energy_mean'] < 0.05:
                        score += 1.0
                    elif rule == 'energy_medium' and 0.05 <= features['energy_mean'] <= 0.1:
                        score += 1.0
                    elif rule == 'spectral_centroid_high' and features['spectral_centroid_mean'] > 2000:
                        score += 1.0
                    elif rule == 'spectral_centroid_low' and features['spectral_centroid_mean'] < 1000:
                        score += 1.0
                    elif rule == 'zero_crossing_high' and features['zero_crossing_rate_mean'] > 0.1:
                        score += 1.0
                    elif rule == 'mfcc_variance' == 'high' and features['mfcc_std'] > 2.0:
                        score += 1.0
                    elif rule == 'mfcc_variance' == 'low' and features['mfcc_std'] < 1.0:
                        score += 1.0
                    elif rule == 'mfcc_variance' == 'medium' and 1.0 <= features['mfcc_std'] <= 2.0:
                        score += 1.0
                
                emotion_scores[emotion] = score / total_rules
            
            # Sort by confidence
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            return [(emotion, score) for emotion, score in sorted_emotions if score > 0.3]
            
        except Exception as e:
            logger.error(f"Error classifying audio emotion: {e}")
            return [('neutral', 0.5)]

class VisualEmotionProcessor:
    """Visual emotion recognition using DeepFace"""
    
    def __init__(self):
        # Import DeepFace processor
        try:
            from .deepface_visual_emotion import DeepFaceVisualEmotionProcessor
            self.deepface_processor = DeepFaceVisualEmotionProcessor(
                detector_backend='opencv',
                enforce_detection=False  # Robust for various conditions
            )
            self.use_deepface = True
            logger.info("VisualEmotionProcessor initialized with DeepFace backend")
        except ImportError:
            logger.warning("DeepFace not available, falling back to basic processor")
            self.use_deepface = False
            # Fallback to OpenCV face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_facial_features(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract facial features for emotion recognition"""
        try:
            x, y, w, h = face_rect
            face_roi = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            features = {}
            
            # Simple feature extraction (in a real system, you'd use more sophisticated methods)
            # For now, we'll use basic image statistics as proxies for facial expressions
            
            # Brightness (proxy for eye openness)
            features['brightness'] = np.mean(gray_face)
            
            # Contrast (proxy for expression intensity)
            features['contrast'] = np.std(gray_face)
            
            # Edge density (proxy for facial muscle tension)
            edges = cv2.Canny(gray_face, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (w * h)
            
            # Histogram analysis
            hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
            features['histogram_skew'] = np.sum(hist * np.arange(256)) / np.sum(hist)
            
            # Region-based analysis (simplified)
            # Upper face (eyes, brows)
            upper_face = gray_face[:h//2, :]
            features['upper_face_variance'] = np.var(upper_face)
            
            # Lower face (mouth, cheeks)
            lower_face = gray_face[h//2:, :]
            features['lower_face_variance'] = np.var(lower_face)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {e}")
            return {'brightness': 0.5, 'contrast': 0.5, 'edge_density': 0.5, 
                   'histogram_skew': 128, 'upper_face_variance': 0.5, 'lower_face_variance': 0.5}
    
    def classify_visual_emotion(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Classify emotions based on facial features"""
        try:
            emotion_scores = {}
            
            # Rule-based classification using facial features
            for emotion, rules in self.expression_rules.items():
                score = 0.0
                total_rules = len(rules)
                
                for rule, condition in rules.items():
                    if rule == 'smile' and features['lower_face_variance'] > 100:
                        score += 1.0
                    elif rule == 'frown' and features['lower_face_variance'] < 50:
                        score += 1.0
                    elif rule == 'eye_widen' and features['upper_face_variance'] > 80:
                        score += 1.0
                    elif rule == 'eye_squint' and features['upper_face_variance'] < 30:
                        score += 1.0
                    elif rule == 'brow_furrow' and features['edge_density'] > 0.1:
                        score += 1.0
                    elif rule == 'mouth_open' and features['contrast'] > 30:
                        score += 1.0
                    elif rule == 'relaxed' and 20 <= features['contrast'] <= 40:
                        score += 1.0
                
                emotion_scores[emotion] = score / total_rules
            
            # Sort by confidence
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            return [(emotion, score) for emotion, score in sorted_emotions if score > 0.3]
            
        except Exception as e:
            logger.error(f"Error classifying visual emotion: {e}")
            return [('neutral', 0.5)]

class MultimodalEmotionProcessor:
    """Main multimodal emotion processor that combines all modalities"""
    
    def __init__(self, model_path: str = "models/best_emotion_model_resume.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize text processor (ERADEM)
        self.text_processor = self._init_text_processor(model_path)
        
        # Initialize audio processor
        self.audio_processor = AudioEmotionProcessor()
        
        # Initialize visual processor
        self.visual_processor = VisualEmotionProcessor()
        
        # Emotion fusion weights
        self.modality_weights = {
            'text': 0.4,
            'audio': 0.3,
            'visual': 0.3
        }
        
        # Emotion mapping for consistency
        self.emotion_mapping = {
            'joy': ['joy', 'excitement', 'happiness'],
            'sadness': ['sadness', 'grief', 'disappointment'],
            'anger': ['anger', 'annoyance', 'disapproval'],
            'fear': ['fear', 'nervousness', 'anxiety'],
            'surprise': ['surprise', 'realization'],
            'disgust': ['disgust'],
            'neutral': ['neutral']
        }
    
    def _init_text_processor(self, model_path: str):
        """Initialize the text-based ERADEM model"""
        try:
            # Load label mapping
            with open("data/processed/label_mapping.json", "r") as f:
                label_mapping = json.load(f)
            
            id2label = label_mapping["id2label"]
            num_emotions = len(id2label)
            
            # Load model
            model = ImprovedEmotionModel(768, 256, num_emotions)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model.to(self.device)
            
            return {
                'model': model,
                'tokenizer': AutoTokenizer.from_pretrained("roberta-base"),
                'id2label': id2label
            }
            
        except Exception as e:
            logger.error(f"Error initializing text processor: {e}")
            return None
    
    def process_text_emotion(self, text: str) -> List[EmotionResult]:
        """Process text emotion using ERADEM"""
        try:
            if not self.text_processor:
                return [EmotionResult('neutral', 0.5, 'text', time.time())]
            
            model = self.text_processor['model']
            tokenizer = self.text_processor['tokenizer']
            id2label = self.text_processor['id2label']
            
            # Tokenize and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                predictions = torch.sigmoid(logits)
            
            # Get emotion scores
            all_scores = {id2label[str(idx)]: score.item() for idx, score in enumerate(predictions[0])}
            sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top emotions
            results = []
            for emotion, score in sorted_emotions[:3]:  # Top 3 emotions
                if score > 0.1:
                    results.append(EmotionResult(
                        emotion=emotion,
                        confidence=score,
                        modality='text',
                        timestamp=time.time()
                    ))
            
            return results if results else [EmotionResult('neutral', 0.5, 'text', time.time())]
            
        except Exception as e:
            logger.error(f"Error processing text emotion: {e}")
            return [EmotionResult('neutral', 0.5, 'text', time.time())]
    
    def process_audio_emotion(self, audio_data: np.ndarray) -> List[EmotionResult]:
        """Process audio emotion"""
        try:
            features = self.audio_processor.extract_audio_features(audio_data)
            emotions = self.audio_processor.classify_audio_emotion(features)
            
            results = []
            for emotion, score in emotions:
                results.append(EmotionResult(
                    emotion=emotion,
                    confidence=score,
                    modality='audio',
                    timestamp=time.time(),
                    metadata={'features': features}
                ))
            
            return results if results else [EmotionResult('neutral', 0.5, 'audio', time.time())]
            
        except Exception as e:
            logger.error(f"Error processing audio emotion: {e}")
            return [EmotionResult('neutral', 0.5, 'audio', time.time())]
    
    def process_visual_emotion(self, frame: np.ndarray) -> List[EmotionResult]:
        """Process visual emotion from video frame"""
        try:
            faces = self.visual_processor.detect_faces(frame)
            
            if not faces:
                return [EmotionResult('neutral', 0.5, 'visual', time.time())]
            
            # Process the largest face (assumed to be the main subject)
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            features = self.visual_processor.extract_facial_features(frame, largest_face)
            emotions = self.visual_processor.classify_visual_emotion(features)
            
            results = []
            for emotion, score in emotions:
                results.append(EmotionResult(
                    emotion=emotion,
                    confidence=score,
                    modality='visual',
                    timestamp=time.time(),
                    metadata={'features': features, 'face_rect': largest_face}
                ))
            
            return results if results else [EmotionResult('neutral', 0.5, 'visual', time.time())]
            
        except Exception as e:
            logger.error(f"Error processing visual emotion: {e}")
            return [EmotionResult('neutral', 0.5, 'visual', time.time())]
    
    def fuse_emotions(self, text_emotions: List[EmotionResult], 
                     audio_emotions: List[EmotionResult], 
                     visual_emotions: List[EmotionResult]) -> MultimodalEmotionState:
        """Fuse emotions from all modalities"""
        try:
            # Normalize emotions to common categories
            def normalize_emotion(emotion: str) -> str:
                for category, emotions in self.emotion_mapping.items():
                    if emotion.lower() in [e.lower() for e in emotions]:
                        return category
                return emotion.lower()
            
            # Weighted fusion
            emotion_scores = defaultdict(float)
            total_weight = 0
            
            # Process text emotions
            for emotion_result in text_emotions:
                normalized_emotion = normalize_emotion(emotion_result.emotion)
                weight = self.modality_weights['text'] * emotion_result.confidence
                emotion_scores[normalized_emotion] += weight
                total_weight += weight
            
            # Process audio emotions
            for emotion_result in audio_emotions:
                normalized_emotion = normalize_emotion(emotion_result.emotion)
                weight = self.modality_weights['audio'] * emotion_result.confidence
                emotion_scores[normalized_emotion] += weight
                total_weight += weight
            
            # Process visual emotions
            for emotion_result in visual_emotions:
                normalized_emotion = normalize_emotion(emotion_result.emotion)
                weight = self.modality_weights['visual'] * emotion_result.confidence
                emotion_scores[normalized_emotion] += weight
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_weight
            
            # Get dominant emotion
            if emotion_scores:
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_emotion_name, dominant_confidence = dominant_emotion
            else:
                dominant_emotion_name = 'neutral'
                dominant_confidence = 0.5
            
            # Calculate emotional intensity
            emotional_intensity = sum(emotion_scores.values()) / len(emotion_scores) if emotion_scores else 0.5
            
            # Create combined emotions list
            combined_emotions = []
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:
                    combined_emotions.append(EmotionResult(
                        emotion=emotion,
                        confidence=score,
                        modality='multimodal',
                        timestamp=time.time()
                    ))
            
            return MultimodalEmotionState(
                text_emotions=text_emotions,
                audio_emotions=audio_emotions,
                visual_emotions=visual_emotions,
                combined_emotions=combined_emotions,
                dominant_emotion=dominant_emotion_name,
                emotional_intensity=emotional_intensity,
                confidence=dominant_confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error fusing emotions: {e}")
            return MultimodalEmotionState(
                text_emotions=[EmotionResult('neutral', 0.5, 'text', time.time())],
                audio_emotions=[EmotionResult('neutral', 0.5, 'audio', time.time())],
                visual_emotions=[EmotionResult('neutral', 0.5, 'visual', time.time())],
                combined_emotions=[EmotionResult('neutral', 0.5, 'multimodal', time.time())],
                dominant_emotion='neutral',
                emotional_intensity=0.5,
                confidence=0.5,
                timestamp=time.time()
            )
    
    def process_multimodal_input(self, text: str = None, audio_data: np.ndarray = None, 
                                frame: np.ndarray = None) -> MultimodalEmotionState:
        """Process multimodal input and return comprehensive emotional state"""
        try:
            # Process each modality
            text_emotions = self.process_text_emotion(text) if text else []
            audio_emotions = self.process_audio_emotion(audio_data) if audio_data is not None else []
            visual_emotions = self.process_visual_emotion(frame) if frame is not None else []
            
            # If no modalities provided, return neutral state
            if not text_emotions and not audio_emotions and not visual_emotions:
                return MultimodalEmotionState(
                    text_emotions=[EmotionResult('neutral', 0.5, 'text', time.time())],
                    audio_emotions=[EmotionResult('neutral', 0.5, 'audio', time.time())],
                    visual_emotions=[EmotionResult('neutral', 0.5, 'visual', time.time())],
                    combined_emotions=[EmotionResult('neutral', 0.5, 'multimodal', time.time())],
                    dominant_emotion='neutral',
                    emotional_intensity=0.5,
                    confidence=0.5,
                    timestamp=time.time()
                )
            
            # Fuse emotions
            return self.fuse_emotions(text_emotions, audio_emotions, visual_emotions)
            
        except Exception as e:
            logger.error(f"Error processing multimodal input: {e}")
            return MultimodalEmotionState(
                text_emotions=[EmotionResult('neutral', 0.5, 'text', time.time())],
                audio_emotions=[EmotionResult('neutral', 0.5, 'audio', time.time())],
                visual_emotions=[EmotionResult('neutral', 0.5, 'visual', time.time())],
                combined_emotions=[EmotionResult('neutral', 0.5, 'multimodal', time.time())],
                dominant_emotion='neutral',
                emotional_intensity=0.5,
                confidence=0.5,
                timestamp=time.time()
            )

def main():
    """Test the multimodal emotion processor"""
    processor = MultimodalEmotionProcessor()
    
    # Test text-only
    print("Testing text emotion recognition:")
    text_result = processor.process_multimodal_input(
        text="I am absolutely furious about what they did!"
    )
    print(f"Dominant emotion: {text_result.dominant_emotion} (confidence: {text_result.confidence:.3f})")
    print(f"Emotional intensity: {text_result.emotional_intensity:.3f}")
    
    # Test with different text
    print("\nTesting different text:")
    text_result2 = processor.process_multimodal_input(
        text="I'm so excited about the new opportunity!"
    )
    print(f"Dominant emotion: {text_result2.dominant_emotion} (confidence: {text_result2.confidence:.3f})")
    print(f"Emotional intensity: {text_result2.emotional_intensity:.3f}")

if __name__ == "__main__":
    main() 