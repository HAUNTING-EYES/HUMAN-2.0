"""
Voice-to-Text Emotion Recognition System
Combines speech-to-text, tone analysis, and semantic emotion detection
"""

import speech_recognition as sr
import numpy as np
import librosa
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
import queue
import time

@dataclass
class EmotionResult:
    """Structured emotion recognition result"""
    primary_emotion: str
    confidence: float
    secondary_emotions: List[Tuple[str, float]]
    text_content: str
    tone_features: Dict[str, float]
    semantic_emotions: Dict[str, float]
    fusion_confidence: float
    timestamp: float

class VoiceEmotionRecognizer:
    """Advanced voice emotion recognition using multimodal fusion"""
    
    def __init__(self, model_path: str = None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize emotion detection models
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Emotion classification model
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Audio processing
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mfcc = 13
        
        # Emotion categories
        self.emotion_categories = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 
            'disgust', 'neutral', 'excitement', 'frustration'
        ]
        
        # Tone feature weights
        self.tone_weights = {
            'pitch_mean': 0.3,
            'pitch_std': 0.2,
            'energy_mean': 0.25,
            'energy_std': 0.15,
            'speech_rate': 0.1
        }
        
        # Semantic emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'fantastic'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'lonely', 'crying'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'terrible', 'awful', 'horrible'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'incredible'],
            'disgust': ['disgusting', 'gross', 'nasty', 'terrible', 'awful'],
            'excitement': ['excited', 'thrilled', 'pumped', 'awesome', 'incredible'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'upset', 'disappointed']
        }
        
        self.logger = logging.getLogger(__name__)
        
    def extract_tone_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract acoustic features from audio"""
        try:
            # Convert to float if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is not empty
            if len(audio_data) == 0:
                return self._get_default_features()
            
            # Extract pitch (fundamental frequency) with better parameters
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                threshold=0.1
            )
            
            # Get dominant pitch values (more robust)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                # Only include realistic pitch values for human speech
                if 80 <= pitch <= 400:  # Realistic human speech range
                    pitch_values.append(pitch)
            
            pitch_values = np.array(pitch_values)
            
            # Extract energy/amplitude
            energy = librosa.feature.rms(
                y=audio_data, 
                hop_length=self.hop_length
            ).flatten()
            
            # Extract additional features
            mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=self.sample_rate
            ).flatten()
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, 
                sr=self.sample_rate
            ).flatten()
            
            # Zero crossing rate (speech rate indicator)
            zcr = librosa.feature.zero_crossing_rate(audio_data).flatten()
            
            # Calculate features with better validation
            features = {
                'pitch_mean': float(np.mean(pitch_values)) if len(pitch_values) > 0 else 150.0,
                'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 20.0,
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'speech_rate': float(len(audio_data) / self.sample_rate),  # Duration
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zcr_mean': float(np.mean(zcr)),
                'mfcc_variance': float(np.var(mfcc[1:])),  # Variance of higher MFCCs
                'voice_activity': float(np.sum(energy > np.mean(energy) * 0.5) / len(energy))
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting tone features: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when extraction fails"""
        return {
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'energy_mean': 0.05,
            'energy_std': 0.02,
            'speech_rate': 3.0,
            'spectral_centroid_mean': 1000.0,
            'spectral_rolloff_mean': 2000.0,
            'zcr_mean': 0.1,
            'mfcc_variance': 1.0,
            'voice_activity': 0.5
        }
    
    def analyze_semantic_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions from text content using keywords and sentiment"""
        try:
            text_lower = text.lower()
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
            
            # Keyword-based analysis
            for emotion, keywords in self.emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_scores[emotion] += 0.3
            
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)[0]
            sentiment_score = sentiment_result['score']
            sentiment_label = sentiment_result['label'].lower()
            
            # Map sentiment to emotions
            if sentiment_label == 'positive':
                emotion_scores['joy'] += sentiment_score * 0.5
                emotion_scores['excitement'] += sentiment_score * 0.3
            elif sentiment_label == 'negative':
                emotion_scores['sadness'] += sentiment_score * 0.4
                emotion_scores['anger'] += sentiment_score * 0.3
                emotion_scores['frustration'] += sentiment_score * 0.3
            
            # Emotion classification
            emotion_results = self.emotion_classifier(text)[0]
            for result in emotion_results:
                emotion = result['label'].lower()
                score = result['score']
                if emotion in emotion_scores:
                    emotion_scores[emotion] += score * 0.4
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Error in semantic emotion analysis: {e}")
            return {emotion: 0.0 for emotion in self.emotion_categories}
    
    def fuse_emotions(self, tone_emotions: Dict[str, float], 
                     semantic_emotions: Dict[str, float]) -> EmotionResult:
        """Fuse tone and semantic emotion analysis"""
        try:
            # Weight the different sources
            tone_weight = 0.4
            semantic_weight = 0.6
            
            # Combine emotions
            fused_emotions = {}
            for emotion in self.emotion_categories:
                tone_score = tone_emotions.get(emotion, 0.0)
                semantic_score = semantic_emotions.get(emotion, 0.0)
                fused_emotions[emotion] = (
                    tone_score * tone_weight + 
                    semantic_score * semantic_weight
                )
            
            # Get primary emotion
            primary_emotion = max(fused_emotions.items(), key=lambda x: x[1])
            
            # Get secondary emotions (top 3)
            sorted_emotions = sorted(fused_emotions.items(), key=lambda x: x[1], reverse=True)
            secondary_emotions = sorted_emotions[1:4]
            
            # Calculate confidence
            confidence = primary_emotion[1]
            
            return EmotionResult(
                primary_emotion=primary_emotion[0],
                confidence=confidence,
                secondary_emotions=secondary_emotions,
                text_content="",  # Will be filled by caller
                tone_features={},  # Will be filled by caller
                semantic_emotions=semantic_emotions,
                fusion_confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error fusing emotions: {e}")
            return EmotionResult(
                primary_emotion="neutral",
                confidence=0.0,
                secondary_emotions=[],
                text_content="",
                tone_features={},
                semantic_emotions={},
                fusion_confidence=0.0,
                timestamp=time.time()
            )
    
    def recognize_emotion_from_audio(self, audio_file_path: str) -> EmotionResult:
        """Recognize emotion from audio file using multimodal fusion"""
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            # Extract tone features
            tone_features = self.extract_tone_features(audio_data)
            
            # Convert speech to text
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
                
            try:
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = ""
                self.logger.warning("Speech not recognized")
            except sr.RequestError as e:
                text = ""
                self.logger.error(f"Speech recognition error: {e}")
            
            # Analyze semantic emotions
            semantic_emotions = self.analyze_semantic_emotions(text) if text else {
                emotion: 0.0 for emotion in self.emotion_categories
            }
            
            # Convert tone features to emotion scores
            tone_emotions = self.tone_features_to_emotions(tone_features)
            
            # Fuse results
            result = self.fuse_emotions(tone_emotions, semantic_emotions)
            result.text_content = text
            result.tone_features = tone_features
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in emotion recognition: {e}")
            return EmotionResult(
                primary_emotion="neutral",
                confidence=0.0,
                secondary_emotions=[],
                text_content="",
                tone_features={},
                semantic_emotions={},
                fusion_confidence=0.0,
                timestamp=time.time()
            )
    
    def tone_features_to_emotions(self, tone_features: Dict[str, float]) -> Dict[str, float]:
        """Convert tone features to emotion scores using research-backed patterns"""
        emotions = {emotion: 0.0 for emotion in self.emotion_categories}
        
        # Get features with defaults
        pitch_mean = tone_features.get('pitch_mean', 150.0)
        pitch_std = tone_features.get('pitch_std', 20.0)
        energy_mean = tone_features.get('energy_mean', 0.05)
        energy_std = tone_features.get('energy_std', 0.02)
        spectral_centroid = tone_features.get('spectral_centroid_mean', 1000.0)
        zcr_mean = tone_features.get('zcr_mean', 0.1)
        voice_activity = tone_features.get('voice_activity', 0.5)
        
        # JOY/EXCITEMENT: High pitch, high energy, high spectral centroid
        if pitch_mean > 180 and energy_mean > 0.08 and spectral_centroid > 1200:
            emotions['joy'] += 0.4
            emotions['excitement'] += 0.3
        
        # SADNESS: Low pitch, low energy, low spectral centroid
        if pitch_mean < 140 and energy_mean < 0.04 and spectral_centroid < 800:
            emotions['sadness'] += 0.5
        
        # ANGER: High pitch variance, high energy variance, high spectral centroid
        if pitch_std > 40 and energy_std > 0.05 and spectral_centroid > 1500:
            emotions['anger'] += 0.4
            emotions['frustration'] += 0.3
        
        # FEAR: High pitch, high energy variance, low voice activity
        if pitch_mean > 200 and energy_std > 0.08 and voice_activity < 0.4:
            emotions['fear'] += 0.4
        
        # SURPRISE: Very high pitch, high energy, high zero crossing rate
        if pitch_mean > 250 and energy_mean > 0.1 and zcr_mean > 0.15:
            emotions['surprise'] += 0.5
        
        # DISGUST: Low pitch, high energy variance, low spectral centroid
        if pitch_mean < 130 and energy_std > 0.06 and spectral_centroid < 700:
            emotions['disgust'] += 0.4
        
        # FRUSTRATION: Medium-high pitch variance, medium energy variance
        if 30 < pitch_std < 60 and 0.03 < energy_std < 0.08:
            emotions['frustration'] += 0.3
        
        # NEUTRAL: Balanced features
        if (140 <= pitch_mean <= 180 and 
            0.04 <= energy_mean <= 0.08 and
            800 <= spectral_centroid <= 1200 and
            pitch_std < 30):
            emotions['neutral'] += 0.4
        
        # Additional patterns based on research
        
        # High pitch + high energy + high activity = excitement
        if pitch_mean > 200 and energy_mean > 0.09 and voice_activity > 0.7:
            emotions['excitement'] += 0.3
        
        # Low pitch + low energy + low activity = sadness
        if pitch_mean < 130 and energy_mean < 0.03 and voice_activity < 0.3:
            emotions['sadness'] += 0.3
        
        # High variance in all features = anger/frustration
        if pitch_std > 50 and energy_std > 0.1 and spectral_centroid > 2000:
            emotions['anger'] += 0.2
            emotions['frustration'] += 0.2
        
        # Normalize scores
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        else:
            # If no patterns detected, default to neutral
            emotions['neutral'] = 1.0
        
        return emotions
    
    def listen_and_recognize(self, duration: int = 5) -> EmotionResult:
        """Listen to microphone and recognize emotion in real-time"""
        try:
            print(f"🎤 Listening for {duration} seconds... (speak now!)")
            
            with self.microphone as source:
                # Adjust for ambient noise
                print("🔇 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio with more lenient parameters
                print("🎵 Listening for speech...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=duration + 2,  # Add buffer time
                    phrase_time_limit=duration,
                    snowboy_configuration=None  # Disable snowboy for better compatibility
                )
            
            print("🔄 Processing speech...")
            
            # Convert to text
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Recognized: '{text}'")
            except sr.UnknownValueError:
                text = ""
                print("❌ Speech not recognized clearly")
            except sr.RequestError as e:
                text = ""
                print(f"❌ Speech recognition service error: {e}")
            
            # Convert audio to numpy array for analysis
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Extract tone features
            print("🎵 Analyzing tone...")
            tone_features = self.extract_tone_features(audio_data)
            
            # Analyze semantic emotions
            print("📖 Analyzing text emotions...")
            semantic_emotions = self.analyze_semantic_emotions(text) if text else {
                emotion: 0.0 for emotion in self.emotion_categories
            }
            
            # Convert tone features to emotions
            tone_emotions = self.tone_features_to_emotions(tone_features)
            
            # Fuse results
            print("🔗 Fusing results...")
            result = self.fuse_emotions(tone_emotions, semantic_emotions)
            result.text_content = text
            result.tone_features = tone_features
            
            return result
            
        except sr.WaitTimeoutError:
            print("⏰ Timeout: No speech detected. Please try again.")
            return EmotionResult(
                primary_emotion="neutral",
                confidence=0.0,
                secondary_emotions=[],
                text_content="",
                tone_features={},
                semantic_emotions={},
                fusion_confidence=0.0,
                timestamp=time.time()
            )
        except Exception as e:
            print(f"❌ Error in real-time recognition: {e}")
            return EmotionResult(
                primary_emotion="neutral",
                confidence=0.0,
                secondary_emotions=[],
                text_content="",
                tone_features={},
                semantic_emotions={},
                fusion_confidence=0.0,
                timestamp=time.time()
            )

class RealTimeVoiceEmotionMonitor:
    """Real-time voice emotion monitoring with continuous listening"""
    
    def __init__(self):
        self.recognizer = VoiceEmotionRecognizer()
        self.is_listening = False
        self.emotion_queue = queue.Queue()
        self.listen_thread = None
        
    def start_monitoring(self):
        """Start continuous emotion monitoring"""
        if not self.is_listening:
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self._monitor_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            print("🎤 Voice emotion monitoring started...")
    
    def stop_monitoring(self):
        """Stop emotion monitoring"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join()
        print("🛑 Voice emotion monitoring stopped.")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.is_listening:
            try:
                result = self.recognizer.listen_and_recognize(duration=3)
                if result.confidence > 0.3:  # Only queue confident results
                    self.emotion_queue.put(result)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def get_latest_emotion(self) -> Optional[EmotionResult]:
        """Get the latest emotion result"""
        try:
            return self.emotion_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_emotion_history(self, max_results: int = 10) -> List[EmotionResult]:
        """Get recent emotion history"""
        history = []
        while len(history) < max_results:
            try:
                result = self.emotion_queue.get_nowait()
                history.append(result)
            except queue.Empty:
                break
        return history

# Example usage and testing
if __name__ == "__main__":
    # Test the voice emotion recognition system
    recognizer = VoiceEmotionRecognizer()
    
    print("🎤 Voice Emotion Recognition System")
    print("=" * 50)
    print("This system analyzes both your words AND your tone!")
    print("Try saying different emotional phrases like:")
    print("  - 'I'm so happy today!' (joy)")
    print("  - 'I'm really frustrated with this' (frustration)")
    print("  - 'I'm scared about the future' (fear)")
    print("  - 'I'm angry about what happened' (anger)")
    print("=" * 50)
    
    while True:
        try:
            # Test real-time recognition
            print("\n🎤 Speak for 5 seconds to test emotion recognition...")
            print("(Press Ctrl+C to exit)")
            
            result = recognizer.listen_and_recognize(duration=5)
            
            print(f"\n📊 RESULTS:")
            print(f"📝 Recognized Text: '{result.text_content}'")
            print(f"😊 Primary Emotion: {result.primary_emotion} (confidence: {result.confidence:.2f})")
            
            if result.secondary_emotions:
                print(f"🎭 Secondary Emotions:")
                for emotion, score in result.secondary_emotions:
                    print(f"   - {emotion}: {score:.2f}")
            
            if result.tone_features:
                print(f"🔊 Tone Analysis:")
                for feature, value in result.tone_features.items():
                    print(f"   - {feature}: {value:.3f}")
            
            if result.semantic_emotions:
                print(f"📖 Text Emotion Analysis:")
                for emotion, score in result.semantic_emotions.items():
                    if score > 0.1:  # Only show significant emotions
                        print(f"   - {emotion}: {score:.2f}")
            
            print(f"🎯 Fusion Confidence: {result.fusion_confidence:.2f}")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Thanks for testing! Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Trying again...")
            continue 
 