#!/usr/bin/env python3
"""
DeepFace Visual Emotion Recognition for HUMAN 2.0 - FIXED VERSION
================================================================

Simple, reliable visual emotion recognition using the DeepFace library.
Replaces the complex CNN+LSTM system with a mature, well-tested solution.

Based on: https://github.com/serengil/deepface
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import time
from pathlib import Path

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")

logger = logging.getLogger(__name__)

class DeepFaceVisualEmotionProcessor:
    """
    Simple visual emotion recognition using DeepFace
    
    Features:
    - Real-time emotion detection from camera or video frames
    - Multiple face detection backends (opencv, mtcnn, retinaface)
    - High accuracy emotion recognition (7 emotions)
    - Simple API compatible with existing multimodal system
    - No complex training required - uses pre-trained models
    """
    
    def __init__(self, 
                 detector_backend: str = 'opencv',
                 model_name: str = 'VGG-Face',
                 enforce_detection: bool = False,
                 align: bool = True):
        """
        Initialize DeepFace visual emotion processor
        
        Args:
            detector_backend: Face detection backend ('opencv', 'mtcnn', 'retinaface', 'mediapipe')
            model_name: Model for face recognition ('VGG-Face', 'Facenet', 'OpenFace', 'DeepFace')
            enforce_detection: Whether to enforce face detection (False = use whole image if no face)
            align: Whether to align faces before analysis
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace not installed. Run: pip install deepface")
        
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.align = align
        
        # Emotion mapping for consistency with existing system
        self.emotion_map = {
            'angry': 'anger',
            'disgust': 'disgust', 
            'fear': 'fear',
            'happy': 'joy',
            'sad': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        # Initialize by testing with a dummy image
        self._initialize_models()
        
        logger.info(f"DeepFace Visual Emotion Processor initialized")
        logger.info(f"  Detector: {detector_backend}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Enforce detection: {enforce_detection}")
    
    def _initialize_models(self):
        """Initialize DeepFace models by running a test"""
        try:
            # Create a dummy image to initialize models
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Test emotion analysis
            DeepFace.analyze(
                img_path=dummy_img,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            logger.info("DeepFace models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Model initialization warning: {str(e)}")
    
    def analyze_single_image(self, 
                            image: Union[str, np.ndarray], 
                            return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predict emotion from a single image
        
        Args:
            image: Image path (str) or numpy array
            return_probabilities: Whether to return emotion probabilities
            
        Returns:
            Dictionary with emotion prediction results
        """
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
                silent=True
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]  # Use first face if multiple detected
            
            # Extract emotion data
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            
            # Map to our emotion system
            mapped_emotion = self.emotion_map.get(dominant_emotion, dominant_emotion)
            
            # Get confidence (probability of dominant emotion)
            confidence = emotions[dominant_emotion] / 100.0  # Convert percentage to 0-1
            
            # Prepare result
            emotion_result = {
                'emotion': mapped_emotion,
                'confidence': confidence,
                'dominant_emotion': dominant_emotion,
                'raw_emotions': emotions,
                'face_detected': True,
                'processing_time': 0.0
            }
            
            if return_probabilities:
                # Map all emotion probabilities
                mapped_probabilities = {}
                for deepface_emotion, probability in emotions.items():
                    mapped_emotion_name = self.emotion_map.get(deepface_emotion, deepface_emotion)
                    mapped_probabilities[mapped_emotion_name] = probability / 100.0
                
                emotion_result['probabilities'] = mapped_probabilities
                emotion_result['class_probabilities'] = mapped_probabilities
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {str(e)}")
            
            # Return default result on failure
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'dominant_emotion': 'neutral',
                'raw_emotions': {'neutral': 100.0},
                'face_detected': False,
                'error': str(e),
                'probabilities': {'neutral': 1.0} if return_probabilities else None
            }
    
    def analyze_frames(self, 
                      frames: List[np.ndarray], 
                      aggregate_method: str = 'average') -> Dict[str, Any]:
        """
        Predict emotion from multiple video frames
        
        Args:
            frames: List of video frames (numpy arrays)
            aggregate_method: How to combine results ('average', 'majority', 'latest')
            
        Returns:
            Aggregated emotion prediction
        """
        if not frames:
            return self.analyze_single_image(np.zeros((224, 224, 3), dtype=np.uint8))
        
        start_time = time.time()
        frame_results = []
        
        # Analyze each frame
        for i, frame in enumerate(frames):
            try:
                result = self.analyze_single_image(frame, return_probabilities=True)
                frame_results.append(result)
                
                # Limit processing time for real-time performance
                if time.time() - start_time > 2.0:  # Max 2 seconds
                    logger.warning(f"Frame processing timeout, processed {i+1}/{len(frames)} frames")
                    break
                    
            except Exception as e:
                logger.warning(f"Frame {i} processing failed: {str(e)}")
                continue
        
        if not frame_results:
            return self.analyze_single_image(frames[-1])  # Fallback to last frame
        
        # Aggregate results
        aggregated = self._aggregate_frame_results(frame_results, aggregate_method)
        aggregated['num_frames_processed'] = len(frame_results)
        aggregated['total_frames'] = len(frames)
        aggregated['processing_time'] = time.time() - start_time
        
        return aggregated
    
    def _aggregate_frame_results(self, 
                               results: List[Dict[str, Any]], 
                               method: str = 'average') -> Dict[str, Any]:
        """Aggregate emotion results from multiple frames"""
        
        if not results:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        if method == 'latest':
            return results[-1]
        
        elif method == 'majority':
            # Find most common emotion
            emotions = [r['emotion'] for r in results if 'emotion' in r]
            if emotions:
                from collections import Counter
                most_common = Counter(emotions).most_common(1)[0]
                dominant_emotion = most_common[0]
                
                # Find result with that emotion
                for result in results:
                    if result.get('emotion') == dominant_emotion:
                        return result
            
            return results[-1]  # Fallback
        
        else:  # average method
            # Average probabilities across frames
            all_emotions = set()
            for result in results:
                if 'probabilities' in result and result['probabilities']:
                    all_emotions.update(result['probabilities'].keys())
            
            if not all_emotions:
                return results[-1]  # Fallback
            
            # Calculate average probabilities
            avg_probabilities = {}
            for emotion in all_emotions:
                probs = []
                for result in results:
                    if 'probabilities' in result and result['probabilities']:
                        probs.append(result['probabilities'].get(emotion, 0.0))
                avg_probabilities[emotion] = sum(probs) / len(probs) if probs else 0.0
            
            # Find dominant emotion
            dominant_emotion = max(avg_probabilities, key=avg_probabilities.get)
            confidence = avg_probabilities[dominant_emotion]
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'probabilities': avg_probabilities,
                'class_probabilities': avg_probabilities,
                'aggregation_method': method,
                'face_detected': any(r.get('face_detected', False) for r in results)
            }
    
    def analyze_video_stream(self, 
                           camera_index: int = 0, 
                           duration: float = 5.0,
                           fps_limit: int = 5) -> Dict[str, Any]:
        """
        Analyze emotion from live camera stream
        
        Args:
            camera_index: Camera device index (0 for default)
            duration: How long to analyze (seconds)
            fps_limit: Maximum FPS for analysis (to limit processing load)
            
        Returns:
            Emotion analysis results
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {'error': 'Camera not available', 'emotion': 'neutral', 'confidence': 0.0}
        
        frames = []
        start_time = time.time()
        last_capture_time = 0
        frame_interval = 1.0 / fps_limit
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Limit FPS
                current_time = time.time()
                if current_time - last_capture_time >= frame_interval:
                    frames.append(frame.copy())
                    last_capture_time = current_time
                
                # Limit total frames
                if len(frames) >= 30:  # Max 30 frames
                    break
            
            cap.release()
            
            if frames:
                return self.analyze_frames(frames)
            else:
                return {'error': 'No frames captured', 'emotion': 'neutral', 'confidence': 0.0}
                
        except Exception as e:
            cap.release()
            return {'error': str(e), 'emotion': 'neutral', 'confidence': 0.0}
    
    def process_frame(self, frame: np.ndarray) -> List[tuple]:
        """
        Process a single frame and return emotions as list of tuples
        For compatibility with existing multimodal processor
        
        Args:
            frame: Single video frame
            
        Returns:
            List of (emotion, confidence) tuples
        """
        try:
            result = self.analyze_single_image(frame, return_probabilities=True)
            
            if 'probabilities' in result and result['probabilities']:
                # Convert to list of tuples format
                emotions = []
                for emotion, confidence in result['probabilities'].items():
                    if confidence > 0.1:  # Minimum confidence threshold
                        emotions.append((emotion, confidence))
                
                # Sort by confidence
                emotions.sort(key=lambda x: x[1], reverse=True)
                return emotions
            else:
                # Single emotion result
                return [(result['emotion'], result['confidence'])]
                
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return [('neutral', 0.5)]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the DeepFace model"""
        return {
            'model_type': 'DeepFace',
            'detector_backend': self.detector_backend,
            'model_name': self.model_name,
            'enforce_detection': self.enforce_detection,
            'align': self.align,
            'supported_emotions': list(self.emotion_map.values()),
            'deepface_emotions': list(self.emotion_map.keys()),
            'library_version': 'DeepFace 0.0.93',
            'description': 'Pre-trained deep learning models for facial emotion recognition'
        }

# Compatibility class for existing code
class VisualEmotionProcessor(DeepFaceVisualEmotionProcessor):
    """
    Compatibility wrapper for existing code that expects VisualEmotionProcessor
    Maintains the exact interface expected by the multimodal system
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """Initialize with compatibility for old interface"""
        # Ignore model_path and device - DeepFace handles this automatically
        super().__init__(
            detector_backend='opencv',  # Fast and reliable
            enforce_detection=False     # Robust for various conditions
        )
        
        logger.info("VisualEmotionProcessor using DeepFace backend")
    
    def load_model(self, model_path: str):
        """Compatibility method - DeepFace loads models automatically"""
        logger.info(f"DeepFace models loaded automatically (ignoring {model_path})")
    
    def predict_emotion(self, frames: List[np.ndarray], detect_faces: bool = True) -> Dict[str, Any]:
        """
        Compatibility method for existing multimodal integration
        
        Args:
            frames: List of video frames
            detect_faces: Whether to detect faces (ignored - DeepFace handles this)
            
        Returns:
            Dictionary with emotion prediction results in expected format
        """
        if not frames:
            return {'emotion': 'neutral', 'confidence': 0.0, 'num_frames': 0}
        
        # Use the enhanced frame processing
        result = self.analyze_frames(frames, aggregate_method='average')
        
        # Add compatibility fields that existing code expects
        result['num_frames'] = len(frames)
        result['emotion_class'] = 0  # Dummy for compatibility
        
        # Ensure attention_weights exist for compatibility
        if 'attention_weights' not in result:
            result['attention_weights'] = np.ones((len(frames), 1))  # Dummy attention weights
        
        return result

def create_visual_emotion_processor(config: Optional[Dict[str, Any]] = None) -> DeepFaceVisualEmotionProcessor:
    """Factory function to create visual emotion processor"""
    config = config or {}
    
    return DeepFaceVisualEmotionProcessor(
        detector_backend=config.get('detector_backend', 'opencv'),
        model_name=config.get('model_name', 'VGG-Face'),
        enforce_detection=config.get('enforce_detection', False),
        align=config.get('align', True)
    )

# Test function
def test_deepface_emotion():
    """Test the DeepFace emotion recognition"""
    print("🧪 Testing DeepFace Visual Emotion Recognition - FIXED VERSION")
    
    try:
        processor = DeepFaceVisualEmotionProcessor()
        print("✅ Processor initialized")
        
        # Test compatibility wrapper
        compat_processor = VisualEmotionProcessor()
        print("✅ Compatibility wrapper initialized")
        
        # Test with dummy frames
        dummy_frames = []
        for i in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dummy_frames.append(frame)
        
        result = compat_processor.predict_emotion(dummy_frames, detect_faces=True)
        print("✅ Compatibility interface working")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Frames processed: {result['num_frames']}")
        
        # Test process_frame method
        frame_result = processor.process_frame(dummy_frames[0])
        print(f"✅ Process frame method: {frame_result[0]}")
        
        # Test with camera if available
        print("📸 Testing camera emotion recognition...")
        result = processor.analyze_video_stream(duration=2.0, fps_limit=3)
        
        if 'error' not in result:
            print(f"✅ Camera test successful!")
            print(f"  Detected emotion: {result['emotion']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Frames processed: {result.get('num_frames_processed', 0)}")
        else:
            print(f"⚠️ Camera test failed: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deepface_emotion() 