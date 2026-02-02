"""
Real-Time Visual Emotion Recognition System
Optimized for AMD GPU with DirectML, using webcam for live emotion detection
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from typing import Dict, List, Tuple, Optional
import time
import threading
import queue
from dataclasses import dataclass
import logging
from pathlib import Path
import os

@dataclass
class VisualEmotionResult:
    """Result from visual emotion recognition"""
    primary_emotion: str
    confidence: float
    secondary_emotions: List[Tuple[str, float]]
    face_detected: bool
    face_count: int
    processing_time: float
    timestamp: float

class SimpleVisualEmotionModel(nn.Module):
    """Simple CNN-based visual emotion model (AMD GPU compatible)"""
    
    def __init__(self, num_classes: int = 7):
        super(SimpleVisualEmotionModel, self).__init__()
        
        # Use ResNet18 as backbone (pre-trained)
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Custom classifier for emotions
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Emotion labels
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        
    def forward(self, x):
        # Extract features with backbone
        features = self.backbone(x)  # (batch_size, 512)
        
        # Classify emotions
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities, features

class RealTimeVisualEmotionRecognizer:
    """Real-time visual emotion recognition using webcam"""
    
    def __init__(self, device: str = 'auto', camera_id: int = 0, model_path: str = None):
        self.device = self._get_device(device)
        self.camera_id = camera_id
        
        # Initialize model
        self.model = SimpleVisualEmotionModel()
        
        # Load trained model if available
        if model_path is None:
            model_path = "models/visual_emotion_model_robust.pth"
        
        if self.load_trained_model(model_path):
            self.model.to(self.device)
            self.model.eval()
        else:
            print("🔄 Using untrained model (random weights)")
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Emotion mapping
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        
        # Camera and threading
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print(f"🎥 Visual emotion recognizer initialized on device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device with AMD GPU support"""
        if device == 'auto':
            # Try DirectML for AMD GPU
            try:
                import torch_directml
                dml_device = torch_directml.device()
                print(f"Using DirectML device: {dml_device}")
                return dml_device
            except ImportError:
                print("DirectML not available, using CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
        """Preprocess face ROI for model input"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        face_tensor = self.transform(face_rgb)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def predict_emotion(self, face_tensor: torch.Tensor) -> VisualEmotionResult:
        """Predict emotion from face tensor"""
        start_time = time.time()
        
        with torch.no_grad():
            logits, probabilities, features = self.model(face_tensor)
            
            # Get predictions
            probs = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            # Get top 3 emotions
            top_indices = np.argsort(probs)[::-1][:3]
            secondary_emotions = [
                (self.emotion_labels[i], float(probs[i])) 
                for i in top_indices[1:] if probs[i] > 0.1
            ]
            
            processing_time = time.time() - start_time
            
            return VisualEmotionResult(
                primary_emotion=self.emotion_labels[predicted_class],
                confidence=confidence,
                secondary_emotions=secondary_emotions,
                face_detected=True,
                face_count=1,
                processing_time=processing_time,
                timestamp=time.time()
            )
    
    def process_frame(self, frame: np.ndarray) -> Optional[VisualEmotionResult]:
        """Process a single frame for emotion recognition"""
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return VisualEmotionResult(
                    primary_emotion="no_face",
                    confidence=0.0,
                    secondary_emotions=[],
                    face_detected=False,
                    face_count=0,
                    processing_time=0.0,
                    timestamp=time.time()
                )
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Add padding to face ROI
            padding = int(min(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess and predict
            face_tensor = self.preprocess_face(face_roi)
            result = self.predict_emotion(face_tensor)
            result.face_count = len(faces)
            
            return result
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def start_camera(self) -> bool:
        """Start camera capture with working Media Foundation backend"""
        try:
            # Use Media Foundation backend (known working)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {self.camera_id} with Media Foundation backend")
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    print(f"❌ Failed to open camera {self.camera_id} with default backend")
                    return False
                print(f"✅ Camera {self.camera_id} opened with default backend")
            else:
                print(f"✅ Camera {self.camera_id} opened with Media Foundation backend")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
    
    def capture_loop(self):
        """Camera capture loop (runs in separate thread)"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
    
    def processing_loop(self):
        """Emotion processing loop (runs in separate thread)"""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process emotion
                result = self.process_frame(frame)
                
                if result:
                    # Add result to queue
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # Remove oldest result
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
                # Update FPS
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def start_real_time_recognition(self):
        """Start real-time emotion recognition"""
        if not self.start_camera():
            return False
        
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.processing_thread = threading.Thread(target=self.processing_loop)
        
        self.capture_thread.daemon = True
        self.processing_thread.daemon = True
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        print("🎥 Real-time visual emotion recognition started!")
        return True
    
    def stop_real_time_recognition(self):
        """Stop real-time emotion recognition"""
        self.is_running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        self.stop_camera()
        print("🛑 Real-time visual emotion recognition stopped")
    
    def get_latest_result(self) -> Optional[VisualEmotionResult]:
        """Get the latest emotion recognition result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[VisualEmotionResult]:
        """Get all available emotion recognition results"""
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def display_live_feed(self):
        """Display live camera feed with emotion annotations"""
        if not self.cap:
            print("❌ Camera not started")
            return
        
        print("🎥 Displaying live feed with emotion recognition...")
        print("Press 'q' to quit, 'r' to reset, 's' to save screenshot")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Get latest emotion result
            result = self.get_latest_result()
            
            # Detect faces for visualization
            faces = self.detect_faces(frame)
            
            # Draw face rectangles and emotions
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw emotion text
                if result and result.face_detected:
                    emotion_text = f"{result.primary_emotion}: {result.confidence:.2f}"
                    cv2.putText(frame, emotion_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw secondary emotions
                    for i, (emotion, conf) in enumerate(result.secondary_emotions):
                        secondary_text = f"{emotion}: {conf:.2f}"
                        cv2.putText(frame, secondary_text, (x, y-30-i*20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw FPS and info
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if result:
                cv2.putText(frame, f"Device: {self.device}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Process Time: {result.processing_time*1000:.1f}ms", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Display frame
            cv2.imshow('Real-Time Visual Emotion Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Resetting...")
                # Clear queues
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"emotion_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
    
    def load_trained_model(self, model_path: str = "models/visual_emotion_model_amd_optimized.pth"):
        """Load trained model weights"""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found: {model_path}")
                return False
            
            # Load checkpoint with device mapping fix for DirectML
            device_str = 'cpu'  # Always load to CPU first to avoid device comparison issues
            checkpoint = torch.load(model_path, map_location=device_str)
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained model from {model_path}")
                print(f"📊 Model accuracy: {checkpoint.get('accuracy', 'N/A')}%")
                return True
            else:
                # Try loading directly as state dict
                self.model.load_state_dict(checkpoint)
                print(f"✅ Loaded trained model from {model_path}")
                return True
                
        except Exception as e:
            print(f"⚠️ Failed to load trained model: {e}")
            print(f"   Error details: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("🎥 Real-Time Visual Emotion Recognition System")
    print("=" * 60)
    
    try:
        # Initialize recognizer
        recognizer = RealTimeVisualEmotionRecognizer(camera_id=0)
        
        # Start real-time recognition
        if recognizer.start_real_time_recognition():
            # Display live feed
            recognizer.display_live_feed()
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        # Clean up
        if 'recognizer' in locals():
            recognizer.stop_real_time_recognition()
        print("👋 Visual emotion recognition stopped") 