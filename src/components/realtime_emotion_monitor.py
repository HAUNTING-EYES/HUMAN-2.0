#!/usr/bin/env python3
"""
Real-time Emotion Monitor for HUMAN 2.0
Captures audio and video to provide continuous emotion analysis
"""

import cv2
import numpy as np
import pyaudio
import threading
import queue
import time
import json
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import librosa
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings('ignore')

from .multimodal_emotion_processor import MultimodalEmotionProcessor, MultimodalEmotionState, EmotionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionHistory:
    """Stores emotion history for trend analysis"""
    emotions: deque
    timestamps: deque
    max_history: int = 100
    
    def __post_init__(self):
        if not hasattr(self.emotions, 'maxlen'):
            self.emotions = deque(maxlen=self.max_history)
        if not hasattr(self.timestamps, 'maxlen'):
            self.timestamps = deque(maxlen=self.max_history)
    
    def add_emotion(self, emotion: str, confidence: float, timestamp: float):
        """Add emotion to history"""
        self.emotions.append((emotion, confidence))
        self.timestamps.append(timestamp)
    
    def get_recent_emotions(self, seconds: int = 30) -> list:
        """Get emotions from the last N seconds"""
        current_time = time.time()
        recent_emotions = []
        
        for i, timestamp in enumerate(self.timestamps):
            if current_time - timestamp <= seconds:
                recent_emotions.append(self.emotions[i])
        
        return recent_emotions
    
    def get_emotion_trend(self) -> Dict[str, float]:
        """Get emotion trend over time"""
        if not self.emotions:
            return {}
        
        emotion_counts = {}
        total_confidence = {}
        
        for emotion, confidence in self.emotions:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                total_confidence[emotion] = 0
            emotion_counts[emotion] += 1
            total_confidence[emotion] += confidence
        
        # Calculate average confidence for each emotion
        emotion_trends = {}
        for emotion in emotion_counts:
            emotion_trends[emotion] = total_confidence[emotion] / emotion_counts[emotion]
        
        return emotion_trends

class AudioCapture:
    """Handles real-time audio capture"""
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024, channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = deque(maxlen=50)  # Store last 50 chunks
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
    
    def start_recording(self):
        """Start audio recording"""
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            logger.info("Audio recording started")
        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        logger.info("Audio recording stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function"""
        if self.is_recording:
            try:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self.audio_buffer.append(audio_data)
                self.audio_queue.put(audio_data)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get the latest audio chunk"""
        try:
            if not self.audio_queue.empty():
                return self.audio_queue.get_nowait()
        except queue.Empty:
            pass
        return None
    
    def get_audio_buffer(self) -> np.ndarray:
        """Get the entire audio buffer"""
        if self.audio_buffer:
            return np.concatenate(list(self.audio_buffer))
        return np.array([])

class VideoCapture:
    """Handles real-time video capture"""
    
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=5)  # Keep only latest frames
        self.is_recording = False
        self.cap = None
    
    def start_recording(self):
        """Start video recording"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            self.is_recording = True
            self._video_thread = threading.Thread(target=self._video_capture_loop)
            self._video_thread.daemon = True
            self._video_thread.start()
            logger.info("Video recording started")
        except Exception as e:
            logger.error(f"Error starting video recording: {e}")
    
    def stop_recording(self):
        """Stop video recording"""
        self.is_recording = False
        if self.cap:
            self.cap.release()
        logger.info("Video recording stopped")
    
    def _video_capture_loop(self):
        """Video capture loop"""
        while self.is_recording:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Put frame in queue, remove old frames if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error in video capture loop: {e}")
                break
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest video frame"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        return None

class EmotionVisualizer:
    """Visualizes emotion data in real-time"""
    
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-time Emotion Analysis', fontsize=16)
        
        # Emotion history for plotting
        self.emotion_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
        # Color mapping for emotions
        self.emotion_colors = {
            'joy': 'yellow',
            'sadness': 'blue',
            'anger': 'red',
            'fear': 'purple',
            'surprise': 'orange',
            'disgust': 'green',
            'neutral': 'gray'
        }
    
    def update_plot(self, emotion_state: MultimodalEmotionState):
        """Update the emotion visualization"""
        try:
            # Add current emotion to history
            self.emotion_history.append(emotion_state.dominant_emotion)
            self.time_history.append(emotion_state.timestamp)
            
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot 1: Emotion timeline
            if len(self.emotion_history) > 1:
                emotions = list(self.emotion_history)
                times = list(self.time_history)
                
                # Convert emotions to numeric values for plotting
                emotion_to_num = {emotion: i for i, emotion in enumerate(self.emotion_colors.keys())}
                emotion_nums = [emotion_to_num.get(emotion, 0) for emotion in emotions]
                
                self.ax1.plot(times, emotion_nums, 'b-', linewidth=2)
                self.ax1.set_ylabel('Emotion')
                self.ax1.set_title('Emotion Timeline')
                self.ax1.set_yticks(list(emotion_to_num.values()))
                self.ax1.set_yticklabels(list(emotion_to_num.keys()))
                self.ax1.grid(True, alpha=0.3)
            
            # Plot 2: Current emotion confidence
            if emotion_state.combined_emotions:
                emotions = [e.emotion for e in emotion_state.combined_emotions[:5]]
                confidences = [e.confidence for e in emotion_state.combined_emotions[:5]]
                colors = [self.emotion_colors.get(emotion, 'gray') for emotion in emotions]
                
                bars = self.ax2.bar(emotions, confidences, color=colors, alpha=0.7)
                self.ax2.set_ylabel('Confidence')
                self.ax2.set_title('Current Emotion Confidence')
                self.ax2.set_ylim(0, 1)
                
                # Add confidence values on bars
                for bar, conf in zip(bars, confidences):
                    height = bar.get_height()
                    self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{conf:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            logger.error(f"Error updating plot: {e}")

class RealtimeEmotionMonitor:
    """Main real-time emotion monitoring system"""
    
    def __init__(self, emotion_callback: Optional[Callable] = None):
        self.emotion_processor = MultimodalEmotionProcessor()
        self.audio_capture = AudioCapture()
        self.video_capture = VideoCapture()
        self.emotion_history = EmotionHistory()
        self.emotion_visualizer = EmotionVisualizer()
        
        self.is_monitoring = False
        self.emotion_callback = emotion_callback
        self.monitoring_thread = None
        
        # Processing intervals
        self.audio_interval = 1.0  # Process audio every 1 second
        self.video_interval = 0.5  # Process video every 0.5 seconds
        self.last_audio_process = 0
        self.last_video_process = 0
    
    def start_monitoring(self):
        """Start real-time emotion monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        try:
            # Start audio and video capture
            self.audio_capture.start_recording()
            self.video_capture.start_recording()
            
            # Start monitoring thread
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Real-time emotion monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time emotion monitoring"""
        self.is_monitoring = False
        
        # Stop audio and video capture
        self.audio_capture.stop_recording()
        self.video_capture.stop_recording()
        
        logger.info("Real-time emotion monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Process audio
                if current_time - self.last_audio_process >= self.audio_interval:
                    audio_data = self.audio_capture.get_audio_buffer()
                    if len(audio_data) > 0:
                        self._process_audio(audio_data)
                    self.last_audio_process = current_time
                
                # Process video
                if current_time - self.last_video_process >= self.video_interval:
                    frame = self.video_capture.get_latest_frame()
                    if frame is not None:
                        self._process_video(frame)
                    self.last_video_process = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _process_audio(self, audio_data: np.ndarray):
        """Process audio data"""
        try:
            emotion_state = self.emotion_processor.process_multimodal_input(audio_data=audio_data)
            self._handle_emotion_result(emotion_state)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    def _process_video(self, frame: np.ndarray):
        """Process video frame"""
        try:
            emotion_state = self.emotion_processor.process_multimodal_input(frame=frame)
            self._handle_emotion_result(emotion_state)
        except Exception as e:
            logger.error(f"Error processing video: {e}")
    
    def _handle_emotion_result(self, emotion_state: MultimodalEmotionState):
        """Handle emotion processing result"""
        try:
            # Add to history
            self.emotion_history.add_emotion(
                emotion_state.dominant_emotion,
                emotion_state.confidence,
                emotion_state.timestamp
            )
            
            # Update visualization
            self.emotion_visualizer.update_plot(emotion_state)
            
            # Call callback if provided
            if self.emotion_callback:
                self.emotion_callback(emotion_state)
            
            # Log significant emotions
            if emotion_state.confidence > 0.7:
                logger.info(f"High confidence emotion detected: {emotion_state.dominant_emotion} "
                          f"(confidence: {emotion_state.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error handling emotion result: {e}")
    
    def get_current_emotion_state(self) -> MultimodalEmotionState:
        """Get the current emotional state"""
        # This would return the most recent emotion state
        # For now, return a neutral state
        return MultimodalEmotionState(
            text_emotions=[],
            audio_emotions=[],
            visual_emotions=[],
            combined_emotions=[EmotionResult('neutral', 0.5, 'multimodal', time.time())],
            dominant_emotion='neutral',
            emotional_intensity=0.5,
            confidence=0.5,
            timestamp=time.time()
        )
    
    def get_emotion_trend(self) -> Dict[str, float]:
        """Get emotion trend over time"""
        return self.emotion_history.get_emotion_trend()
    
    def add_text_input(self, text: str):
        """Add text input for emotion analysis"""
        try:
            emotion_state = self.emotion_processor.process_multimodal_input(text=text)
            self._handle_emotion_result(emotion_state)
        except Exception as e:
            logger.error(f"Error processing text input: {e}")

class SimpleEmotionGUI:
    """Simple GUI for emotion monitoring"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HUMAN 2.0 Emotion Monitor")
        self.root.geometry("600x400")
        
        self.monitor = RealtimeEmotionMonitor(emotion_callback=self._on_emotion_update)
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Setup the GUI components"""
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", 
                                      command=self._start_monitoring)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", 
                                     command=self._stop_monitoring, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready to start monitoring")
        self.status_label.pack(pady=5)
        
        # Emotion display frame
        emotion_frame = ttk.LabelFrame(self.root, text="Current Emotion")
        emotion_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.emotion_label = ttk.Label(emotion_frame, text="No emotion detected", 
                                      font=('Arial', 16))
        self.emotion_label.pack(pady=20)
        
        self.confidence_label = ttk.Label(emotion_frame, text="Confidence: 0.0")
        self.confidence_label.pack()
        
        # Text input frame
        text_frame = ttk.LabelFrame(self.root, text="Text Input")
        text_frame.pack(fill='x', padx=10, pady=5)
        
        self.text_entry = ttk.Entry(text_frame)
        self.text_entry.pack(side='left', fill='x', expand=True, padx=5, pady=5)
        
        self.text_button = ttk.Button(text_frame, text="Analyze Text", 
                                     command=self._analyze_text)
        self.text_button.pack(side='right', padx=5, pady=5)
    
    def _start_monitoring(self):
        """Start emotion monitoring"""
        self.monitor.start_monitoring()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Monitoring active")
    
    def _stop_monitoring(self):
        """Stop emotion monitoring"""
        self.monitor.stop_monitoring()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Monitoring stopped")
    
    def _analyze_text(self):
        """Analyze text input"""
        text = self.text_entry.get().strip()
        if text:
            self.monitor.add_text_input(text)
            self.text_entry.delete(0, tk.END)
    
    def _on_emotion_update(self, emotion_state: MultimodalEmotionState):
        """Handle emotion updates"""
        self.emotion_label.config(text=f"Dominant: {emotion_state.dominant_emotion}")
        self.confidence_label.config(text=f"Confidence: {emotion_state.confidence:.3f}")
    
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        finally:
            self.monitor.stop_monitoring()

def main():
    """Test the real-time emotion monitor"""
    print("Starting Real-time Emotion Monitor...")
    
    # Create and run GUI
    gui = SimpleEmotionGUI()
    gui.run()

if __name__ == "__main__":
    main() 