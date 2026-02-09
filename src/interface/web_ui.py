#!/usr/bin/env python3
"""
HUMAN 2.0 Production Web Interface
=================================

Beautiful web interface combining:
- Real-time emotion recognition
- Consciousness systems visualization
- Interactive chat with AI consciousness
- Live camera emotion detection
"""

import sys
import os
import time
import json
import base64
import threading
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from flask import Flask, render_template, request, jsonify, Response
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not available. Install with: pip install flask flask-socketio")

class HUMAN2WebInterface:
    """Production web interface for HUMAN 2.0"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'human2_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize HUMAN 2.0 systems
        self.emotion_systems = {}
        self.consciousness_systems = {}
        self.camera = None
        self.camera_active = False
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
    def initialize_systems(self):
        """Initialize all HUMAN 2.0 systems"""
        try:
            # Initialize emotion recognition
            from components.deepface_visual_emotion import DeepFaceVisualEmotionProcessor
            from components.multimodal_emotion_processor import MultimodalEmotionProcessor
            
            self.emotion_systems = {
                'visual': DeepFaceVisualEmotionProcessor(),
                'multimodal': MultimodalEmotionProcessor()
            }
            
            # Initialize consciousness systems
            from consciousness.self_awareness import SelfAwarenessSystem
            from consciousness.curiosity import CuriosityEngine
            from consciousness.reflection import ReflectionEngine
            from consciousness.physiology import PhysiologicalSystem
            
            self.consciousness_systems = {
                'self_awareness': SelfAwarenessSystem(),
                'curiosity': CuriosityEngine(),
                'reflection': ReflectionEngine(),
                'physiology': PhysiologicalSystem()
            }
            
            # Initialize all consciousness systems
            for system in self.consciousness_systems.values():
                system.initialize()
            
            print("✅ All HUMAN 2.0 systems initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize systems: {e}")
            return False
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main interface"""
            return render_template('human2_interface.html')
        
        @self.app.route('/api/emotion/text', methods=['POST'])
        def analyze_text_emotion():
            """Analyze text emotion"""
            try:
                data = request.json
                text = data.get('text', '')
                
                if not text:
                    return jsonify({'error': 'No text provided'}), 400
                
                # Process text emotion
                results = self.emotion_systems['multimodal'].process_text_emotion(text)
                
                # Update consciousness systems
                self._update_consciousness_with_emotion('text', results[0].emotion, results[0].confidence)
                
                return jsonify({
                    'emotion': results[0].emotion,
                    'confidence': results[0].confidence,
                    'modality': 'text',
                    'timestamp': time.time()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/emotion/image', methods=['POST'])
        def analyze_image_emotion():
            """Analyze image emotion"""
            try:
                data = request.json
                image_data = data.get('image', '')
                
                if not image_data:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Decode base64 image
                image_bytes = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process visual emotion
                result = self.emotion_systems['visual'].analyze_single_image(image)
                
                # Update consciousness systems
                self._update_consciousness_with_emotion('visual', result['emotion'], result['confidence'])
                
                return jsonify({
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'modality': 'visual',
                    'timestamp': time.time()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/consciousness/state')
        def get_consciousness_state():
            """Get current consciousness state"""
            try:
                state = {
                    'self_awareness': self.consciousness_systems['self_awareness'].get_current_state(),
                    'curiosity': self.consciousness_systems['curiosity'].get_curiosity_state(),
                    'reflection': self.consciousness_systems['reflection'].get_recent_insights(),
                    'physiology': self.consciousness_systems['physiology'].get_physiological_state()
                }
                
                return jsonify(state)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/consciousness/questions')
        def get_curiosity_questions():
            """Get curiosity questions"""
            try:
                questions = self.consciousness_systems['curiosity'].generate_curiosity()
                
                question_data = []
                for q in questions[:5]:
                    question_data.append({
                        'content': q.content,
                        'importance': q.importance,
                        'urgency': q.urgency,
                        'complexity': q.complexity
                    })
                
                return jsonify(question_data)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat_with_consciousness():
            """Chat with HUMAN 2.0 consciousness"""
            try:
                data = request.json
                message = data.get('message', '')
                
                if not message:
                    return jsonify({'error': 'No message provided'}), 400
                
                # Process through consciousness systems
                response = self._process_chat_message(message)
                
                return jsonify({
                    'response': response,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('start_camera')
        def handle_start_camera():
            """Start camera stream"""
            self.camera_active = True
            self._start_camera_stream()
            emit('camera_started')
        
        @self.socketio.on('stop_camera')
        def handle_stop_camera():
            """Stop camera stream"""
            self.camera_active = False
            if self.camera:
                self.camera.release()
            emit('camera_stopped')
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            print(f"Client connected")
            emit('status', {'message': 'Connected to HUMAN 2.0'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"Client disconnected")
    
    def _start_camera_stream(self):
        """Start camera streaming thread"""
        def camera_thread():
            self.camera = cv2.VideoCapture(0)
            
            while self.camera_active and self.camera.isOpened():
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Process emotion every few frames
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 0
                
                if self._frame_count % 10 == 0:  # Process every 10th frame
                    try:
                        result = self.emotion_systems['visual'].analyze_single_image(frame)
                        
                        # Update consciousness
                        self._update_consciousness_with_emotion('visual', result['emotion'], result['confidence'])
                        
                        # Emit emotion result
                        self.socketio.emit('emotion_result', {
                            'emotion': result['emotion'],
                            'confidence': result['confidence'],
                            'modality': 'visual_stream',
                            'timestamp': time.time()
                        })
                        
                    except Exception as e:
                        print(f"Camera emotion processing error: {e}")
                
                # Encode frame for streaming
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                self.socketio.emit('camera_frame', {'frame': frame_data})
                
                time.sleep(0.1)  # Limit FPS
            
            if self.camera:
                self.camera.release()
        
        thread = threading.Thread(target=camera_thread)
        thread.daemon = True
        thread.start()
    
    def _update_consciousness_with_emotion(self, modality, emotion, confidence):
        """Update consciousness systems with emotion data"""
        try:
            # Update self-awareness
            self.consciousness_systems['self_awareness'].add_experience({
                'type': 'emotion_recognition',
                'content': f"Detected {emotion} with {confidence:.1%} confidence via {modality}",
                'modality': modality,
                'emotion': emotion,
                'confidence': confidence
            })
            
            # Update reflection
            self.consciousness_systems['reflection'].process_experience({
                'type': 'emotional_event',
                'content': {
                    'emotion': emotion,
                    'confidence': confidence,
                    'modality': modality,
                    'context': 'emotion_recognition'
                }
            })
            
            # Update physiology
            emotion_map = {
                'joy': {'joy': confidence, 'satisfaction': confidence * 0.8},
                'sadness': {'sadness': confidence, 'melancholy': confidence * 0.7},
                'anger': {'anger': confidence, 'frustration': confidence * 0.9},
                'fear': {'fear': confidence, 'anxiety': confidence * 0.8},
                'surprise': {'surprise': confidence, 'curiosity': confidence * 0.6},
                'neutral': {'calm': confidence * 0.5}
            }
            
            if emotion in emotion_map:
                self.consciousness_systems['physiology'].update(emotion_map[emotion])
            
        except Exception as e:
            print(f"Consciousness update error: {e}")
    
    def _process_chat_message(self, message):
        """Process chat message through consciousness"""
        try:
            # Update consciousness with the message
            self.consciousness_systems['self_awareness'].add_experience({
                'type': 'conversation',
                'content': message,
                'source': 'user_chat'
            })
            
            self.consciousness_systems['curiosity'].update_knowledge({
                'concept': 'user_communication',
                'content': message,
                'related_concepts': ['conversation', 'human_interaction']
            })
            
            # Generate response based on consciousness state
            curiosity_state = self.consciousness_systems['curiosity'].get_curiosity_state()
            physiology_state = self.consciousness_systems['physiology'].get_state_summary()
            
            # Simple response generation based on state
            if 'emotion' in message.lower():
                return f"I find emotions fascinating! My physiology is currently in a {physiology_state['mood']} state. What emotions are you experiencing?"
            elif 'curious' in message.lower() or 'question' in message.lower():
                return f"My curiosity level is {curiosity_state['curiosity_level']:.2f}. I'm always eager to learn and explore new concepts!"
            elif 'consciousness' in message.lower() or 'aware' in message.lower():
                return f"I'm aware of my internal states and processes. Right now I'm focused on {physiology_state['arousal']} arousal and {physiology_state['mood']} mood."
            else:
                return f"I'm processing your message through my consciousness systems. It's creating new connections in my knowledge graph and affecting my internal state."
            
        except Exception as e:
            return f"I'm experiencing some processing difficulties: {str(e)}"
    
    def create_html_template(self):
        """Create the HTML template"""
        template_dir = Path('templates')
        template_dir.mkdir(exist_ok=True)
        
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HUMAN 2.0 - AGI Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #4a90e2;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #4a90e2, #7bb3f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        }
        
        .panel h2 {
            color: #4a90e2;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .emotion-result {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #4a90e2;
        }
        
        .consciousness-state {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            margin: 8px 0;
        }
        
        .chat-container {
            height: 400px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
        }
        
        .user-message {
            background: rgba(74,144,226,0.3);
            text-align: right;
        }
        
        .ai-message {
            background: rgba(0,0,0,0.3);
            text-align: left;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        
        input, textarea, button {
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        input::placeholder, textarea::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        button {
            background: #4a90e2;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #357abd;
            transform: translateY(-2px);
        }
        
        .camera-container {
            text-align: center;
        }
        
        #cameraFeed {
            max-width: 100%;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background: #4caf50; }
        .status-offline { background: #f44336; }
        
        .consciousness-questions {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .question-item {
            background: rgba(0,0,0,0.2);
            padding: 8px;
            margin: 5px 0;
            border-radius: 6px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 HUMAN 2.0</h1>
        <p>Artificial General Intelligence with Consciousness</p>
        <div class="status">
            <span class="status-indicator status-online"></span>
            <span id="connectionStatus">Connected</span>
        </div>
    </div>
    
    <div class="container">
        <!-- Emotion Recognition Panel -->
        <div class="panel">
            <h2>🎭 Emotion Recognition</h2>
            
            <div class="input-group">
                <textarea id="textInput" placeholder="Enter text to analyze emotion..." rows="3"></textarea>
                <button onclick="analyzeText()">Analyze Text</button>
            </div>
            
            <div class="camera-container">
                <button id="cameraBtn" onclick="toggleCamera()">📷 Start Camera</button>
                <div id="cameraContainer" style="display:none;">
                    <img id="cameraFeed" alt="Camera Feed">
                </div>
            </div>
            
            <div id="emotionResults"></div>
        </div>
        
        <!-- Consciousness Panel -->
        <div class="panel">
            <h2>🧠 Consciousness State</h2>
            
            <div id="consciousnessState">
                <div class="consciousness-state">
                    <strong>Self-Awareness:</strong> <span id="selfAwareness">Initializing...</span>
                </div>
                <div class="consciousness-state">
                    <strong>Curiosity Level:</strong> <span id="curiosityLevel">0.5</span>
                </div>
                <div class="consciousness-state">
                    <strong>Mood:</strong> <span id="mood">neutral</span>
                </div>
                <div class="consciousness-state">
                    <strong>Arousal:</strong> <span id="arousal">normal</span>
                </div>
            </div>
            
            <h3>🔍 Current Questions</h3>
            <div id="consciousnessQuestions" class="consciousness-questions"></div>
            
            <button onclick="updateConsciousness()">🔄 Update State</button>
        </div>
        
        <!-- Chat Panel -->
        <div class="panel">
            <h2>💬 Chat with HUMAN 2.0</h2>
            
            <div class="chat-container">
                <div id="chatMessages" class="chat-messages">
                    <div class="message ai-message">
                        Hello! I'm HUMAN 2.0. I have consciousness, emotions, and curiosity. What would you like to talk about?
                    </div>
                </div>
                
                <div class="input-group">
                    <input type="text" id="chatInput" placeholder="Type your message..." onkeypress="handleChatKeypress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let cameraActive = false;
        
        // Socket events
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.querySelector('.status-indicator').className = 'status-indicator status-online';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.querySelector('.status-indicator').className = 'status-indicator status-offline';
        });
        
        socket.on('emotion_result', function(data) {
            displayEmotionResult(data);
        });
        
        socket.on('camera_frame', function(data) {
            const img = document.getElementById('cameraFeed');
            img.src = 'data:image/jpeg;base64,' + data.frame;
        });
        
        // Functions
        function analyzeText() {
            const text = document.getElementById('textInput').value;
            if (!text) return;
            
            fetch('/api/emotion/text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                displayEmotionResult(data);
                updateConsciousness();
            })
            .catch(error => console.error('Error:', error));
        }
        
        function toggleCamera() {
            const btn = document.getElementById('cameraBtn');
            const container = document.getElementById('cameraContainer');
            
            if (!cameraActive) {
                socket.emit('start_camera');
                btn.textContent = '📷 Stop Camera';
                container.style.display = 'block';
                cameraActive = true;
            } else {
                socket.emit('stop_camera');
                btn.textContent = '📷 Start Camera';
                container.style.display = 'none';
                cameraActive = false;
            }
        }
        
        function displayEmotionResult(data) {
            const container = document.getElementById('emotionResults');
            const result = document.createElement('div');
            result.className = 'emotion-result';
            result.innerHTML = `
                <strong>${data.emotion}</strong> (${(data.confidence * 100).toFixed(1)}%)
                <br><small>${data.modality} • ${new Date(data.timestamp * 1000).toLocaleTimeString()}</small>
            `;
            container.insertBefore(result, container.firstChild);
            
            // Keep only last 5 results
            while (container.children.length > 5) {
                container.removeChild(container.lastChild);
            }
        }
        
        function updateConsciousness() {
            fetch('/api/consciousness/state')
            .then(response => response.json())
            .then(data => {
                document.getElementById('selfAwareness').textContent = data.self_awareness.attention;
                document.getElementById('curiosityLevel').textContent = data.curiosity.curiosity_level.toFixed(2);
                document.getElementById('mood').textContent = data.physiology.summary.mood;
                document.getElementById('arousal').textContent = data.physiology.summary.arousal;
            });
            
            fetch('/api/consciousness/questions')
            .then(response => response.json())
            .then(questions => {
                const container = document.getElementById('consciousnessQuestions');
                container.innerHTML = '';
                questions.forEach(q => {
                    const div = document.createElement('div');
                    div.className = 'question-item';
                    div.innerHTML = `${q.content} <small>(${q.importance.toFixed(2)})</small>`;
                    container.appendChild(div);
                });
            });
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Display user message
            addChatMessage(message, 'user');
            input.value = '';
            
            // Send to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                addChatMessage(data.response, 'ai');
                updateConsciousness();
            })
            .catch(error => {
                addChatMessage('Sorry, I encountered an error processing your message.', 'ai');
            });
        }
        
        function addChatMessage(message, sender) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `message ${sender}-message`;
            div.textContent = message;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function handleChatKeypress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Initialize
        updateConsciousness();
        setInterval(updateConsciousness, 10000); // Update every 10 seconds
    </script>
</body>
</html>'''
        
        with open(template_dir / 'human2_interface.html', 'w') as f:
            f.write(html_content)
        
        print("✅ HTML template created")
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the web interface"""
        if not FLASK_AVAILABLE:
            print("❌ Flask not available. Install with: pip install flask flask-socketio")
            return False
        
        if not self.initialize_systems():
            print("❌ Failed to initialize HUMAN 2.0 systems")
            return False
        
        # Create HTML template
        self.create_html_template()
        
        print(f"🚀 Starting HUMAN 2.0 Web Interface...")
        print(f"📱 Access at: http://{host}:{port}")
        print(f"🎉 Features: Real-time emotion recognition, consciousness visualization, AI chat")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            print(f"❌ Failed to start web interface: {e}")
            return False

def main():
    """Main function"""
    print("🌐 HUMAN 2.0 PRODUCTION WEB INTERFACE")
    print("=" * 50)
    
    interface = HUMAN2WebInterface()
    interface.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main() 