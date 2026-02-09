import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import pickle
from collections import Counter
import pandas as pd

class RobustEmotionModel(nn.Module):
    def __init__(self, num_classes=7, input_size=128):
        super(RobustEmotionModel, self).__init__()
        
        # Feature extraction layers
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
        
        # Classification head
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

def extract_audio_features(audio_data, sample_rate=16000, input_size=128):
    """Extract audio features matching the real-time script"""
    try:
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            window='hann'
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract additional features
        features = extract_advanced_features(audio_data, sample_rate)
        
        # Combine features
        combined_features = np.concatenate([
            mel_spec_db.flatten(),
            features
        ])
        
        # Ensure correct size
        if len(combined_features) > input_size:
            combined_features = combined_features[:input_size]
        elif len(combined_features) < input_size:
            combined_features = np.pad(combined_features, 
                                     (0, input_size - len(combined_features)))
        
        return combined_features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(input_size)

def extract_advanced_features(audio_data, sample_rate):
    """Extract advanced audio features"""
    features = []
    
    try:
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        ).mean()
        features.append(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate
        ).mean()
        features.append(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=sample_rate
        ).mean()
        features.append(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data).mean()
        features.append(zcr)
        
        # MFCCs (first 5 coefficients)
        mfccs = librosa.feature.mfcc(
            y=audio_data, sr=sample_rate, n_mfcc=13
        )[:5].flatten()
        features.extend(mfccs)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_mean = np.mean(pitches[magnitudes > 0.1])
        pitch_std = np.std(pitches[magnitudes > 0.1])
        features.extend([pitch_mean, pitch_std])
        
        # Energy features
        rms = librosa.feature.rms(y=audio_data).mean()
        features.append(rms)
        
        # Harmonic features
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive)) + 1e-8)
        features.append(harmonic_ratio)
        
    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        features = [0.0] * 15
    
    return np.array(features)

def test_model_on_data():
    """Test the model on test data to verify it's working correctly"""
    
    # Load model and config
    try:
        with open('models/robust_model_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize model with correct parameters
        model = RobustEmotionModel(
            num_classes=config['n_classes'],
            input_size=128  # Fixed input size
        )
        
        # Load trained weights
        model_path = 'models/robust_emotion_model.pth'
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Load label encoder
        with open('models/balanced_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        emotion_labels = label_encoder.classes_
        print(f"Model loaded successfully. Classes: {emotion_labels}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    try:
        test_data = pd.read_parquet('data/processed/balanced_test.parquet')
        print(f"Loaded test data: {len(test_data)} samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Test predictions
    predictions = []
    confidences = []
    correct = 0
    total = 0
    
    print("\nTesting model predictions...")
    
    for idx, row in test_data.head(50).iterrows():  # Test first 50 samples
        try:
            # Load audio file
            audio_path = row['file_path']
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            features = extract_audio_features(audio_data, sr, 128)  # Fixed input size
            
            # Predict
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_emotion = emotion_labels[predicted_idx.item()]
            confidence_value = confidence.item()
            
            # Get true label
            true_emotion = row['emotion']
            
            # Check if correct
            is_correct = predicted_emotion == true_emotion
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append(predicted_emotion)
            confidences.append(confidence_value)
            
            print(f"Sample {idx}: True={true_emotion}, Predicted={predicted_emotion}, "
                  f"Confidence={confidence_value:.3f}, Correct={is_correct}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Print results
    accuracy = correct / total if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Show prediction distribution
    pred_counts = Counter(predictions)
    print(f"\nPrediction Distribution:")
    for emotion, count in pred_counts.most_common():
        print(f"  {emotion}: {count}")
    
    # Check for bias
    print(f"\nBias Analysis:")
    most_common_pred = pred_counts.most_common(1)[0] if pred_counts else None
    if most_common_pred:
        emotion, count = most_common_pred
        percentage = count / len(predictions) * 100
        print(f"  Most predicted emotion: {emotion} ({percentage:.1f}%)")
        
        if percentage > 50:
            print(f"  WARNING: Model shows bias towards {emotion}")
        else:
            print(f"  Model predictions look balanced")

if __name__ == "__main__":
    test_model_on_data() 