import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import random

# Try to use DirectML if available
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Model definition (must match training and real-time)
class AudioCNN(nn.Module):
    def __init__(self, n_mfcc, n_classes, n_frames=None):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        if n_frames is None:
            conv_output_size = (n_mfcc // 8) * 128
        else:
            conv_output_size = (n_frames // 8) * (n_mfcc // 8) * 128
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        self.n_frames = n_frames
    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def main():
    # Load model config
    with open('models/expanded_model_config.json', 'r') as f:
        model_config = json.load(f)
    # Load label encoder
    with open('models/expanded_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    # Load model
    model = AudioCNN(
        n_mfcc=model_config['n_mfcc'],
        n_classes=model_config['n_classes'],
        n_frames=model_config['n_frames']
    ).to(device)
    model.load_state_dict(torch.load('models/audio_emotion_expanded_best_model.pth', map_location=device))
    model.eval()
    # Load combined dataset
    with open('data/audio_features/combined_dataset.pkl', 'rb') as f:
        combined_data = pickle.load(f)
    # Pick 20 random test samples
    random.seed(42)
    samples = random.sample(combined_data, 20)
    print("\nTesting model on 20 random samples from the combined dataset:")
    for i, sample in enumerate(samples):
        mfcc = np.array(sample['mfcc'])
        label = sample['label']
        # Preprocess: pad/truncate to match model input
        target_frames = model_config['n_frames']
        if mfcc.shape[0] < target_frames:
            padding = np.zeros((target_frames - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.concatenate([mfcc, padding], axis=0)
        else:
            mfcc = mfcc[:target_frames, :]
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        pred_emotion = label_encoder.classes_[predicted_class]
        print(f"Sample {i+1:2d}: True={label:12s} | Pred={pred_emotion:12s} | Conf={confidence:.2f} | Probs={probabilities.cpu().numpy()[0]}")

if __name__ == "__main__":
    main() 