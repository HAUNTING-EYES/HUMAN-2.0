import logging
from pathlib import Path
from models.emotion_recognition import EmotionRecognitionSystem
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_emotion_recognition():
    """Test the emotion recognition system"""
    
    # Initialize model
    logger.info("Initializing model...")
    system = EmotionRecognitionSystem(
        model_name="roberta-base",
        device="cpu"
    )
    
    # Test texts
    test_texts = [
        "I'm so happy about this!",
        "This makes me really angry.",
        "I'm feeling quite sad today.",
        "Wow, this is amazing!",
        "I'm really worried about tomorrow."
    ]
    
    # Test prediction
    logger.info("\nTesting predictions:")
    for text in test_texts:
        emotions = system.predict(text)
        logger.info(f"\nText: {text}")
        logger.info(f"Detected emotions: {emotions}")
        
        # Get embedding
        embedding = system.get_embedding(text)
        logger.info(f"Embedding shape: {embedding.shape}")

if __name__ == "__main__":
    test_emotion_recognition() 