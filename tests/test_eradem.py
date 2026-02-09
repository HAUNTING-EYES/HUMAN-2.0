import unittest
import torch
from datetime import datetime
from src.components.emotional_memory import ERADEM

class TestERADEM(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.eradem = ERADEM(hidden_size=64)  # Use smaller hidden size for testing
        
    def test_initialization(self):
        """Test ERADEM initialization."""
        # Check emotion dimensions
        self.assertIn('valence', self.eradem.emotion_dimensions)
        self.assertIn('arousal', self.eradem.emotion_dimensions)
        self.assertIn('dominance', self.eradem.emotion_dimensions)
        
        # Check basic emotions
        self.assertIn('joy', self.eradem.basic_emotions)
        self.assertIn('sadness', self.eradem.basic_emotions)
        self.assertIn('anger', self.eradem.basic_emotions)
        self.assertIn('fear', self.eradem.basic_emotions)
        self.assertIn('surprise', self.eradem.basic_emotions)
        self.assertIn('disgust', self.eradem.basic_emotions)
        
        # Check model components
        self.assertIsInstance(self.eradem.emotion_embedding, torch.nn.Linear)
        self.assertIsInstance(self.eradem.emotion_attention, torch.nn.MultiheadAttention)
        
    def test_emotional_embedding(self):
        """Test emotional embedding computation."""
        # Test with joy emotion
        joy_vector = self.eradem.basic_emotions['joy']
        embedding = self.eradem.compute_emotional_embedding(joy_vector)
        
        # Check embedding shape
        self.assertEqual(embedding.shape[0], self.eradem.hidden_size)
        
        # Test with custom emotion vector
        custom_vector = {'valence': 0.5, 'arousal': 0.3, 'dominance': 0.4}
        embedding = self.eradem.compute_emotional_embedding(custom_vector)
        self.assertEqual(embedding.shape[0], self.eradem.hidden_size)
        
    def test_emotional_attention(self):
        """Test emotional attention mechanism."""
        # Create test embeddings
        emotion_embedding = self.eradem.compute_emotional_embedding(
            self.eradem.basic_emotions['joy']
        )
        context_embeddings = torch.randn(5, self.eradem.hidden_size)  # 5 tokens
        
        # Apply attention
        attended_embeddings = self.eradem.apply_emotional_attention(
            emotion_embedding,
            context_embeddings
        )
        
        # Check output shape
        self.assertEqual(attended_embeddings.shape[0], 1)  # Batch size 1
        self.assertEqual(attended_embeddings.shape[1], self.eradem.hidden_size)
        
    def test_emotion_diffusion(self):
        """Test emotion diffusion mechanism."""
        # Test with joy and sadness
        current_emotion = self.eradem.basic_emotions['joy']
        context_emotion = self.eradem.basic_emotions['sadness']
        
        diffused = self.eradem.diffuse_emotion(current_emotion, context_emotion)
        
        # Check dimensions
        self.assertIn('valence', diffused)
        self.assertIn('arousal', diffused)
        self.assertIn('dominance', diffused)
        
        # Check value ranges
        for dim, value in diffused.items():
            self.assertGreaterEqual(value, self.eradem.emotion_dimensions[dim]['min'])
            self.assertLessEqual(value, self.eradem.emotion_dimensions[dim]['max'])
            
    def test_emotional_memory(self):
        """Test emotional memory management."""
        # Add some memories
        emotions = ['joy', 'sadness', 'anger']
        contexts = ['happy moment', 'sad situation', 'angry encounter']
        
        for emotion, context in zip(emotions, contexts):
            self.eradem.update_emotional_memory(
                self.eradem.basic_emotions[emotion],
                context,
                datetime.now()
            )
            
        # Check memory size
        self.assertEqual(len(self.eradem.emotion_memory), 3)
        
        # Check memory structure
        memory = self.eradem.emotion_memory[0]
        self.assertIn('emotion', memory)
        self.assertIn('context', memory)
        self.assertIn('timestamp', memory)
        self.assertIn('importance', memory)
        
        # Test memory pruning
        for _ in range(1000):  # Add more memories to trigger pruning
            self.eradem.update_emotional_memory(
                self.eradem.basic_emotions['joy'],
                'test context',
                datetime.now()
            )
            
        # Check that pruning occurred
        self.assertLess(len(self.eradem.emotion_memory), 1000)
        
    def test_emotional_context(self):
        """Test emotional context retrieval."""
        # Add some memories with different emotions
        emotions = ['joy', 'sadness', 'anger']
        contexts = ['happy moment', 'sad situation', 'angry encounter']
        
        for emotion, context in zip(emotions, contexts):
            self.eradem.update_emotional_memory(
                self.eradem.basic_emotions[emotion],
                context,
                datetime.now()
            )
            
        # Get emotional context
        context = self.eradem.get_emotional_context(datetime.now())
        
        # Check context structure
        self.assertIn('valence', context)
        self.assertIn('arousal', context)
        self.assertIn('dominance', context)
        
        # Check value ranges
        for dim, value in context.items():
            self.assertGreaterEqual(value, self.eradem.emotion_dimensions[dim]['min'])
            self.assertLessEqual(value, self.eradem.emotion_dimensions[dim]['max'])
            
if __name__ == '__main__':
    unittest.main() 