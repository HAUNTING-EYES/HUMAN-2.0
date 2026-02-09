import argparse
import logging
from pathlib import Path
from models.emotion_recognition import EmotionRecognitionSystem
import json
from typing import Dict, List, Optional
import torch
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionRecognitionPipeline:
    """Pipeline for running emotion recognition"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
        max_sequence_length: int = 128
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.system = EmotionRecognitionSystem(
            model_path=model_path,
            device=device
        )
        
        # Load label mapping
        label_map_path = Path(model_path).parent / "label_mapping.json"
        with open(label_map_path, "r") as f:
            self.label_mapping = json.load(f)
            
        logger.info("Pipeline initialized!")
        
    def process_text(
        self,
        text: str,
        return_probabilities: bool = False,
        threshold: float = 0.3
    ) -> Dict[str, float]:
        """Process a single text input"""
        
        # Get predictions
        start_time = time.time()
        emotions = self.system.predict(text)
        inference_time = time.time() - start_time
        
        # Filter and format results
        if not return_probabilities:
            emotions = {k: v for k, v in emotions.items() if v >= threshold}
            
        return {
            "emotions": emotions,
            "inference_time": inference_time
        }
        
    def process_batch(
        self,
        texts: List[str],
        return_probabilities: bool = False,
        threshold: float = 0.3
    ) -> List[Dict[str, float]]:
        """Process a batch of texts"""
        
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            
            # Get predictions
            start_time = time.time()
            batch_emotions = self.system.predict(batch_texts)
            inference_time = time.time() - start_time
            
            # Process each prediction
            for emotions in batch_emotions:
                if not return_probabilities:
                    emotions = {k: v for k, v in emotions.items() if v >= threshold}
                    
                results.append({
                    "emotions": emotions,
                    "inference_time": inference_time / len(batch_texts)
                })
                
        return results
        
    def get_emotion_embedding(self, text: str) -> Dict[str, torch.Tensor]:
        """Get emotion embedding for text"""
        
        # Get embedding
        start_time = time.time()
        embedding = self.system.get_embedding(text)
        inference_time = time.time() - start_time
        
        return {
            "embedding": embedding,
            "inference_time": inference_time
        }

def main():
    parser = argparse.ArgumentParser(description="Run emotion recognition")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--return_probabilities", action="store_true")
    parser.add_argument("--get_embeddings", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EmotionRecognitionPipeline(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size
    )
    
    # Process input
    if args.input_text:
        if args.get_embeddings:
            result = pipeline.get_emotion_embedding(args.input_text)
            print(f"Embedding shape: {result['embedding'].shape}")
            print(f"Inference time: {result['inference_time']:.3f}s")
        else:
            result = pipeline.process_text(
                args.input_text,
                return_probabilities=args.return_probabilities,
                threshold=args.threshold
            )
            print(f"Detected emotions: {result['emotions']}")
            print(f"Inference time: {result['inference_time']:.3f}s")
            
    elif args.input_file:
        # Read input file
        with open(args.input_file, "r") as f:
            texts = [line.strip() for line in f]
            
        # Process texts
        results = pipeline.process_batch(
            texts,
            return_probabilities=args.return_probabilities,
            threshold=args.threshold
        )
        
        # Save results
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))
            
    else:
        logger.error("Either --input_text or --input_file must be provided")

if __name__ == "__main__":
    main() 