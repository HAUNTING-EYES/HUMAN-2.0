from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_prepare_dataset(
    cache_dir: str = "data/cache",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Download and prepare the GoEmotions dataset"""
    
    logger.info("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", cache_dir=cache_dir)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    
    # Get label names
    label_names = dataset["train"].features["labels"].feature.names
    
    # Convert multi-hot labels to single label (most prominent emotion)
    def get_primary_emotion(labels):
        if len(labels) == 0:
            return -1  # No emotion
        return labels[0]  # Take first emotion as primary
    
    train_df['primary_emotion'] = train_df['labels'].apply(get_primary_emotion)
    test_df['primary_emotion'] = test_df['labels'].apply(get_primary_emotion)
    
    # Remove samples with no emotion
    train_df = train_df[train_df['primary_emotion'] != -1]
    test_df = test_df[test_df['primary_emotion'] != -1]
    
    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df['primary_emotion']
    )
    
    # Save processed data
    train_df.to_parquet(output_path / "train.parquet")
    val_df.to_parquet(output_path / "val.parquet")
    test_df.to_parquet(output_path / "test.parquet")
    
    # Save label mapping
    with open(output_path / "label_mapping.json", "w") as f:
        json.dump(
            {
                "id2label": {str(i): label for i, label in enumerate(label_names)},
                "label2id": {label: str(i) for i, label in enumerate(label_names)}
            },
            f,
            indent=2
        )
    
    # Log statistics
    logger.info("Dataset statistics:")
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Log class distribution
    logger.info("\nClass distribution in training set:")
    train_dist = train_df['primary_emotion'].value_counts()
    for emotion_id, count in train_dist.items():
        logger.info(f"{label_names[emotion_id]}: {count}")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

def get_class_weights(labels: List[int], num_classes: int = 28) -> np.ndarray:
    """Calculate class weights for imbalanced dataset"""
    
    # Count occurrences of each class
    class_counts = np.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1
    
    # Add small constant to avoid division by zero
    class_counts = class_counts + 1e-6
    
    # Calculate weights
    total_samples = sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    return class_weights

if __name__ == "__main__":
    # Download and prepare dataset
    datasets = download_and_prepare_dataset()
    
    # Calculate and save class weights
    class_weights = get_class_weights(datasets["train"]["primary_emotion"])
    np.save("data/processed/class_weights.npy", class_weights)
    
    logger.info("\nDataset preparation completed!") 