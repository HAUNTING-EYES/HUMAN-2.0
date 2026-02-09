import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json
import os
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import shutil
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def create_backup(data, epoch, backup_dir='backups'):
    """Create a backup of the training data for the current epoch."""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(backup_dir, f'train_data_epoch_{epoch}_{timestamp}.parquet')
    
    if isinstance(data, pd.DataFrame):
        data.to_parquet(backup_file)
    else:
        pd.DataFrame(data).to_parquet(backup_file)
    
    logging.info(f"Created backup at {backup_file}")
    return backup_file

def load_data(data_path: str) -> pd.DataFrame:
    """Load training data from parquet file"""
    return pd.read_parquet(data_path)

def load_label_mapping() -> Dict[str, Dict[str, str]]:
    """Load label mapping"""
    with open("data/processed/label_mapping.json", "r") as f:
        return json.load(f)

def prepare_dataset(df: pd.DataFrame, num_labels: int) -> Dict[str, Dataset]:
    """Prepare dataset for training"""
    # Split data into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Ensure labels are in correct format
    def format_labels(labels):
        # Initialize zero array
        label_array = np.zeros(num_labels)
        
        if isinstance(labels, (int, float)):
            # Single label case
            label_array[int(labels)] = 1
        elif isinstance(labels, list):
            # Multi-label case as list
            for label in labels:
                label_array[int(label)] = 1
        elif isinstance(labels, np.ndarray):
            # Multi-label case as array
            if len(labels.shape) == 1:
                # Array of label indices
                for label in labels:
                    label_array[int(label)] = 1
            else:
                # Already one-hot encoded
                return labels.tolist()
        else:
            raise ValueError(f"Unexpected label format: {type(labels)}")
            
        return label_array.tolist()
    
    # Format labels for each dataset
    for df in [train_df, val_df, test_df]:
        df['labels'] = df['labels'].apply(format_labels)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    labels = np.array(labels)
    # Convert predictions to binary (0 or 1) using threshold
    pred_bin = (predictions > 0.5).astype(int)

    # Print a batch for inspection
    print("Sample predictions (first 5):", pred_bin[:5])
    print("Sample labels (first 5):", labels[:5])
    print("Label distribution (sum per class):", labels.sum(axis=0))

    # Multi-label metrics
    accuracy = accuracy_score(labels, pred_bin)
    f1 = f1_score(labels, pred_bin, average='samples', zero_division=0)
    precision = precision_score(labels, pred_bin, average='samples', zero_division=0)
    recall = recall_score(labels, pred_bin, average='samples', zero_division=0)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model():
    # Load data and label mapping
    print("Loading data...")
    df = load_data("data/processed/train.parquet")
    label_mapping = load_label_mapping()
    num_labels = len(label_mapping["id2label"])
    
    # Create initial backup of training data
    backup_dir = "backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    initial_backup = os.path.join(backup_dir, f'train_data_initial_{timestamp}.parquet')
    df.to_parquet(initial_backup)
    logging.info(f"Created initial backup of training data at {initial_backup}")
    
    # Prepare datasets
    print("Preparing datasets...")
    datasets = prepare_dataset(df, num_labels)
    
    # Initialize tokenizer and model
    print("Initializing model...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=label_mapping["id2label"],
        label2id=label_mapping["label2id"]
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Tokenization function
    def tokenize_function(examples):
        """Tokenize text and format for multi-label classification"""
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        for split, dataset in datasets.items()
    }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/eradem_retrained",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Create backup of the final model
    model_backup_dir = "model_backups"
    if not os.path.exists(model_backup_dir):
        os.makedirs(model_backup_dir)
    final_model_backup = os.path.join(model_backup_dir, f'final_model_{timestamp}.pt')
    trainer.save_model(final_model_backup)
    logging.info(f"Created backup of final model at {final_model_backup}")
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"Validation Precision: {eval_results['eval_precision']:.4f}")
    print(f"Validation Recall: {eval_results['eval_recall']:.4f}")
    
    # Test model
    print("\nTesting model...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\nTest Results:")
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1: {test_results['eval_f1']:.4f}")
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall: {test_results['eval_recall']:.4f}")

    # Advanced evaluation and visualization
    print("\nAdvanced Evaluation and Visualization:")
    # Get predictions and labels for the test set
    predictions = trainer.predict(tokenized_datasets["test"])
    pred_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
    pred_bin = (pred_probs > 0.5).astype(int)
    labels = np.array(predictions.label_ids)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, pred_bin, zero_division=0))

    # Confusion matrix (multi-label)
    mcm = multilabel_confusion_matrix(labels, pred_bin)
    fig, axes = plt.subplots(nrows=1, ncols=min(5, mcm.shape[0]), figsize=(15, 3))
    for i, ax in enumerate(axes):
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Class {i}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.show()

    # Label distribution plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(labels.shape[1]), labels.sum(axis=0))
    plt.title("True Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    plt.bar(range(pred_bin.shape[1]), pred_bin.sum(axis=0))
    plt.title("Predicted Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Qualitative examples
    print("\nQualitative Examples:")
    test_df = datasets["test"].to_pandas()
    for idx in np.random.choice(len(test_df), size=5, replace=False):
        print(f"Text: {test_df.iloc[idx]['text']}")
        print(f"True labels: {np.where(labels[idx]==1)[0]}")
        print(f"Predicted labels: {np.where(pred_bin[idx]==1)[0]}")
        print("-")
    
    # Save model
    print("\nSaving model...")
    model_path = "models/final_model"
    
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save model configuration
    config = model.config
    config.save_pretrained(model_path)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer.save_pretrained(model_path)
    
    # Save label mappings
    with open(os.path.join(model_path, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model() 