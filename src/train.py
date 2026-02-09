import argparse
import logging
import numpy as np
from pathlib import Path
from models.emotion_recognition import EmotionRecognitionSystem, EmotionDataset
import torch
import wandb
from typing import Dict, List
import json
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare GoEmotions dataset"""
    logger.info("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified")
    
    # Initialize multi-label binarizer
    mlb = MultiLabelBinarizer()
    
    # Prepare datasets
    train_labels = mlb.fit_transform(dataset["train"]["labels"])
    val_labels = mlb.transform(dataset["validation"]["labels"])
    test_labels = mlb.transform(dataset["test"]["labels"])
    
    return {
        "train": {
            "texts": dataset["train"]["text"],
            "labels": train_labels
        },
        "val": {
            "texts": dataset["validation"]["text"],
            "labels": val_labels
        },
        "test": {
            "texts": dataset["test"]["text"],
            "labels": test_labels
        }
    }, mlb.classes_

def train_model(
    model_dir: str = "models",
    batch_size: int = 16,
    max_epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    model_name: str = "roberta-base",
    hub_token: str = None,
    hub_model_id: str = None,
    use_wandb: bool = True,
    resume_from_checkpoint: str = None
):
    """Train the emotion recognition model"""
    
    # Load data
    logger.info("Loading data...")
    data, emotion_labels = load_data()
    
    # Load hierarchy info
    with open("data/processed/hierarchy_info.json", "r") as f:
        hierarchy_info = json.load(f)
    
    # Create datasets
    train_dataset = EmotionDataset(
        texts=data["train"]["texts"],
        labels=data["train"]["labels"]
    )
    
    val_dataset = EmotionDataset(
        texts=data["val"]["texts"],
        labels=data["val"]["labels"]
    )
    
    test_dataset = EmotionDataset(
        texts=data["test"]["texts"],
        labels=data["test"]["labels"]
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = EmotionRecognitionSystem(
        model_name=model_name,
        num_labels=len(emotion_labels),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_wandb=use_wandb
    )
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            filename="emotion-recognition-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        ),
        TQDMProgressBar(refresh_rate=1)
    ]
    
    if use_wandb:
        wandb_logger = WandbLogger(project="emotion_recognition")
        trainer_logger = wandb_logger
    else:
        trainer_logger = True
    
    # Find latest checkpoint if resume_from_checkpoint is not provided
    ckpt_path = None
    if resume_from_checkpoint:
        ckpt_path = resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        ckpt_files = sorted(glob.glob(f"{model_dir}/emotion-recognition-epoch=*.ckpt"))
        if ckpt_files:
            ckpt_path = ckpt_files[-1]
            logger.info(f"Auto-resuming from latest checkpoint: {ckpt_path}")
        else:
            logger.info("No checkpoint found, starting training from scratch.")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=trainer_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4
        ),
        ckpt_path=ckpt_path
    )
    
    # Test model
    logger.info("Testing model...")
    test_results = trainer.test(
        model,
        dataloaders=DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=4
        )
    )
    
    # Save model
    logger.info("Saving model...")
    model.save_pretrained(model_dir)
    
    # Save test results
    with open(os.path.join(model_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Training completed!")
    return model, test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=16)  # Default batch size reduced
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train_model(**vars(args)) 