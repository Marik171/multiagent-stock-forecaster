#!/usr/bin/env python3
"""
fine_tune_distilbert_news.py

Fine‐tunes Hugging Face's distilbert-base-uncased model on NVDA news headlines (positive/neutral/negative).
Input CSV: /data/annotated/NVDA_daily_news_annotated.csv (columns: datetime, title, source, sentiment_score, sentiment_label).
Output: a saved DistilBERT classifier under ./distilbert_news_{TICKER}/ that, at inference, returns softmax
probabilities over {NEGATIVE, NEUTRAL, POSITIVE}. Also provides an inference helper at the bottom.
"""

import os
import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

# === Configure Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === Check Device ===
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
logger.info(f"Using device: {device}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine‐tune DistilBERT on NVDA news headlines for 3‐way sentiment classification"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker prefix for input CSV (e.g., NVDA). Expects /data/annotated/{TICKER}_daily_news_annotated.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the fine‐tuned model & tokenizer. Defaults to ./distilbert_news_{TICKER}/"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    return parser.parse_args()

class SentimentDataset(Dataset):
    """Custom PyTorch dataset for sentiment classification."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        
        return item

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def load_model_for_inference(model_dir: str):
    """
    Load the fine‐tuned DistilBERT classifier from model_dir.
    Returns (tokenizer, model_on_device).
    """
    config_inf = AutoConfig.from_pretrained(model_dir)
    tokenizer_inf = AutoTokenizer.from_pretrained(model_dir)
    model_inf = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config_inf
    )
    model_inf.to(device)
    model_inf.eval()
    return tokenizer_inf, model_inf

def predict_sentiment(text: str, tokenizer, model):
    """
    Given a headline string, return a dict of softmax probabilities:
      {"NEGATIVE": x, "NEUTRAL": y, "POSITIVE": z}
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1,3)
        probs = F.softmax(logits, dim=-1).squeeze().tolist()  # [p_neg, p_neu, p_pos]

    id2label_inf = model.config.id2label
    return {id2label_inf[i]: probs[i] for i in range(len(probs))}


def train_model(model, train_loader, val_loader, optimizer, num_epochs, output_dir):
    """
    Train the model using the provided data loaders.
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # For tracking best model
    best_val_loss = float('inf')
    best_val_metrics = None
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        train_start = time.time()
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            loss = criterion(outputs.logits, batch['labels'])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            
            # Print progress every 50 batches
            if train_steps % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Batch {train_steps} - Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / train_steps
        train_time = time.time() - train_start
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_train_loss:.4f} - Time: {train_time:.2f}s")
        
        # Evaluation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        val_start = time.time()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                loss = criterion(outputs.logits, batch['labels'])
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                total_val_loss += loss.item()
                val_steps += 1
        
        # Calculate metrics
        avg_val_loss = total_val_loss / val_steps
        val_metrics = compute_metrics(all_labels, all_preds)
        val_time = time.time() - val_start
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {avg_val_loss:.4f} - Accuracy: {val_metrics['accuracy']:.4f} - F1: {val_metrics['f1']:.4f} - Time: {val_time:.2f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = val_metrics
            
            # Save model and tokenizer
            logger.info(f"New best model found! Saving to {output_dir}")
            model.save_pretrained(output_dir)
            
    logger.info(f"Training complete. Best validation metrics: {best_val_metrics}")
    return best_val_metrics

if __name__ == "__main__":
    # === Parse Arguments ===
    args = parse_args()
    TICKER = args.ticker.upper()
    OUTPUT_DIR = args.output_dir or f"./distilbert_news_{TICKER}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # === Load & Inspect CSV ===
    csv_path = f"/data/annotated/{TICKER}_daily_news_annotated.csv"
    # Check if the path exists, otherwise try relative path
    if not os.path.exists(csv_path):
        csv_path = f"/Users/marik/Desktop/multiagent_stock_forecaster/data/annotated/{TICKER}_daily_news_annotated.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info("First 3 rows of the data:\n" + df.head(3).to_string(index=False))
    logger.info(f"Columns available: {df.columns.tolist()}")
    
    # === Preprocess Dataset ===
    # Rename title column to text
    df = df.rename(columns={"title": "text"})
    
    # Drop rows where text or sentiment_label is missing
    df = df.dropna(subset=["text", "sentiment_label"])
    
    # Select only relevant columns
    df = df[["text", "sentiment_label"]]
    
    # Log label distribution
    logger.info("Label distribution:\n" + df["sentiment_label"].value_counts().to_string())
    
    # === Encode Labels & Split ===
    # Create a mapping from string labels to integers
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)  # 3
    
    # Map sentiment_label to a new integer column label
    df["label"] = df["sentiment_label"].map(label2id)
    
    # Perform a stratified train/validation split (15% validation)
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label"]
    )
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    logger.info("Train label counts:\n" + train_df["label"].map(id2label).value_counts().to_string())
    logger.info("Val   label counts:\n" + val_df["label"].map(id2label).value_counts().to_string())
    
    # === Load Tokenizer & Model ===
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Save tokenizer immediately (we'll need it for inference later)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )
    model.to(device)
    
    # === Create PyTorch Datasets and DataLoaders ===
    train_dataset = SentimentDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer
    )
    
    val_dataset = SentimentDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # === Setup Optimizer ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # === Train Model ===
    try:
        best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            output_dir=OUTPUT_DIR
        )
        
        # Save final metrics
        import json
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(best_metrics, f)
            
        # Generate confusion matrix for best model
        model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
        model.to(device)
        model.eval()
        
        # Get predictions
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Log confusion matrix
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save confusion matrix
        np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # === Inference Example ===
    try:
        tokenizer_inf, model_inf = load_model_for_inference(OUTPUT_DIR)
        example_text = "NVDA shares soar after strong quarterly earnings"
        probs = predict_sentiment(example_text, tokenizer_inf, model_inf)
        logger.info(f"Sentiment probabilities for '{example_text}': {probs}")
        
        example_text2 = "NVDA stock plummets after disappointing quarterly results"
        probs2 = predict_sentiment(example_text2, tokenizer_inf, model_inf)
        logger.info(f"Sentiment probabilities for '{example_text2}': {probs2}")
        
        example_text3 = "NVDA announces new graphics card lineup, analysts remain cautious"
        probs3 = predict_sentiment(example_text3, tokenizer_inf, model_inf)
        logger.info(f"Sentiment probabilities for '{example_text3}': {probs3}")
    except Exception as e:
        logger.error(f"Inference example failed: {e}")
