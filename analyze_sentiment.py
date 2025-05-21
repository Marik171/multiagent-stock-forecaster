#!/usr/bin/env python3
"""
Real-time sentiment analysis on news headlines using the fine-tuned DistilBERT model.
This script can be used to analyze new headlines or batch analyze a CSV file.
"""

import os
import argparse
import logging
import pandas as pd
import torch
import torch.nn.functional as F
import time
from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# === Configure Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Check device
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
        description="Analyze sentiment of financial news headlines using a fine-tuned DistilBERT model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./distilbert_news_NVDA/",
        help="Directory containing the fine-tuned model"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a CSV file containing headlines to analyze (must have 'title' column)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the results CSV file (defaults to input_file with _sentiment suffix)"
    )
    parser.add_argument(
        "--headline",
        type=str,
        help="Single headline to analyze (alternative to input_file)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol (for record keeping)"
    )
    return parser.parse_args()

def load_model(model_dir):
    """
    Load the fine-tuned sentiment analysis model.
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        # Load config, tokenizer, and model
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config
        )
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def analyze_sentiment(headline, tokenizer, model):
    """
    Analyze the sentiment of a financial news headline.
    
    Args:
        headline: The headline text to analyze
        tokenizer: The tokenizer for preprocessing
        model: The sentiment analysis model
        
    Returns:
        Dictionary with sentiment probabilities and predicted label
    """
    # Tokenize input
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    # Map to labels
    id2label = model.config.id2label
    probabilities = {id2label[i]: prob for i, prob in enumerate(probs)}
    
    # Get predicted sentiment
    predicted_sentiment = max(probabilities.items(), key=lambda x: x[1])[0]
    confidence = max(probabilities.values())
    
    return {
        "headline": headline,
        "predicted_sentiment": predicted_sentiment,
        "confidence": confidence,
        "probabilities": probabilities
    }

def batch_analyze_headlines(headlines, tokenizer, model, batch_size=32):
    """
    Analyze sentiment for a batch of headlines for better efficiency.
    
    Args:
        headlines: List of headlines to analyze
        tokenizer: The tokenizer for preprocessing
        model: The sentiment analysis model
        batch_size: Batch size for processing
        
    Returns:
        List of sentiment analysis results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        
        # Process each item in the batch
        id2label = model.config.id2label
        for j, headline in enumerate(batch):
            headline_probs = probs[j].tolist()
            probabilities = {id2label[k]: prob for k, prob in enumerate(headline_probs)}
            predicted_sentiment = max(probabilities.items(), key=lambda x: x[1])[0]
            confidence = max(probabilities.values())
            
            results.append({
                "headline": headline,
                "predicted_sentiment": predicted_sentiment,
                "confidence": confidence,
                "negative_prob": probabilities["NEGATIVE"],
                "neutral_prob": probabilities["NEUTRAL"],
                "positive_prob": probabilities["POSITIVE"]
            })
    
    return results

def process_csv_file(input_file, output_file, tokenizer, model):
    """
    Process a CSV file containing headlines and save results.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        tokenizer: The tokenizer for preprocessing
        model: The sentiment analysis model
    """
    try:
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Check if title column exists
        if 'title' not in df.columns:
            raise ValueError("Input CSV must contain a 'title' column")
        
        # Get headlines
        headlines = df['title'].tolist()
        
        # Analyze sentiment
        start_time = time.time()
        results = batch_analyze_headlines(headlines, tokenizer, model)
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {len(headlines)} headlines in {elapsed_time:.2f} seconds " + 
                   f"({len(headlines)/elapsed_time:.2f} headlines/sec)")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Merge with original dataframe
        output_df = pd.concat([df, results_df.drop('headline', axis=1)], axis=1)
        
        # Save to file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Log sentiment distribution
        sentiment_counts = output_df['predicted_sentiment'].value_counts()
        logger.info(f"Sentiment distribution:\n{sentiment_counts.to_string()}")
        
        return output_df
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    
    # Load the model
    logger.info(f"Loading model from {args.model_dir}")
    tokenizer, model = load_model(args.model_dir)
    
    if args.headline:
        # Single headline mode
        logger.info(f"Analyzing headline: {args.headline}")
        result = analyze_sentiment(args.headline, tokenizer, model)
        logger.info(f"Predicted sentiment: {result['predicted_sentiment']} (confidence: {result['confidence']:.4f})")
        logger.info(f"Probabilities: {result['probabilities']}")
    
    elif args.input_file:
        # Batch mode with CSV file
        if not args.output_file:
            # Generate default output filename
            base, ext = os.path.splitext(args.input_file)
            args.output_file = f"{base}_sentiment{ext}"
        
        logger.info(f"Processing headlines from {args.input_file}")
        process_csv_file(args.input_file, args.output_file, tokenizer, model)
    
    else:
        logger.info("No input provided. Please specify --headline or --input_file.")
        logger.info("Running interactive mode...")
        
        print("\n=== DistilBERT Sentiment Analyzer for Financial Headlines ===")
        print("Type a headline to analyze or 'quit' to exit\n")
        
        while True:
            user_input = input("Enter headline: ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
                
            if not user_input.strip():
                continue
                
            try:
                result = analyze_sentiment(user_input, tokenizer, model)
                print(f"\nPredicted sentiment: {result['predicted_sentiment']} (confidence: {result['confidence']:.4f})")
                print(f"Probabilities: NEGATIVE={result['probabilities']['NEGATIVE']:.4f}, " + 
                      f"NEUTRAL={result['probabilities']['NEUTRAL']:.4f}, " + 
                      f"POSITIVE={result['probabilities']['POSITIVE']:.4f}\n")
            except Exception as e:
                print(f"Error analyzing headline: {e}")
