#!/usr/bin/env python3
"""
Test the fine-tuned DistilBERT model on a variety of news headlines.
"""

import os
import torch
import torch.nn.functional as F
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Check for device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
logger.info(f"Using device: {device}")

def load_model_for_inference(model_dir: str):
    """
    Load the fine‐tuned DistilBERT classifier from model_dir.
    Returns (tokenizer, model_on_device).
    """
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config
    )
    model.to(device)
    model.eval()
    return tokenizer, model

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

    id2label = model.config.id2label
    return {id2label[i]: probs[i] for i in range(len(probs))}

def test_headlines(headlines, model_dir="./distilbert_news_NVDA/"):
    """Test a list of headlines and print the results."""
    tokenizer, model = load_model_for_inference(model_dir)
    
    results = []
    for headline in headlines:
        probs = predict_sentiment(headline, tokenizer, model)
        prediction = max(probs.items(), key=lambda x: x[1])[0]
        confidence = max(probs.values())
        results.append({
            "headline": headline,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs
        })
    
    # Print results in a formatted way
    logger.info(f"{'='*80}")
    logger.info(f"Testing {len(headlines)} headlines with model from {model_dir}")
    logger.info(f"{'='*80}")
    
    for i, result in enumerate(results):
        logger.info(f"Headline {i+1}: {result['headline']}")
        logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        prob_str = ", ".join([f"{k}: {v:.4f}" for k, v in result['probabilities'].items()])
        logger.info(f"Probabilities: {prob_str}")
        logger.info(f"{'-'*80}")
    
    return results

if __name__ == "__main__":
    headlines_to_test = [
        # Clearly positive headlines
        "NVDA stock hits all-time high after earnings beat",
        "NVIDIA's new AI chip sales exceed expectations, boosting investor confidence",
        "NVIDIA partners with major tech firms to develop next-generation computing platform",
        
        # Clearly negative headlines
        "NVDA shares drop 10% as market concerns grow",
        "NVIDIA faces lawsuit over patent infringement claims",
        "NVIDIA delays chip release, citing production issues",
        
        # Neutral headlines
        "NVIDIA to present at upcoming tech conference",
        "NVIDIA CEO discusses future of computing in interview",
        "NVIDIA announces regular quarterly dividend",
        
        # Mixed or ambiguous sentiment
        "NVIDIA restructures amid market challenges, analysts remain optimistic",
        "NVIDIA sales growth slows but still beats conservative estimates",
        "NVIDIA faces increased competition, but maintains market leadership"
    ]
    
    test_headlines(headlines_to_test)
