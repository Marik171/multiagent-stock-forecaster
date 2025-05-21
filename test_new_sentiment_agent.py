#!/usr/bin/env python3
"""
Test script for the updated SentimentAnalyzerAgent using the fine-tuned DistilBERT model.
"""

import sys
import os
import logging
import pandas as pd
import time
from pathlib import Path

# Add the parent directory to the path to import the agents module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent
from agents.sentiment_agent import SentimentAnalyzerAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def test_single_headlines():
    """Test the agent on individual headlines."""
    # Initialize the agent
    agent = SentimentAnalyzerAgent()
    
    # Define test headlines with expected sentiments
    test_headlines = [
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
    
    logger.info("Testing individual headlines:")
    logger.info("-" * 80)
    
    for headline in test_headlines:
        result = agent.analyze_text(headline)
        logger.info(f"Headline: {headline}")
        logger.info(f"Sentiment: {result['predicted_label']} with confidence: {result['confidence']:.4f}")
        logger.info(f"Probabilities: {', '.join([f'{k}: {v:.4f}' for k, v in result['probabilities'].items()])}")
        logger.info(f"Rule signal: {result['rule_signal']}")
        logger.info("-" * 80)
    
    return True

def test_batch_processing():
    """Test the batch processing capability of the agent."""
    # Initialize the agent
    agent = SentimentAnalyzerAgent()
    
    # Create a sample DataFrame
    data = {
        "datetime": [f"2025-05-{i+1}" for i in range(5)],
        "title": [
            "NVDA stock surges on strong AI demand",
            "NVIDIA announces partnership with leading cloud providers",
            "NVIDIA delays product launch, shares fall",
            "Analysts remain neutral on NVIDIA's prospects",
            "NVIDIA reports quarterly earnings in line with expectations"
        ],
        "source": ["Source A"] * 5
    }
    df = pd.DataFrame(data)
    
    logger.info("Testing batch processing:")
    logger.info("-" * 80)
    
    # Process DataFrame
    start_time = time.time()
    result_df = agent.process_dataframe(df)
    elapsed = time.time() - start_time
    
    logger.info(f"Processed {len(df)} headlines in {elapsed:.4f} seconds")
    logger.info(f"\nResults:\n{result_df[['title', 'sentiment_label', 'sentiment_confidence', 'sentiment_score']].to_string(index=False)}")
    logger.info(f"\nSentiment Distribution:\n{result_df['sentiment_label'].value_counts().to_string()}")
    
    return True

def compare_models():
    """Compare the balanced model with the original model."""
    # Initialize the agents
    script_dir = Path(__file__).parent
    original_model_path = script_dir / "distilbert_news_NVDA"
    balanced_model_path = script_dir / "distilbert_news_NVDA_balanced"
    
    # Check if both models exist
    if not original_model_path.exists() or not balanced_model_path.exists():
        logger.warning("One or both models not found. Skipping comparison.")
        return False
    
    original_agent = SentimentAnalyzerAgent(model_path=original_model_path)
    balanced_agent = SentimentAnalyzerAgent(model_path=balanced_model_path)
    
    # Define test headlines for comparison
    test_headlines = [
        # Headlines that might be classified differently
        "NVDA announces new products but faces supply chain challenges",
        "NVIDIA reports growth but at a slower pace than last quarter",
        "Analysts have mixed opinions on NVIDIA's outlook",
        "NVIDIA stock price stabilizes after recent volatility",
        "NVIDIA's market share remains steady despite new competition"
    ]
    
    logger.info("Comparing original vs. balanced models:")
    logger.info("-" * 80)
    
    for headline in test_headlines:
        original_result = original_agent.analyze_text(headline)
        balanced_result = balanced_agent.analyze_text(headline)
        
        logger.info(f"Headline: {headline}")
        logger.info(f"Original model: {original_result['predicted_label']} with confidence: {original_result['confidence']:.4f}")
        logger.info(f"Balanced model: {balanced_result['predicted_label']} with confidence: {balanced_result['confidence']:.4f}")
        
        # Print differences if any
        if original_result['predicted_label'] != balanced_result['predicted_label']:
            logger.info("DIFFERENT CLASSIFICATIONS!")
            
        logger.info("-" * 80)
    
    return True

if __name__ == "__main__":
    logger.info("Running tests for SentimentAnalyzerAgent with DistilBERT model")
    
    # Run tests
    test_single_headlines()
    test_batch_processing()
    compare_models()
    
    logger.info("All tests complete!")
