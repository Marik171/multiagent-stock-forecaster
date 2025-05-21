"""
SentimentAnalyzerAgent applies sentiment analysis using a fine-tuned DistilBERT model 
to analyze financial news headlines and compute sentiment probabilities.
"""
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
import time
import numpy as np
import re
import glob
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    """Agent for analyzing sentiment in financial news headlines using fine-tuned DistilBERT."""
    
    # Financial context patterns for rule-based enhancement
    POSITIVE_PATTERNS = [
        r'raises? (price )?target',
        r'upgrades?(?! to sell| to negative)',
        r'outperform',
        r'buy rating',
        r'strong buy',
        r'bullish',
        r'exceeded expectations',
        r'beat.{1,20}estimates',
        r'higher.{1,20}guidance',
        r'raises?.{1,20}guidance',
        r'positive.{1,20}outlook',
        r'partnership',
        r'collaboration',
        r'strategic alliance',
    ]
    
    NEGATIVE_PATTERNS = [
        r'downgrades?',
        r'cuts? (price )?target',
        r'sell rating',
        r'bearish',
        r'missed.{1,20}expectations',
        r'missed.{1,20}estimates',
        r'lower.{1,20}guidance',
        r'reduces?.{1,20}guidance',
        r'negative.{1,20}outlook',
        r'concerns',
        r'investigation',
        r'lawsuit',
        r'legal issues',
    ]
    
    def __init__(self, model_path=None, use_rules=True, device=None):
        """
        Initialize the DistilBERT sentiment analyzer.
        
        Args:
            model_path: Path to the fine-tuned DistilBERT model directory
            use_rules: Whether to use rule-based enhancements
            device: Device to run the model on (cuda, mps, cpu)
        """
        self.use_rules = use_rules
        
        # Set device 
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        if model_path is None:
            # Default model path
            script_dir = Path(__file__).parent.parent
            model_path = script_dir / "distilbert_news_NVDA_balanced"
        
        self.load_model(model_path)
        logger.info(f"Loaded DistilBERT sentiment model from {model_path}")
        
        # Compile patterns
        if self.use_rules:
            self.positive_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.POSITIVE_PATTERNS]
            self.negative_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.NEGATIVE_PATTERNS]
        
        # Set up paths
        base_path = Path(__file__).parent.parent
        self.data_processed = base_path / "data" / "processed"
        self.data_sentiment_file = base_path / "data" / "sentiment_file"
        
        # Ensure directories exist
        self.data_processed.mkdir(exist_ok=True, parents=True)
        self.data_sentiment_file.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Data processed directory: {self.data_processed}")
        logger.info(f"Sentiment output directory: {self.data_sentiment_file}")
    
    def load_model(self, model_path):
        """
        Load the fine-tuned DistilBERT model.
        
        Args:
            model_path: Path to the model directory
        """
        try:
            self.config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=self.config
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Get the label mapping
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def check_patterns(self, text):
        """
        Check text for positive and negative patterns using regex.
        
        Args:
            text: Text to check
            
        Returns:
            1 for positive pattern match, -1 for negative pattern match, 0 for no match
        """
        if not self.use_rules:
            return 0
            
        # Check for positive patterns
        for pattern in self.positive_patterns:
            if pattern.search(text):
                return 1
                
        # Check for negative patterns
        for pattern in self.negative_patterns:
            if pattern.search(text):
                return -1
                
        return 0
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single piece of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment probabilities and predicted label
        """
        # Rule-based check
        rule_signal = self.check_patterns(text)
        
        # Model-based analysis
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
        
        # Get the probabilities as a dictionary
        probabilities = {self.id2label[i]: prob for i, prob in enumerate(probs)}
        
        # Adjust probabilities if we have a rule-based signal
        if rule_signal == 1:  # Positive signal
            # Boost positive probability
            probabilities["POSITIVE"] = min(1.0, probabilities["POSITIVE"] * 1.2)
            # Reduce negative probability
            probabilities["NEGATIVE"] = max(0.0, probabilities["NEGATIVE"] * 0.8)
        elif rule_signal == -1:  # Negative signal
            # Boost negative probability
            probabilities["NEGATIVE"] = min(1.0, probabilities["NEGATIVE"] * 1.2)
            # Reduce positive probability
            probabilities["POSITIVE"] = max(0.0, probabilities["POSITIVE"] * 0.8)
        
        # Normalize probabilities to sum to 1
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        # Get the predicted label
        predicted_label = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = max(probabilities.values())
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "rule_signal": rule_signal
        }
    
    def batch_analyze(self, texts, batch_size=32):
        """
        Analyze sentiment for a batch of texts for better efficiency.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Check rule-based patterns
            rule_signals = [self.check_patterns(text) for text in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
            
            # Process each item in the batch
            for j, text in enumerate(batch):
                # Get probabilities and adjust based on rule signal
                probabilities = {self.id2label[k]: prob for k, prob in enumerate(probs[j].tolist())}
                
                # Adjust probabilities if we have a rule-based signal
                if rule_signals[j] == 1:  # Positive signal
                    # Boost positive probability
                    probabilities["POSITIVE"] = min(1.0, probabilities["POSITIVE"] * 1.2)
                    # Reduce negative probability
                    probabilities["NEGATIVE"] = max(0.0, probabilities["NEGATIVE"] * 0.8)
                elif rule_signals[j] == -1:  # Negative signal
                    # Boost negative probability
                    probabilities["NEGATIVE"] = min(1.0, probabilities["NEGATIVE"] * 1.2)
                    # Reduce positive probability
                    probabilities["POSITIVE"] = max(0.0, probabilities["POSITIVE"] * 0.8)
                
                # Normalize probabilities to sum to 1
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v/total for k, v in probabilities.items()}
                
                # Get the predicted label
                predicted_label = max(probabilities.items(), key=lambda x: x[1])[0]
                confidence = max(probabilities.values())
                
                results.append({
                    "text": text,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "rule_signal": rule_signals[j]
                })
        
        return results
    
    def process_dataframe(self, df, text_column="title", batch_size=32):
        """
        Process a DataFrame containing a column of text data.
        
        Args:
            df: DataFrame to process
            text_column: Name of the column containing text data
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with sentiment analysis results added as columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Get texts to analyze
        texts = df[text_column].tolist()
        
        # Analyze sentiment
        start_time = time.time()
        results = self.batch_analyze(texts, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {len(texts)} texts in {elapsed_time:.2f} seconds " + 
                   f"({len(texts)/elapsed_time:.2f} texts/sec)")
        
        # Add results to DataFrame
        df["sentiment_label"] = [r["predicted_label"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]
        df["sentiment_negative_prob"] = [r["probabilities"]["NEGATIVE"] for r in results]
        df["sentiment_neutral_prob"] = [r["probabilities"]["NEUTRAL"] for r in results]
        df["sentiment_positive_prob"] = [r["probabilities"]["POSITIVE"] for r in results]
        
        # Map sentiment_label to sentiment_score
        sentiment_map = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
        df["sentiment_score"] = df["sentiment_label"].map(sentiment_map)
        
        # Add news_sentiment_score as positive - negative probabilities
        df["news_sentiment_score"] = df["sentiment_positive_prob"] - df["sentiment_negative_prob"]
        
        return df
    
    def find_latest_sentiment_file(self, ticker):
        """
        Find the most recent merged_for_sentiment file for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Path to the most recent file or None if not found
        """
        # Look for any files in the /data/processed directory
        all_files = list(self.data_processed.glob("*.csv"))
        
        if not all_files:
            logger.error("No CSV files found in data/processed directory")
            return None
            
        # Sort by modification time, most recent first
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        logger.info(f"Found {len(all_files)} CSV files, most recent: {all_files[0].name}")
        return all_files[0]
    
    def run(self, ticker="NVDA"):
        """
        Run sentiment analysis on the most recent processed data file.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        # Find and load the most recent file
        file_path = self.find_latest_sentiment_file(ticker)
        if not file_path:
            raise FileNotFoundError("No processed data files found")
            
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Extract ticker from filename if not provided
        if ticker == "NVDA" and "_" in file_path.stem:
            ticker = file_path.stem.split("_")[0]
            logger.info(f"Extracted ticker {ticker} from filename")
        
        # Process the dataframe with sentiment analysis
        text_column = None
        for col in ["news_text", "title", "headline"]:
            if col in df.columns:
                text_column = col
                break
                
        if not text_column:
            raise ValueError("Could not find text column in the dataframe")
            
        logger.info(f"Analyzing {text_column} column for sentiment")
        
        # Calculate sentiment without modifying the original dataframe
        texts = df[text_column].tolist()
        start_time = time.time()
        results = self.batch_analyze(texts, batch_size=32)
        elapsed_time = time.time() - start_time
        logger.info(f"Processed {len(texts)} texts in {elapsed_time:.2f} seconds " + 
                   f"({len(texts)/elapsed_time:.2f} texts/sec)")
        
        # Extract only the sentiment scores we need
        result_df = df.copy()
        result_df["news_sentiment_score"] = [
            r["probabilities"]["POSITIVE"] - r["probabilities"]["NEGATIVE"] 
            for r in results
        ]
        
        # Ensure the sentiment_file directory exists
        sentiment_dir = Path(__file__).parent.parent / "data" / "sentiment_file"
        sentiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the result with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = sentiment_dir / f"{ticker}_sentiment_analysis_{timestamp}.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved sentiment analysis to {output_path} with {len(result_df.columns)} columns")
        
        return result_df

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentiment Analysis for Financial News")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--input_file", type=str, help="Path to CSV file with headlines")
    parser.add_argument("--output_file", type=str, help="Path to save output CSV")
    parser.add_argument("--text_column", type=str, default="title", help="Column containing text data")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol")
    args = parser.parse_args()
    
    # Initialize the agent
    agent = SentimentAnalyzerAgent(model_path=args.model_path)
    
    if args.input_file:
        # Process CSV file
        df = pd.read_csv(args.input_file)
        
        # Calculate sentiment without modifying the original dataframe
        texts = df[args.text_column].tolist()
        results = agent.batch_analyze(texts)
        
        # Extract only the sentiment scores we need
        result_df = df.copy()
        result_df["news_sentiment_score"] = [
            r["probabilities"]["POSITIVE"] - r["probabilities"]["NEGATIVE"] 
            for r in results
        ]
        
        # Save results
        output_file = args.output_file or args.input_file.replace(".csv", "_sentiment.csv")
        result_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Print sentiment distribution 
        print(f"Added news_sentiment_score to {len(df)} rows")
    else:
        # Run with default settings
        try:
            result_df = agent.run(ticker=args.ticker)
            print(f"Successfully processed {len(result_df)} rows")
            print(f"News sentiment score range: {result_df['news_sentiment_score'].min():.4f} to {result_df['news_sentiment_score'].max():.4f}")
            print(f"Average sentiment score: {result_df['news_sentiment_score'].mean():.4f}")
        except Exception as e:
            print(f"Error running sentiment analysis: {e}")
            print("Entering interactive mode...")
            # Interactive mode
            print("Interactive mode. Type 'quit' to exit.")
            while True:
                text = input("Enter text to analyze: ")
                if text.lower() == 'quit':
                    break
                result = agent.analyze_text(text)
                print(f"Sentiment: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
                print(f"Probabilities: {result['probabilities']}")

# Remove duplicate main block - this was causing the script to run twice