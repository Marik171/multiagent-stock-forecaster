"""
PreprocessingAgent is responsible for loading, cleaning, and merging data from multiple sources.
"""
import logging
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from typing import Optional, Dict, List
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingAgent:
    """Agent for preprocessing and merging stock and news data (YouTube removed)."""
    
    def __init__(self):
        """Initialize PreprocessingAgent."""
        self.data_raw = Path(__file__).parent.parent / "data" / "raw"
        self.data_processed = Path(__file__).parent.parent / "data" / "processed"
        self.data_processed.mkdir(parents=True, exist_ok=True)
        
        # Regex patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]", flags=re.UNICODE)
        self.hashtag_pattern = re.compile(r'#\w+')
        self.promo_pattern = re.compile(r'(subscribe|follow|check.*out|visit|join|dm|message)', re.IGNORECASE)
    
    def run(self, ticker: str = "NVDA", stock_df: pd.DataFrame = None, news_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load, preprocess, merge, and save the dataset. Accepts dataframes directly or loads from file if not provided.
        
        Args:
            ticker (str): Stock ticker symbol
            stock_df (pd.DataFrame, optional): Stock data
            news_df (pd.DataFrame, optional): News data
            
        Returns:
            pd.DataFrame: Merged and processed dataset
        """
        try:
            # Load all data sources if not provided
            if stock_df is None:
                stock_df = self._load_stock_data(ticker)
            if news_df is None:
                news_df = self._load_news_data(ticker)
            else:
                # Ensure 'date' column exists in news_df
                if 'date' not in news_df.columns:
                    if 'datetime' in news_df.columns:
                        news_df['date'] = pd.to_datetime(news_df['datetime']).dt.strftime('%Y-%m-%d')
                    else:
                        raise ValueError("news_df must have a 'date' or 'datetime' column")
                # Clean titles and aggregate if needed
                if 'title' in news_df.columns:
                    news_df['title'] = news_df['title'].apply(self._clean_text)
                    # Group by date and concatenate titles
                    news_df = news_df.groupby('date')['title'].agg(lambda x: ' | '.join(x)).reset_index()
                    news_df.columns = ['date', 'news_text']
                elif 'news_text' not in news_df.columns:
                    raise ValueError("news_df must have either 'title' or 'news_text' column")
            # Ensure merge columns are string type in YYYY-MM-DD format
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')
            news_df['date'] = news_df['date'].astype(str)
            # Use stock data as the base for left join
            merged_df = stock_df.copy()
            merged_df = merged_df.merge(
                news_df,
                left_on='Date',
                right_on='date',
                how='left'
            )
            # Clean up merged dataframe
            merged_df = merged_df.drop(columns=['date'], errors='ignore')
            # Fill NaN values in text columns with empty strings
            text_columns = ['news_text']
            for col in text_columns:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna('')
            # Save processed dataset
            output_path = self.data_processed / f"{ticker}_merged_for_sentiment.csv"
            merged_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Saved merged dataset to {output_path}")
            
            return merged_df
            
        except Exception as exc:
            logger.error(f"Error processing data: {str(exc)}")
            raise

    def _find_latest_file(self, pattern: str) -> Optional[Path]:
        """Find the most recent file matching the pattern in raw data directory."""
        files = list(self.data_raw.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs, emojis, hashtags, and special characters."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub('', text)
        
        # Remove hashtags
        text = self.hashtag_pattern.sub('', text)
        
        # Remove promotional content
        text = self.promo_pattern.sub('', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _load_stock_data(self, ticker: str) -> pd.DataFrame:
        """Load and validate stock data."""
        file_path = self._find_latest_file(f"{ticker}_raw_*.csv")
        if not file_path:
            raise FileNotFoundError(f"No stock data file found for {ticker}")
            
        logger.info(f"Loading stock data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure Date column exists and is formatted correctly
        if 'Date' not in df.columns:
            raise ValueError("Stock data missing Date column")
        
        # Handle potential mixed datetime formats
        df['Date'] = pd.to_datetime(df['Date']).astype(str).str[:10]
        return df
    
    def _load_news_data(self, ticker: str) -> pd.DataFrame:
        """Load and preprocess news data."""
        file_path = self._find_latest_file(f"{ticker}_daily_news_*.csv")
        if not file_path:
            raise FileNotFoundError(f"No news data file found for {ticker}")
            
        logger.info(f"Loading news data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert datetime to date
        df['date'] = pd.to_datetime(df['datetime']).astype(str).str[:10]
        
        # Clean titles
        df['title'] = df['title'].apply(self._clean_text)
        
        # Group by date and concatenate titles
        news_grouped = df.groupby('date')['title'].agg(lambda x: ' | '.join(x)).reset_index()
        news_grouped.columns = ['date', 'news_text']
        
        return news_grouped