"""
Test script for PreprocessingAgent functionality
"""
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from agents.preprocessing_agent import PreprocessingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_preprocessing_agent():
    """Test the main functionality of PreprocessingAgent."""
    try:
        logger.info("Initializing PreprocessingAgent...")
        agent = PreprocessingAgent()
        
        # Test parameters
        ticker = "NVDA"
        
        logger.info(f"Testing PreprocessingAgent with {ticker} data")
        
        # Process data
        df = agent.run(ticker=ticker)
        
        # Verify the result
        logger.info("\nValidating processed dataset...")
        
        # Check if DataFrame is not empty
        if df.empty:
            logger.error("❌ Processed DataFrame is empty")
            return False
            
        # Check required columns
        required_columns = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'news_text', 'youtube_comments'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return False
        
        logger.info("✓ All required columns present")
        
        # Verify data cleaning
        logger.info("\nChecking text cleaning...")
        
        # Check if news text is cleaned (lowercase, no URLs, etc.)
        if not df['news_text'].empty:
            sample_news = df['news_text'].iloc[0]
            logger.info(f"\nSample cleaned news text:\n{sample_news[:200]}...")
            
            # Basic validation of cleaning
            if any(x in sample_news for x in ['http', 'www', '#', '@']):
                logger.error("❌ News text contains unwanted elements")
                return False
        
        # Check if YouTube comments are cleaned
        if not df['youtube_comments'].empty:
            sample_comment = df['youtube_comments'].iloc[0]
            logger.info(f"\nSample cleaned YouTube comment:\n{sample_comment[:200]}...")
            
            # Basic validation of cleaning
            if any(x in sample_comment for x in ['http', 'www', '#', '@']):
                logger.error("❌ YouTube comments contain unwanted elements")
                return False
        
        # Check date format consistency
        date_format = '%Y-%m-%d'
        try:
            pd.to_datetime(df['Date'], format=date_format)
            logger.info("✓ Date format is consistent")
        except ValueError:
            logger.error("❌ Inconsistent date format")
            return False
        
        # Check output file
        output_path = Path(__file__).parent / "data" / "processed" / f"{ticker}_merged_for_sentiment.csv"
        if not output_path.exists():
            logger.error("❌ Output file not created")
            return False
        
        logger.info(f"\nProcessed data shape: {df.shape}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info("\nSample of processed data:")
        logger.info("\n" + str(df.head()))
        
        logger.info("\n✓ PreprocessingAgent test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_preprocessing_agent()
    print(f"\nTest {'✓ succeeded' if success else '❌ failed'}")