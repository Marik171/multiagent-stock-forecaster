"""
Test script for StockFetchAgent functionality with enhanced error handling.
"""
import logging
from agents.stock_fetch_agent import StockFetchAgent
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stock_fetch_agent():
    """Test the main functionality of StockFetchAgent."""
    try:
        logger.info("Initializing StockFetchAgent...")
        agent = StockFetchAgent()
        
        # Test parameters
        symbol = "NVDA"  # Apple stock
        end_date = datetime.now()  # Today's date
        start_date = end_date - timedelta(days=365)  # Last 365 days of data
        
        logger.info(f"Testing StockFetchAgent with {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Fetch data
        logger.info("Fetching stock data...")
        df = agent.fetch_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print basic information
        logger.info("\nDataset Info:")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nColumns (technical indicators):")
        logger.info(df.columns.tolist())
        
        # Print last few rows of key indicators
        logger.info("\nSample of calculated indicators:")
        key_indicators = ['Close', 'SMA_20', 'RSI_14', 'MACD_12_26_9']
        available_indicators = [col for col in key_indicators if col in df.columns]
        if available_indicators:
            logger.info("\n" + str(df[available_indicators].tail()))
        else:
            logger.warning("No matching indicators found in the dataset")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_stock_fetch_agent()