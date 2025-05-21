"""
StockFetchAgent is responsible for fetching historical stock data and calculating technical indicators.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockFetchAgentException(Exception):
    """Custom exception for StockFetchAgent errors."""
    pass

class StockFetchAgent:
    """Agent for fetching stock data and calculating technical indicators."""
    
    def __init__(self):
        """Initialize StockFetchAgent with default settings."""
        self.data_path = Path(__file__).parent.parent / "data" / "raw"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """Calculate MACD"""
        fast_ema = self._calculate_ema(data, fast)
        slow_ema = self._calculate_ema(data, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd_line, signal)
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, data: pd.Series, period: int, std_dev: int) -> tuple:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset."""
        # SMA calculations
        df['SMA_20'] = self._calculate_sma(df['Close'], 20)
        df['SMA_50'] = self._calculate_sma(df['Close'], 50)
        df['SMA_200'] = self._calculate_sma(df['Close'], 200)

        # EMA calculations
        df['EMA_12'] = self._calculate_ema(df['Close'], 12)
        df['EMA_26'] = self._calculate_ema(df['Close'], 26)

        # RSI
        df['RSI_14'] = self._calculate_rsi(df['Close'], 14)

        # MACD
        macd_line, signal_line = self._calculate_macd(df['Close'], 12, 26, 9)
        df['MACD_12_26_9'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Hist'] = macd_line - signal_line

        # Bollinger Bands
        upper_band, lower_band = self._calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_Upper'] = upper_band
        df['BB_Lower'] = lower_band

        # Volume SMA
        df['Volume_SMA_20'] = self._calculate_sma(df['Volume'], 20)

        return df

    def _save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save the data to CSV file."""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_raw_{timestamp}.csv"
            filepath = self.data_path / filename
            
            # Save to CSV
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            error_msg = f"Error saving data for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise StockFetchAgentException(error_msg)

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (Union[str, datetime], optional): Start date
            end_date (Union[str, datetime], optional): End date
            interval (str, optional): Data interval (1d, 1h, etc.)
            
        Returns:
            pd.DataFrame: Stock data with technical indicators
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Initialize ticker
            stock = yf.Ticker(symbol)
            
            # Fetch historical data
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                raise StockFetchAgentException(f"No data found for symbol {symbol}")
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            # Save raw data
            self._save_data(df, symbol)
            
            return df
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise StockFetchAgentException(error_msg)
        
if __name__ == "__main__":
    agent = StockFetchAgent()
    # Interactive mode
    while True:
        symbol = input("Enter stock symbol (or 'exit' to quit): ").strip().upper()
        if symbol == "EXIT":
            break
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip() or datetime.now().strftime("%Y-%m-%d")
        
        try:
            data = agent.fetch_stock_data(symbol, start_date, end_date)
            print(data.head())
        except StockFetchAgentException as e:
            logger.error(f"Failed to fetch data: {e}")

         