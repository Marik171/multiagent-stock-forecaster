#!/usr/bin/env python3
"""
Orchestrator agent for coordinating and managing the end-to-end workflow
of all other agents in the multiagent stock forecasting system.

This module provides a central coordinator that handles:
1. Parallel data fetching (stock data and news)
2. Sequential processing (preprocessing, sentiment, forecasting, visualization)
3. Exception handling and logging
4. Performance timing
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.stock_fetch_agent import StockFetchAgent
from agents.news_scraper_agent import NewsScraperAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.balanced_sentiment_agent import DistilBERTSentimentAgent
from agents.prophet_model_enhanced_simple import PredictorAgent as ProphetModelEnhanced
from agents.visualizer_agent import VisualizerAgent
from agents.reasoning_agent import ReasoningAgent

import logging
import time
import concurrent.futures
import pandas as pd
from typing import Dict, Any, Type, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Coordinator agent that orchestrates the workflow between all other agents
    in the multiagent stock forecasting system.
    """
    
    def __init__(self,
                 stock_agent_cls,
                 news_agent_cls,
                 preprocessing_agent_cls,
                 sentiment_agent_cls,
                 model_agent_cls,
                 visualizer_agent_cls,
                 reasoning_agent_cls=None):
        """
        Initialize the orchestrator with all required agent classes.
        
        Args:
            stock_agent_cls: Class for fetching stock data
            news_agent_cls: Class for scraping news articles
            preprocessing_agent_cls: Class for preprocessing and merging data
            sentiment_agent_cls: Class for sentiment analysis
            model_agent_cls: Class for price forecasting
            visualizer_agent_cls: Class for visualization
            reasoning_agent_cls: Class for investment reasoning (optional)
        """
        self.stock_agent_cls = stock_agent_cls
        self.news_agent_cls = news_agent_cls
        self.preprocessing_agent_cls = preprocessing_agent_cls
        self.sentiment_agent_cls = sentiment_agent_cls
        self.model_agent_cls = model_agent_cls
        self.visualizer_agent_cls = visualizer_agent_cls
        self.reasoning_agent_cls = reasoning_agent_cls or ReasoningAgent
        
        # Create output directories if they don't exist
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("OrchestratorAgent initialized with all required agent classes")
    
    def _parallel_fetch(self, stock_agent, news_agent, ticker, start_date, end_date, max_news):
        """
        Internal helper to fetch stock data and news articles concurrently.
        
        Args:
            stock_agent: Instantiated stock agent
            news_agent: Instantiated news agent
            ticker: Stock ticker symbol
            start_date: Start date for data fetching
            end_date: End date for data fetching
            max_news: Maximum number of news articles to fetch
            
        Returns:
            tuple: (stock_df, news_df) containing the fetched data
        """
        stock_df = None
        news_df = None
        stock_error = None
        news_error = None
        
        logger.info(f"Starting parallel fetch for {ticker} from {start_date} to {end_date}")
        
        # Use ThreadPoolExecutor for concurrent fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both fetch tasks
            stock_future = executor.submit(
                stock_agent.fetch_stock_data, ticker, start_date, end_date
            )
            news_future = executor.submit(
                news_agent.fetch_news, ticker, start_date, end_date, max_news
            )
            
            # Wait for stock data result
            try:
                stock_df = stock_future.result()
                logger.info(f"Stock data fetched successfully: {len(stock_df)} records")
            except Exception as e:
                stock_error = e
                logger.error(f"Stock data fetch failed: {str(e)}")
            
            # Wait for news data result
            try:
                news_df = news_future.result()
                logger.info(f"News data fetched successfully: {len(news_df)} articles")
                
                # Filter out Market Summary entries
                try:
                    from agents.filter_market_summary import filter_market_summary
                    # Find the raw data path from project structure
                    raw_data_path = self.project_root / "data" / "raw"
                    latest_news_file = sorted(list(raw_data_path.glob(f"{ticker}_daily_news_*.csv")), reverse=True)[0]
                    news_df = filter_market_summary(latest_news_file)
                    logger.info(f"Filtered Market Summary entries from news data")
                except Exception as filter_error:
                    logger.warning(f"Could not filter Market Summary entries: {str(filter_error)}")
                    
            except Exception as e:
                news_error = e
                logger.error(f"News data fetch failed: {str(e)}")
        
        # Check if both fetches succeeded
        if stock_error:
            raise stock_error
        if news_error:
            raise news_error
            
        return stock_df, news_df
    
    def run(self,
            ticker: str,
            start_date: str,
            end_date: str,
            forecast_period: int,
            max_news_articles: int = 5000,
            reasoning_strategy: str = 'balanced') -> Dict[str, Any]:
        """
        Orchestrates the complete workflow from data fetching to visualization.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast_period: Number of days to forecast
            max_news_articles: Maximum number of news articles to fetch (default 5000)
            reasoning_strategy: Investment strategy for reasoning ('balanced', 'growth', 'value', 'momentum', 'contrarian')
            
        Returns:
            dict: Dictionary containing outputs from each stage including
                 dataframes, visualization figures, and reasoning analysis
        """
        start_time = time.time()
        logger.info(f"Starting workflow for {ticker} from {start_date} to {end_date}")
        
        # Initialize result dictionary with proper types
        results: Dict[str, Any] = {
            "stock_df": None,
            "news_df": None,
            "merged_df": None,
            "sentiment_df": None,
            "prediction_df": None,
            "price_fig": None,
            "sentiment_fig": None,
            "reasoning_analysis": None
        }
        
        try:
            # 1. Initialize all agents
            stock_agent = self.stock_agent_cls()
            news_agent = self.news_agent_cls()
            preprocessing_agent = self.preprocessing_agent_cls()
            sentiment_agent = self.sentiment_agent_cls()
            model_agent = self.model_agent_cls(forecast_days=forecast_period)
            visualizer_agent = self.visualizer_agent_cls()
            reasoning_agent = self.reasoning_agent_cls()
            
            # 2. Parallel fetch stock data and news
            stock_df, news_df = self._parallel_fetch(
                stock_agent, news_agent, ticker, start_date, end_date, max_news_articles
            )
            results["stock_df"] = stock_df
            results["news_df"] = news_df
            
            # 3. Preprocessing: merge stock and news data
            try:
                logger.info("Starting preprocessing and data merge...")
                # Pass the freshly fetched data to preprocessing agent
                merged_df = preprocessing_agent.run(ticker, stock_df=stock_df, news_df=news_df)
                results["merged_df"] = merged_df
                logger.info(f"Preprocessing completed. Shape: {merged_df.shape}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                raise
            
            # 4. Sentiment analysis
            try:
                logger.info("Starting sentiment analysis...")
                sentiment_output_filename = f"{ticker}_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                sentiment_df = sentiment_agent.process_dataframe(merged_df, output_path=sentiment_output_filename)
                results["sentiment_df"] = sentiment_df
                logger.info(f"Sentiment analysis completed. Shape: {sentiment_df.shape}")
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {str(e)}")
                raise
                
            # 5. Forecasting
            try:
                logger.info(f"Starting price forecasting for {forecast_period} days...")
                model_results = model_agent.run(ticker, sentiment_df)
                mae = model_results.get('mae')
                rmse = model_results.get('rmse')
                prediction_df = model_results.get('predictions_df')
                if prediction_df is not None:
                    prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])
                    prediction_df['ticker'] = ticker
                    results["prediction_df"] = prediction_df
                    logger.info(f"Forecasting completed. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                else:
                    logger.error("No predictions_df returned from Prophet model.")
                    raise ValueError("No predictions_df returned from Prophet model.")
            except Exception as e:
                logger.error(f"Forecasting failed: {str(e)}")
                raise
            
            # 6. Investment Reasoning Analysis
            try:
                logger.info("Starting investment reasoning analysis...")
                reasoning_analysis = reasoning_agent.analyze_predictions(
                    prediction_df, ticker, strategy=reasoning_strategy, save_analysis=True
                )
                results["reasoning_analysis"] = reasoning_analysis
                logger.info(f"Reasoning analysis completed. Recommendation: {reasoning_analysis['recommendation']['action']}")
            except Exception as e:
                logger.error(f"Reasoning analysis failed: {str(e)}")
                # Non-critical error, continue without raising
            
            # 7. Visualization
            try:
                logger.info("Generating visualizations...")
                price_fig = visualizer_agent.plot_price(prediction_df)
                sentiment_fig = visualizer_agent.plot_sentiment_dist(prediction_df)
                results["price_fig"] = price_fig
                results["sentiment_fig"] = sentiment_fig
                logger.info("Visualizations generated successfully")
                
                # No longer saving figures as PNG in reports directory
                logger.info("Visualizations available in the results dictionary")
            except Exception as e:
                logger.error(f"Visualization failed: {str(e)}")
                # Non-critical error, continue without raising
                
            # Calculate and log total runtime
            runtime = time.time() - start_time
            logger.info(f"Workflow completed in {runtime:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            # Return partial results
            return results


if __name__ == "__main__":
    from agents.stock_fetch_agent import StockFetchAgent
    from agents.news_scraper_agent import NewsScraperAgent
    from agents.preprocessing_agent import PreprocessingAgent
    from agents.balanced_sentiment_agent import DistilBERTSentimentAgent
    from agents.prophet_model_enhanced_simple import PredictorAgent as ProphetModelEnhanced
    from agents.visualizer_agent import VisualizerAgent
    from agents.reasoning_agent import ReasoningAgent

    orchestrator = OrchestratorAgent(
        StockFetchAgent,
        NewsScraperAgent,
        PreprocessingAgent,
        DistilBERTSentimentAgent,
        ProphetModelEnhanced,
        VisualizerAgent,
        ReasoningAgent
    )
    
    outputs = orchestrator.run(
        ticker="NVDA",
        start_date="2025-02-01",
        end_date="2025-05-25",
        forecast_period=14
    )
    
    print("Workflow completed. Outputs available in 'outputs' dict.")
