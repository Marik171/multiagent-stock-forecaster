#!/usr/bin/env python3
"""
Knowledge Base Creator for Stock Analysis

This module creates comprehensive knowledge bases from stock analysis data
that can be easily queried by LLMs like Llama3.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StockAnalysisData:
    """Data structure to hold all stock analysis components."""
    ticker: str
    prediction_df: Optional[pd.DataFrame] = None
    reasoning_analysis: Optional[Dict[str, Any]] = None
    stock_df: Optional[pd.DataFrame] = None
    sentiment_df: Optional[pd.DataFrame] = None
    news_df: Optional[pd.DataFrame] = None

class KnowledgeBaseCreator:
    """
    Creates comprehensive knowledge bases from stock analysis data.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the knowledge base creator.
        
        Args:
            base_dir: Base directory for the project (defaults to parent of this file)
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
            
        # Set up directories
        self.knowledge_base_dir = self.base_dir / "knowledge_base"
        self.data_predictions_dir = self.base_dir / "data_predictions"
        self.data_analysis_dir = self.base_dir / "data" / "analysis"
        self.data_raw_dir = self.base_dir / "data" / "raw"
        self.data_sentiment_dir = self.base_dir / "data" / "sentiment_file"
        
        # Create directories if they don't exist
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"KnowledgeBaseCreator initialized with base dir: {self.base_dir}")
    
    def create_knowledge_base_from_orchestrator_output(self, 
                                                     ticker: str, 
                                                     orchestrator_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a knowledge base from orchestrator output data.
        
        Args:
            ticker: Stock ticker symbol
            orchestrator_outputs: Dictionary containing all orchestrator outputs
            
        Returns:
            Dictionary containing the knowledge base
        """
        try:
            logger.info(f"Creating knowledge base for {ticker} from orchestrator outputs")
            
            # Extract data from orchestrator outputs
            analysis_data = StockAnalysisData(
                ticker=ticker,
                prediction_df=orchestrator_outputs.get('prediction_df'),
                reasoning_analysis=orchestrator_outputs.get('reasoning_analysis'),
                stock_df=orchestrator_outputs.get('stock_df'),
                sentiment_df=orchestrator_outputs.get('sentiment_df'),
                news_df=orchestrator_outputs.get('news_df')
            )
            
            # Generate the knowledge base
            knowledge_base = self._generate_comprehensive_knowledge_base(analysis_data)
            
            # Save the knowledge base
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_knowledge_base_{timestamp}.json"
            filepath = self.knowledge_base_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Knowledge base saved to {filepath}")
            
            # Also create a "latest" symlink for easy access
            latest_filepath = self.knowledge_base_dir / f"{ticker}_knowledge_base_latest.json"
            try:
                if latest_filepath.exists():
                    latest_filepath.unlink()
                # On Windows, use copy instead of symlink
                import shutil
                shutil.copy2(filepath, latest_filepath)
                logger.info(f"Latest knowledge base linked to {latest_filepath}")
            except Exception as e:
                logger.warning(f"Could not create latest symlink: {e}")
            
            return knowledge_base
            
        except Exception as e:
            logger.error(f"Error creating knowledge base for {ticker}: {str(e)}")
            raise
    
    def _generate_comprehensive_knowledge_base(self, data: StockAnalysisData) -> Dict[str, Any]:
        """
        Generate a comprehensive knowledge base from the analysis data.
        
        Args:
            data: StockAnalysisData containing all analysis components
            
        Returns:
            Comprehensive knowledge base dictionary
        """
        knowledge_base = {
            "metadata": {
                "ticker": data.ticker,
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data_sources": [],
                "summary": f"Comprehensive knowledge base for {data.ticker} stock analysis"
            },
            "stock_overview": self._extract_stock_overview(data),
            "technical_analysis": self._extract_technical_analysis(data),
            "sentiment_analysis": self._extract_sentiment_analysis(data),
            "news_analysis": self._extract_news_analysis(data),
            "predictions": self._extract_predictions(data),
            "investment_recommendation": self._extract_investment_recommendation(data),
            "risk_assessment": self._extract_risk_assessment(data),
            "key_insights": self._generate_key_insights(data),
            "query_examples": self._generate_query_examples(data.ticker)
        }
        
        # Update metadata with available data sources
        data_sources = []
        if data.stock_df is not None and not data.stock_df.empty:
            data_sources.append("stock_data")
        if data.sentiment_df is not None and not data.sentiment_df.empty:
            data_sources.append("sentiment_data")
        if data.news_df is not None and not data.news_df.empty:
            data_sources.append("news_data")
        if data.prediction_df is not None and not data.prediction_df.empty:
            data_sources.append("prediction_data")
        if data.reasoning_analysis is not None:
            data_sources.append("reasoning_analysis")
            
        knowledge_base["metadata"]["data_sources"] = data_sources
        
        return knowledge_base
    
    def _extract_stock_overview(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract stock overview information."""
        try:
            if data.stock_df is None or data.stock_df.empty:
                return {"error": "No stock data available"}
            
            df = data.stock_df.copy()
            
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                last_date = df['Date'].max()
            else:
                last_date = datetime.now()
            
            # Get current price (last close price)
            current_price = df['Close'].iloc[-1] if 'Close' in df.columns else 0
            
            # Calculate price statistics
            price_stats = {
                "52_week_high": float(df['Close'].max()) if 'Close' in df.columns else 0,
                "52_week_low": float(df['Close'].min()) if 'Close' in df.columns else 0,
                "average_price": float(df['Close'].mean()) if 'Close' in df.columns else 0,
                "price_volatility": float(df['Close'].std()) if 'Close' in df.columns else None
            }
            
            # Volume statistics
            volume_stats = {
                "latest_volume": float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0,
                "average_volume": float(df['Volume'].mean()) if 'Volume' in df.columns else 0
            }
            
            return {
                "current_price": float(current_price),
                "currency": "USD",
                "exchange": "NASDAQ/NYSE",
                "last_updated": last_date,
                "price_statistics": price_stats,
                "volume_statistics": volume_stats,
                "data_range": {
                    "start_date": df['Date'].min() if 'Date' in df.columns else None,
                    "end_date": df['Date'].max() if 'Date' in df.columns else None,
                    "total_trading_days": len(df)
                }
            }
        except Exception as e:
            logger.error(f"Error extracting stock overview: {str(e)}")
            return {"error": str(e)}
    
    def _extract_technical_analysis(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract technical analysis indicators."""
        try:
            # For now, return empty dict - can be enhanced with technical indicators
            return {}
        except Exception as e:
            logger.error(f"Error extracting technical analysis: {str(e)}")
            return {"error": str(e)}
    
    def _extract_sentiment_analysis(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract sentiment analysis information."""
        try:
            if data.sentiment_df is None or data.sentiment_df.empty:
                return {"error": "No sentiment data available"}
            
            df = data.sentiment_df.copy()
            
            # Get sentiment scores if available
            sentiment_cols = ['Sentiment_Score', 'sentiment_score', 'news_sentiment_score']
            sentiment_col = None
            for col in sentiment_cols:
                if col in df.columns:
                    sentiment_col = col
                    break
            
            if sentiment_col is None:
                return {"error": "No sentiment score column found"}
            
            # Calculate sentiment statistics
            sentiment_scores = df[sentiment_col].dropna()
            
            return {
                "average_sentiment": float(sentiment_scores.mean()),
                "sentiment_trend": "positive" if sentiment_scores.mean() > 0.1 else "neutral" if sentiment_scores.mean() >= -0.1 else "negative",
                "sentiment_volatility": float(sentiment_scores.std()),
                "recent_sentiment": float(sentiment_scores.iloc[-5:].mean()) if len(sentiment_scores) >= 5 else float(sentiment_scores.mean()),
                "total_sentiment_datapoints": len(sentiment_scores)
            }
        except Exception as e:
            logger.error(f"Error extracting sentiment analysis: {str(e)}")
            return {"error": str(e)}
    
    def _extract_news_analysis(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract news analysis information."""
        try:
            if data.news_df is None or data.news_df.empty:
                return {
                    "total_articles": 0,
                    "date_range": {"start": None, "end": None},
                    "recent_headlines": [],
                    "key_topics": []
                }
            
            df = data.news_df.copy()
            
            # Get recent headlines
            headlines = []
            if 'Title' in df.columns:
                headlines = df['Title'].dropna().tail(5).tolist()
            elif 'title' in df.columns:
                headlines = df['title'].dropna().tail(5).tolist()
            
            return {
                "total_articles": len(df),
                "date_range": {
                    "start": df['Date'].min() if 'Date' in df.columns else None,
                    "end": df['Date'].max() if 'Date' in df.columns else None
                },
                "recent_headlines": headlines,
                "key_topics": []  # Could be enhanced with topic modeling
            }
        except Exception as e:
            logger.error(f"Error extracting news analysis: {str(e)}")
            return {"error": str(e)}
    
    def _extract_predictions(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract prediction information."""
        try:
            if data.prediction_df is None or data.prediction_df.empty:
                return {"error": "No prediction data available"}
            
            df = data.prediction_df.copy()
            
            # Get prediction columns
            pred_cols = ['yhat', 'Predicted_Close', 'predicted_close']
            pred_col = None
            for col in pred_cols:
                if col in df.columns:
                    pred_col = col
                    break
            
            if pred_col is None:
                return {"error": "No prediction column found"}
            
            # Get future predictions
            predictions = df[pred_col].dropna()
            
            # If we have dates, get future predictions
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                future_predictions = df[df['Date'] > datetime.now()]
                
                if not future_predictions.empty:
                    next_day = future_predictions.iloc[0]
                    next_week = future_predictions.iloc[:7] if len(future_predictions) >= 7 else future_predictions
                    
                    return {
                        "next_day_prediction": float(next_day[pred_col]),
                        "next_week_average": float(next_week[pred_col].mean()),
                        "prediction_range": {
                            "min": float(predictions.min()),
                            "max": float(predictions.max()),
                            "latest": float(predictions.iloc[-1])
                        },
                        "total_predictions": len(predictions)
                    }
            
            return {
                "latest_prediction": float(predictions.iloc[-1]),
                "prediction_range": {
                    "min": float(predictions.min()),
                    "max": float(predictions.max()),
                    "average": float(predictions.mean())
                },
                "total_predictions": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error extracting predictions: {str(e)}")
            return {"error": str(e)}
    
    def _extract_investment_recommendation(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract investment recommendation."""
        try:
            if data.reasoning_analysis is None:
                return {"error": "No reasoning analysis available"}
            
            reasoning = data.reasoning_analysis
            
            # Extract recommendation
            if "recommendation" in reasoning:
                rec = reasoning["recommendation"]
                return {
                    "action": rec.get("action", "HOLD"),
                    "confidence": rec.get("confidence", "MEDIUM"),
                    "position_size": rec.get("position_size", "Small (5-10%)"),
                    "expected_returns": rec.get("expected_returns", {
                        "1_week": 0,
                        "1_month": 0,
                        "3_months": 0
                    }),
                    "risk_adjusted_score": rec.get("risk_adjusted_score", 50),
                    "strategy_used": rec.get("strategy", "balanced"),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            return {"error": "No recommendation found in reasoning analysis"}
            
        except Exception as e:
            logger.error(f"Error extracting investment recommendation: {str(e)}")
            return {"error": str(e)}
    
    def _extract_risk_assessment(self, data: StockAnalysisData) -> Dict[str, Any]:
        """Extract risk assessment information."""
        try:
            if data.reasoning_analysis is None:
                return {"error": "No risk data available"}
            
            reasoning = data.reasoning_analysis
            
            # Check for risk data in different possible keys
            risk_data = None
            if "risk" in reasoning:
                risk_data = reasoning["risk"]
            elif "risk_assessment" in reasoning:
                risk_data = reasoning["risk_assessment"]
            
            if risk_data:
                return {
                    "risk_level": risk_data.get("risk_level", "MEDIUM"),
                    "risk_score": risk_data.get("risk_score", 50),
                    "risk_factors": risk_data.get("risk_factors", []),
                    "volatility_assessment": risk_data.get("volatility", "MEDIUM"),
                    "market_conditions": risk_data.get("market_conditions", "NORMAL")
                }
            
            return {"error": "No risk assessment found"}
            
        except Exception as e:
            logger.error(f"Error extracting risk assessment: {str(e)}")
            return {"error": str(e)}
    
    def _generate_key_insights(self, data: StockAnalysisData) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        try:
            # Add investment recommendation insight
            if data.reasoning_analysis and "recommendation" in data.reasoning_analysis:
                rec = data.reasoning_analysis["recommendation"]
                action = rec.get("action", "HOLD")
                confidence = rec.get("confidence", "MEDIUM")
                insights.append(f"Investment recommendation: {action} with {confidence}% confidence based on comprehensive analysis")
            
            # Add price insight
            if data.stock_df is not None and not data.stock_df.empty and 'Close' in data.stock_df.columns:
                current_price = data.stock_df['Close'].iloc[-1]
                insights.append(f"Current trading price: ${current_price:.2f}")
            
            # Add prediction insight
            if data.prediction_df is not None and not data.prediction_df.empty:
                pred_cols = ['yhat', 'Predicted_Close', 'predicted_close']
                for col in pred_cols:
                    if col in data.prediction_df.columns:
                        latest_pred = data.prediction_df[col].iloc[-1]
                        insights.append(f"Latest price prediction: ${latest_pred:.2f}")
                        break
            
            # Add sentiment insight
            if data.sentiment_df is not None and not data.sentiment_df.empty:
                sentiment_cols = ['Sentiment_Score', 'sentiment_score', 'news_sentiment_score']
                for col in sentiment_cols:
                    if col in data.sentiment_df.columns:
                        avg_sentiment = data.sentiment_df[col].mean()
                        sentiment_label = "positive" if avg_sentiment > 0.1 else "neutral" if avg_sentiment >= -0.1 else "negative"
                        insights.append(f"Overall market sentiment: {sentiment_label} (score: {avg_sentiment:.3f})")
                        break
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Analysis completed with some data limitations")
        
        return insights
    
    def _generate_query_examples(self, ticker: str) -> List[Dict[str, str]]:
        """Generate example queries for the knowledge base."""
        return [
            {
                "query": f"What is the current investment recommendation for {ticker}?",
                "description": "Get the latest investment action and confidence level"
            },
            {
                "query": f"What are the price predictions for {ticker}?",
                "description": "Get short-term and long-term price forecasts"
            },
            {
                "query": f"How risky is investing in {ticker}?",
                "description": "Get risk assessment and suitability for different investor types"
            },
            {
                "query": f"What is the market sentiment for {ticker}?",
                "description": "Get sentiment analysis from news and social media"
            },
            {
                "query": f"What are the key technical indicators for {ticker}?",
                "description": "Get RSI, MACD, moving averages, and other technical signals"
            },
            {
                "query": f"What are the recent news highlights for {ticker}?",
                "description": "Get recent news articles and key developments"
            },
            {
                "query": f"What are the key insights for {ticker} investment?",
                "description": "Get summarized insights and investment highlights"
            }
        ]
    
    def query_knowledge_base(self, knowledge_base_path: str, query: str) -> str:
        """
        Query a knowledge base with natural language.
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file
            query: Natural language query
            
        Returns:
            Formatted response based on the knowledge base
        """
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            
            # Simple keyword-based matching (can be enhanced with NLP)
            query_lower = query.lower()
            
            # Investment recommendation queries
            if any(word in query_lower for word in ['recommend', 'buy', 'sell', 'hold', 'investment']):
                if 'investment_recommendation' in kb:
                    rec = kb['investment_recommendation']
                    if 'error' not in rec:
                        return f"Investment recommendation: {rec.get('action', 'N/A')} with {rec.get('confidence', 'N/A')} confidence. Strategy: {rec.get('strategy_used', 'balanced')}"
            
            # Price prediction queries
            if any(word in query_lower for word in ['price', 'predict', 'forecast', 'target']):
                if 'predictions' in kb:
                    pred = kb['predictions']
                    if 'error' not in pred:
                        if 'next_day_prediction' in pred:
                            return f"Next day prediction: ${pred['next_day_prediction']:.2f}. Next week average: ${pred.get('next_week_average', 0):.2f}"
                        elif 'latest_prediction' in pred:
                            return f"Latest prediction: ${pred['latest_prediction']:.2f}"
            
            # Risk assessment queries
            if any(word in query_lower for word in ['risk', 'risky', 'safe', 'danger']):
                if 'risk_assessment' in kb:
                    risk = kb['risk_assessment']
                    if 'error' not in risk:
                        return f"Risk level: {risk.get('risk_level', 'N/A')}. Volatility: {risk.get('volatility_assessment', 'N/A')}"
            
            # Sentiment queries
            if any(word in query_lower for word in ['sentiment', 'mood', 'feeling', 'opinion']):
                if 'sentiment_analysis' in kb:
                    sent = kb['sentiment_analysis']
                    if 'error' not in sent:
                        return f"Market sentiment: {sent.get('sentiment_trend', 'N/A')} (average score: {sent.get('average_sentiment', 0):.3f})"
            
            # News queries
            if any(word in query_lower for word in ['news', 'headlines', 'articles']):
                if 'news_analysis' in kb:
                    news = kb['news_analysis']
                    headlines = news.get('recent_headlines', [])
                    if headlines:
                        return f"Recent headlines: {'; '.join(headlines[:3])}"
            
            # Key insights queries
            if any(word in query_lower for word in ['insight', 'summary', 'key', 'important']):
                if 'key_insights' in kb:
                    insights = kb['key_insights']
                    if insights:
                        return '. '.join(insights[:3])
            
            # Default response
            return f"I have analysis data for {kb.get('metadata', {}).get('ticker', 'this stock')}. Try asking about recommendations, predictions, risks, sentiment, or news."
            
        except Exception as e:
            logger.error(f"Error querying knowledge base for {knowledge_base_path}: {str(e)}")
            return {"error": str(e)}
        
    def query_by_ticker(self, ticker: str, query: str) -> str:
        """
        Query a knowledge base by ticker symbol with natural language.
        This is a wrapper method that finds the appropriate knowledge base file for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            query: Natural language query
            
        Returns:
            Formatted response based on the knowledge base
        """
        try:
            # Look for the latest knowledge base file for this ticker
            latest_kb_file = self.knowledge_base_dir / f"{ticker}_knowledge_base_latest.json"
            
            if latest_kb_file.exists():
                knowledge_base_path = latest_kb_file
            else:
                # Look for any knowledge base files for this ticker
                kb_files = list(self.knowledge_base_dir.glob(f"{ticker}_knowledge_base_*.json"))
                if kb_files:
                    # Get the most recent one
                    knowledge_base_path = max(kb_files, key=lambda x: x.stat().st_mtime)
                else:
                    return f"No knowledge base found for {ticker}. Please run analysis for this ticker first."
            
            # Use the existing query_knowledge_base method
            return self.query_knowledge_base(str(knowledge_base_path), query)
            
        except Exception as e:
            logger.error(f"Error querying knowledge base for ticker {ticker}: {str(e)}")
            return f"Error accessing knowledge base for {ticker}: {str(e)}"
    
    def save_knowledge_base_latest(self, ticker: str, knowledge_base: Dict[str, Any]) -> str:
        """
        Save a knowledge base with 'latest' filename for easy access.
        
        Args:
            ticker: Stock ticker symbol
            knowledge_base: Knowledge base dictionary
            
        Returns:
            Path to the saved file
        """
        try:
            # Save with latest filename
            latest_filepath = self.knowledge_base_dir / f"{ticker}_knowledge_base_latest.json"
            
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Latest knowledge base saved to {latest_filepath}")
            return str(latest_filepath)
            
        except Exception as e:
            logger.error(f"Error saving latest knowledge base for {ticker}: {str(e)}")
            raise