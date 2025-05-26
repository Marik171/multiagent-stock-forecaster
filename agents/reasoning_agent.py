"""
ReasoningAgent provides intelligent analysis and investment recommendations
based on predicted stock prices, sentiment data, and technical indicators.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReasoningAgent:
    """Agent for analyzing predictions and providing investment recommendations."""
    
    def __init__(self):
        """Initialize ReasoningAgent."""
        self.data_predictions = Path(__file__).parent.parent / "data_predictions"
        self.data_analysis = Path(__file__).parent.parent / "data" / "analysis"
        self.data_analysis.mkdir(parents=True, exist_ok=True)
        
        # Risk tolerance levels
        self.risk_levels = {
            'conservative': {'max_risk': 5, 'min_confidence': 75},
            'moderate': {'max_risk': 10, 'min_confidence': 65},
            'aggressive': {'max_risk': 20, 'min_confidence': 55}
        }
        
        # Investment strategies
        self.strategies = {
            'growth': {'weight_momentum': 0.4, 'weight_sentiment': 0.3, 'weight_trend': 0.3},
            'value': {'weight_momentum': 0.2, 'weight_sentiment': 0.2, 'weight_trend': 0.6},
            'momentum': {'weight_momentum': 0.6, 'weight_sentiment': 0.25, 'weight_trend': 0.15},
            'contrarian': {'weight_momentum': 0.1, 'weight_sentiment': 0.7, 'weight_trend': 0.2}
        }

    def _calculate_price_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price momentum indicators."""
        try:
            # Get training and test data
            historical_data = df[df['Period'].isin(['Training', 'Test'])].copy()
            forecast_data = df[df['Period'] == 'Forecast'].copy()
            
            if historical_data.empty or forecast_data.empty:
                return {'momentum_score': 0, 'trend_strength': 0, 'volatility': 0}
            
            # Calculate recent momentum (last 5 days of historical data)
            recent_prices = historical_data['Actual_Close'].tail(5).values
            if len(recent_prices) >= 2:
                recent_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            else:
                recent_momentum = 0
            
            # Calculate forecast momentum
            forecast_prices = forecast_data['Predicted_Close'].values
            current_price = historical_data['Actual_Close'].iloc[-1]
            
            if len(forecast_prices) > 0:
                short_term_return = (forecast_prices[4] - current_price) / current_price * 100 if len(forecast_prices) > 4 else 0
                medium_term_return = (forecast_prices[9] - current_price) / current_price * 100 if len(forecast_prices) > 9 else 0
                long_term_return = (forecast_prices[-1] - current_price) / current_price * 100
            else:
                short_term_return = medium_term_return = long_term_return = 0
            
            # Calculate trend strength
            if len(forecast_prices) >= 5:
                trend_consistency = np.sum(np.diff(forecast_prices[:10]) > 0) / len(np.diff(forecast_prices[:10])) * 100
            else:
                trend_consistency = 50
            
            # Calculate volatility
            if len(historical_data) >= 10:
                returns = historical_data['Actual_Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            else:
                volatility = 20  # Default moderate volatility
            
            return {
                'recent_momentum': recent_momentum,
                'short_term_return': short_term_return,
                'medium_term_return': medium_term_return,
                'long_term_return': long_term_return,
                'trend_strength': trend_consistency,
                'volatility': volatility,
                'momentum_score': (recent_momentum + short_term_return) / 2
            }
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {str(e)}")
            return {'momentum_score': 0, 'trend_strength': 50, 'volatility': 20}

    def _analyze_sentiment_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze sentiment trends and their implications."""
        try:
            # Filter out rows with missing sentiment scores
            sentiment_data = df[df['news_sentiment_score'].notna()].copy()
            
            if sentiment_data.empty:
                return {'avg_sentiment': 0, 'sentiment_trend': 0, 'sentiment_volatility': 0}
            
            sentiment_scores = sentiment_data['news_sentiment_score'].values
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores)
            
            # Calculate sentiment trend (recent vs older)
            if len(sentiment_scores) >= 10:
                recent_sentiment = np.mean(sentiment_scores[-5:])
                older_sentiment = np.mean(sentiment_scores[:5])
                sentiment_trend = recent_sentiment - older_sentiment
            else:
                sentiment_trend = 0
            
            # Calculate sentiment volatility
            sentiment_volatility = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            # Sentiment momentum (how quickly sentiment is changing)
            if len(sentiment_scores) >= 3:
                sentiment_momentum = np.mean(np.diff(sentiment_scores[-3:]))
            else:
                sentiment_momentum = 0
            
            return {
                'avg_sentiment': avg_sentiment,
                'sentiment_trend': sentiment_trend,
                'sentiment_volatility': sentiment_volatility,
                'sentiment_momentum': sentiment_momentum,
                'bullish_signals': np.sum(np.array(sentiment_scores) > 0.3),
                'bearish_signals': np.sum(np.array(sentiment_scores) < -0.3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'avg_sentiment': 0, 'sentiment_trend': 0, 'sentiment_volatility': 0}

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        try:
            historical_data = df[df['Period'].isin(['Training', 'Test'])].copy()
            forecast_data = df[df['Period'] == 'Forecast'].copy()
            
            if historical_data.empty:
                return {'risk_score': 50, 'max_drawdown': 0, 'prediction_accuracy': 0}
            
            # Calculate prediction accuracy on test data
            test_data = historical_data[historical_data['Period'] == 'Test']
            if len(test_data) > 0:
                prediction_errors = np.abs(test_data['Actual_Close'] - test_data['Predicted_Close']) / test_data['Actual_Close']
                prediction_accuracy = (1 - np.mean(prediction_errors)) * 100
            else:
                prediction_accuracy = 70  # Default assumption
            
            # Calculate maximum drawdown
            prices = historical_data['Actual_Close'].values
            if len(prices) > 1:
                peak = np.maximum.accumulate(prices)
                drawdown = (peak - prices) / peak
                max_drawdown = np.max(drawdown) * 100
            else:
                max_drawdown = 0
            
            # Calculate price volatility
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252) * 100
            else:
                volatility = 20
            
            # Forecast uncertainty (spread in predictions)
            if len(forecast_data) > 1:
                forecast_volatility = np.std(forecast_data['Predicted_Close']) / np.mean(forecast_data['Predicted_Close']) * 100
            else:
                forecast_volatility = 10
            
            # Overall risk score (0-100, higher = riskier)
            risk_score = min(100, (volatility + max_drawdown + forecast_volatility) / 3)
            
            return {
                'risk_score': risk_score,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'prediction_accuracy': prediction_accuracy,
                'forecast_uncertainty': forecast_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'risk_score': 50, 'max_drawdown': 10, 'prediction_accuracy': 70}

    def _generate_price_targets(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate price targets based on different scenarios."""
        try:
            current_price = df[df['Period'].isin(['Training', 'Test'])]['Actual_Close'].iloc[-1]
            forecast_data = df[df['Period'] == 'Forecast'].copy()
            
            if forecast_data.empty:
                return {'current': current_price, 'target_1w': current_price, 'target_1m': current_price, 'target_3m': current_price}
            
            forecast_prices = forecast_data['Predicted_Close'].values
            
            # Different time horizon targets
            target_1w = forecast_prices[4] if len(forecast_prices) > 4 else current_price
            target_1m = forecast_prices[19] if len(forecast_prices) > 19 else current_price
            target_3m = forecast_prices[-1] if len(forecast_prices) > 0 else current_price
            
            # Calculate confidence intervals (conservative estimates)
            volatility = np.std(forecast_prices[:10]) if len(forecast_prices) > 10 else current_price * 0.05
            
            return {
                'current': current_price,
                'target_1w': target_1w,
                'target_1m': target_1m,
                'target_3m': target_3m,
                'conservative_1w': target_1w - volatility,
                'aggressive_1w': target_1w + volatility,
                'conservative_1m': target_1m - volatility * 2,
                'aggressive_1m': target_1m + volatility * 2,
                'conservative_3m': target_3m - volatility * 3,
                'aggressive_3m': target_3m + volatility * 3
            }
            
        except Exception as e:
            logger.error(f"Error generating price targets: {str(e)}")
            current_price = 100  # Default fallback
            return {'current': current_price, 'target_1w': current_price, 'target_1m': current_price, 'target_3m': current_price}

    def _determine_investment_action(self, analysis: Dict[str, Any], strategy: str = 'balanced') -> Dict[str, Any]:
        """Determine the recommended investment action."""
        try:
            momentum = analysis['momentum']
            sentiment = analysis['sentiment']
            risk = analysis['risk']
            targets = analysis['targets']
            
            # Calculate weighted scores based on strategy
            if strategy in self.strategies:
                weights = self.strategies[strategy]
            else:
                weights = {'weight_momentum': 0.33, 'weight_sentiment': 0.33, 'weight_trend': 0.34}
            
            # Normalize scores to 0-100 scale
            momentum_score = max(0, min(100, 50 + momentum['momentum_score'] * 2))
            sentiment_score = max(0, min(100, 50 + sentiment['avg_sentiment'] * 100))
            trend_score = momentum['trend_strength']
            
            # Calculate overall bullish score
            overall_score = (
                momentum_score * weights['weight_momentum'] +
                sentiment_score * weights['weight_sentiment'] +
                trend_score * weights['weight_trend']
            )
            
            # Determine action based on score and risk
            if overall_score >= 70 and risk['risk_score'] <= 30:
                action = "STRONG BUY"
                confidence = min(95, overall_score + (100 - risk['risk_score']) / 5)
            elif overall_score >= 60 and risk['risk_score'] <= 50:
                action = "BUY"
                confidence = min(85, overall_score)
            elif overall_score >= 40 and overall_score < 60:
                action = "HOLD"
                confidence = 60 + (overall_score - 40)
            elif overall_score >= 30 and risk['risk_score'] <= 70:
                action = "WEAK SELL"
                confidence = min(75, 100 - overall_score)
            else:
                action = "SELL"
                confidence = min(90, 100 - overall_score + risk['risk_score'] / 3)
            
            # Calculate position sizing recommendation
            if action in ["STRONG BUY", "BUY"]:
                if risk['risk_score'] <= 20:
                    position_size = "Large (15-25%)"
                elif risk['risk_score'] <= 40:
                    position_size = "Medium (10-15%)"
                else:
                    position_size = "Small (5-10%)"
            elif action == "HOLD":
                position_size = "Maintain current position"
            else:
                position_size = "Reduce or exit position"
            
            # Calculate expected returns
            current = targets['current']
            expected_1w = (targets['target_1w'] - current) / current * 100
            expected_1m = (targets['target_1m'] - current) / current * 100
            expected_3m = (targets['target_3m'] - current) / current * 100
            
            return {
                'action': action,
                'confidence': round(confidence, 1),
                'overall_score': round(overall_score, 1),
                'position_size': position_size,
                'expected_return_1w': round(expected_1w, 2),
                'expected_return_1m': round(expected_1m, 2),
                'expected_return_3m': round(expected_3m, 2),
                'risk_adjusted_score': round(overall_score - risk['risk_score'] / 2, 1)
            }
            
        except Exception as e:
            logger.error(f"Error determining investment action: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 50,
                'overall_score': 50,
                'position_size': 'Small (5-10%)',
                'expected_return_1w': 0,
                'expected_return_1m': 0,
                'expected_return_3m': 0
            }

    def _generate_detailed_reasoning(self, analysis: Dict[str, Any], ticker: str) -> str:
        """Generate detailed textual reasoning for the recommendation."""
        try:
            momentum = analysis['momentum']
            sentiment = analysis['sentiment']
            risk = analysis['risk']
            targets = analysis['targets']
            recommendation = analysis['recommendation']
            
            reasoning = f"""
## Investment Analysis for {ticker}

### Executive Summary
**Recommendation:** {recommendation['action']} (Confidence: {recommendation['confidence']}%)
**Expected Returns:** 1W: {recommendation['expected_return_1w']}%, 1M: {recommendation['expected_return_1m']}%, 3M: {recommendation['expected_return_3m']}%
**Position Size:** {recommendation['position_size']}

### Price Analysis
- **Current Price:** ${targets['current']:.2f}
- **1-Week Target:** ${targets['target_1w']:.2f} ({recommendation['expected_return_1w']:+.1f}%)
- **1-Month Target:** ${targets['target_1m']:.2f} ({recommendation['expected_return_1m']:+.1f}%)
- **3-Month Target:** ${targets['target_3m']:.2f} ({recommendation['expected_return_3m']:+.1f}%)

### Technical Momentum
- **Momentum Score:** {momentum['momentum_score']:.1f}
- **Trend Strength:** {momentum['trend_strength']:.1f}%
- **Recent Performance:** {momentum.get('recent_momentum', 0):.1f}%
- **Volatility:** {momentum['volatility']:.1f}% (annualized)

### Sentiment Analysis
- **Average Sentiment:** {sentiment['avg_sentiment']:.3f} ({'Bullish' if sentiment['avg_sentiment'] > 0.1 else 'Bearish' if sentiment['avg_sentiment'] < -0.1 else 'Neutral'})
- **Sentiment Trend:** {'Improving' if sentiment['sentiment_trend'] > 0.1 else 'Deteriorating' if sentiment['sentiment_trend'] < -0.1 else 'Stable'}
- **Bullish Signals:** {sentiment.get('bullish_signals', 0)}
- **Bearish Signals:** {sentiment.get('bearish_signals', 0)}

### Risk Assessment
- **Overall Risk Score:** {risk['risk_score']:.1f}/100 ({'Low' if risk['risk_score'] < 30 else 'Medium' if risk['risk_score'] < 60 else 'High'} Risk)
- **Maximum Drawdown:** {risk['max_drawdown']:.1f}%
- **Prediction Accuracy:** {risk['prediction_accuracy']:.1f}%
- **Forecast Uncertainty:** {risk.get('forecast_uncertainty', 0):.1f}%

### Key Insights
"""
            
            # Add specific insights based on the data
            if momentum['trend_strength'] > 70:
                reasoning += "- Strong upward trend momentum supports bullish outlook\n"
            elif momentum['trend_strength'] < 30:
                reasoning += "- Weak trend momentum suggests potential reversal\n"
            
            if sentiment['avg_sentiment'] > 0.3:
                reasoning += "- Highly positive market sentiment provides tailwind\n"
            elif sentiment['avg_sentiment'] < -0.3:
                reasoning += "- Negative sentiment creates headwinds for price appreciation\n"
            
            if risk['risk_score'] < 25:
                reasoning += "- Low risk profile suitable for conservative investors\n"
            elif risk['risk_score'] > 75:
                reasoning += "- High volatility requires careful position sizing\n"
            
            if recommendation['expected_return_3m'] > 15:
                reasoning += "- Strong expected returns justify increased position size\n"
            elif recommendation['expected_return_3m'] < -10:
                reasoning += "- Negative expected returns warrant defensive positioning\n"
            
            reasoning += f"""
### Strategy Recommendation
Given the {recommendation['action']} recommendation with {recommendation['confidence']}% confidence, 
investors should consider {recommendation['position_size'].lower()} based on current market conditions 
and risk tolerance. The risk-adjusted score of {recommendation['risk_adjusted_score']} suggests 
{'favorable' if recommendation['risk_adjusted_score'] > 60 else 'unfavorable' if recommendation['risk_adjusted_score'] < 40 else 'neutral'} 
risk-return profile.

### Disclaimers
- This analysis is based on historical data and predicted prices
- Past performance does not guarantee future results
- Consider your personal risk tolerance and investment objectives
- Consult with a financial advisor before making investment decisions
"""
            
            return reasoning.strip()
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            return f"Error generating detailed analysis for {ticker}. Please review the data manually."

    def analyze_predictions(self, 
                          prediction_df: pd.DataFrame, 
                          ticker: str,
                          strategy: str = 'balanced',
                          save_analysis: bool = True) -> Dict[str, Any]:
        """
        Analyze prediction data and generate investment recommendations.
        
        Args:
            prediction_df: DataFrame with prediction data
            ticker: Stock ticker symbol
            strategy: Investment strategy ('growth', 'value', 'momentum', 'contrarian', 'balanced')
            save_analysis: Whether to save analysis to file
            
        Returns:
            Dictionary containing comprehensive analysis and recommendations
        """
        try:
            logger.info(f"Analyzing predictions for {ticker} using {strategy} strategy")
            
            # Perform comprehensive analysis
            momentum_analysis = self._calculate_price_momentum(prediction_df)
            sentiment_analysis = self._analyze_sentiment_trend(prediction_df)
            risk_analysis = self._calculate_risk_metrics(prediction_df)
            price_targets = self._generate_price_targets(prediction_df)
            
            # Compile analysis
            analysis = {
                'ticker': ticker,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'momentum': momentum_analysis,
                'sentiment': sentiment_analysis,
                'risk': risk_analysis,
                'targets': price_targets
            }
            
            # Generate recommendation
            recommendation = self._determine_investment_action(analysis, strategy)
            analysis['recommendation'] = recommendation
            
            # Generate detailed reasoning
            detailed_reasoning = self._generate_detailed_reasoning(analysis, ticker)
            analysis['detailed_reasoning'] = detailed_reasoning
            
            # Save analysis if requested
            if save_analysis:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{ticker}_analysis_{strategy}_{timestamp}.json"
                filepath = self.data_analysis / filename
                
                with open(filepath, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                
                # Also save reasoning as text file
                reasoning_filename = f"{ticker}_reasoning_{strategy}_{timestamp}.md"
                reasoning_filepath = self.data_analysis / reasoning_filename
                with open(reasoning_filepath, 'w') as f:
                    f.write(detailed_reasoning)
                
                logger.info(f"Analysis saved to {filepath}")
                logger.info(f"Reasoning saved to {reasoning_filepath}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing predictions for {ticker}: {str(e)}")
            raise

    def run(self, ticker: str, prediction_file: Optional[str] = None, strategy: str = 'balanced') -> Dict[str, Any]:
        """
        Main method to run the reasoning analysis.
        
        Args:
            ticker: Stock ticker symbol
            prediction_file: Path to prediction CSV file (optional, will search if not provided)
            strategy: Investment strategy to use
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load prediction data
            if prediction_file:
                df = pd.read_csv(prediction_file)
            else:
                # Find the most recent prediction file for the ticker
                prediction_files = list(self.data_predictions.glob(f"{ticker}_prophet_predictions.csv"))
                if not prediction_files:
                    raise FileNotFoundError(f"No prediction files found for {ticker}")
                
                # Use the most recent file
                prediction_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(prediction_file)
                logger.info(f"Loaded prediction data from {prediction_file}")
            
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Run analysis
            analysis = self.analyze_predictions(df, ticker, strategy)
            
            # Print summary
            rec = analysis['recommendation']
            print(f"\n{'='*60}")
            print(f"INVESTMENT ANALYSIS FOR {ticker}")
            print(f"{'='*60}")
            print(f"Recommendation: {rec['action']} (Confidence: {rec['confidence']}%)")
            print(f"Expected Returns: 1W: {rec['expected_return_1w']}%, 1M: {rec['expected_return_1m']}%, 3M: {rec['expected_return_3m']}%")
            print(f"Position Size: {rec['position_size']}")
            print(f"Risk-Adjusted Score: {rec['risk_adjusted_score']}/100")
            print(f"{'='*60}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in reasoning agent run: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    agent = ReasoningAgent()
    
    # Interactive mode
    print("\n=== Investment Reasoning Agent ===")
    print("Analyze stock predictions and get investment recommendations\n")
    
    while True:
        ticker = input("Enter ticker symbol (or 'quit' to exit): ").strip().upper()
        if ticker == 'QUIT':
            break
            
        strategy = input("Enter strategy (growth/value/momentum/contrarian/balanced) [balanced]: ").strip().lower()
        if not strategy:
            strategy = 'balanced'
        
        try:
            analysis = agent.run(ticker, strategy=strategy)
            
            # Show detailed reasoning
            show_details = input("Show detailed reasoning? (y/n) [n]: ").strip().lower()
            if show_details == 'y':
                print("\n" + analysis['detailed_reasoning'])
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "-"*60 + "\n")