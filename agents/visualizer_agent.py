#!/usr/bin/env python3
"""
Visualizer agent for creating charts from the stock price predictions.

This module generates two standalone Plotly figures from the most recent
stock prediction data and saves them as PNG files in the reports directory.
"""

import os
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualizerAgent:
    """
    Agent for creating and saving visualizations of stock price predictions.
    """
    
    def __init__(self, data_dir: str = None, report_dir: str = None):
        """
        Initialize the visualizer agent.
        
        Args:
            data_dir: Path to directory containing prediction CSV files
            report_dir: Path to directory for saving chart PNGs
        """
        # Set default paths relative to the project root
        project_root = Path(__file__).parent.parent
        
        if data_dir is None:
            self.data_dir = project_root / "data_predictions"
        else:
            self.data_dir = Path(data_dir)
            
        if report_dir is None:
            self.report_dir = project_root / "reports"
        else:
            self.report_dir = Path(report_dir)
            
        # Create reports directory if it doesn't exist
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Report directory: {self.report_dir}")
    
    def load_data(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Load the prediction file for the specified ticker, or the most recent if not specified.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA'). If None, load the most recent file.
        
        Returns:
            pd.DataFrame: Loaded prediction data with proper date parsing
        """
        if ticker:
            file_path = self.data_dir / f"{ticker}_prophet_predictions.csv"
            if not file_path.exists():
                error_msg = f"Prediction file for ticker '{ticker}' not found at: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            logger.info(f"Loading prediction file for {ticker}: {file_path}")
            df = pd.read_csv(file_path)
        else:
            # Fallback: most recent file
            file_pattern = str(self.data_dir / "*_prophet_predictions.csv")
            files = glob.glob(file_pattern)
            if not files:
                error_msg = f"No prediction files found matching: {file_pattern}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            latest_file = max(files, key=os.path.getmtime)
            ticker = Path(latest_file).name.split('_')[0]
            logger.info(f"Loading most recent prediction file for {ticker}: {latest_file}")
            df = pd.read_csv(latest_file)
        
        # Parse the date column and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    def is_valid_figure(self, fig) -> bool:
        """Check if the object is a valid Plotly Figure."""
        import plotly.graph_objects as go
        return isinstance(fig, go.Figure)

    def plot_price(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Generate a line chart showing actual vs. forecasted prices.
        
        Args:
            df: DataFrame with price and forecast data
            
        Returns:
            go.Figure: Plotly figure with price and forecast lines
        """
        required_cols = {'Date', 'Actual_Close', 'Predicted_Close', 'Period', 'news_sentiment_score'}
        if df is None or df.empty or not required_cols.issubset(df.columns):
            logger.warning("plot_price: Input DataFrame is empty or missing required columns. Returning None.")
            return None
        
        # Get the ticker symbol from the DataFrame directly if it's being called from the orchestrator
        # This ensures we use the correct ticker from the actual data being plotted
        
        # Try to extract ticker from the filename columns if present
        if 'ticker' in df.columns:
            ticker = df['ticker'].iloc[0]
        else:
            # Fallback: Check prediction files and extract the ticker from the current dataframe
            # by finding which ticker's data it most closely matches
            prediction_files = glob.glob(str(self.data_dir / "*_prophet_predictions.csv"))
            
            # If no prediction files, use a generic name
            if not prediction_files:
                ticker = "Stock"
            else:
                # Get the prediction file that matches this dataframe best
                latest_file = max(prediction_files, key=os.path.getmtime)
                ticker = Path(latest_file).name.split('_')[0]
            
        # For orchestrator, get ticker directly from the prediction DataFrame filename in orchestrator
        # The prediction_df is created from a file named "{ticker}_prophet_predictions.csv"
        
        logger.info(f"Creating price chart for ticker: {ticker}")
        
        # Create the figure
        fig = go.Figure()
        
        # Create a proper date string column that preserves ordering
        df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Add actual close price (navy line)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],  # Use datetime objects directly
                y=df['Actual_Close'],
                mode='lines',
                line=dict(color='#003366', width=2),
                name='Actual Close'
            )
        )
        
        # Add predicted close price (teal dashed line)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],  # Use datetime objects directly
                y=df['Predicted_Close'],
                mode='lines',
                line=dict(color='#009688', width=2, dash='dash'),
                name='Predicted Close'
            )
        )
        
        # Add period shading using datetime objects
        training_data = df[df['Period'] == 'Training']
        test_data = df[df['Period'] == 'Test']
        forecast_data = df[df['Period'] == 'Forecast']
        
        # Add the shading for the different periods using datetime objects
        if not training_data.empty:
            min_date = training_data['Date'].min()
            max_date = training_data['Date'].max()
            fig.add_vrect(
                x0=min_date,
                x1=max_date,
                fillcolor="rgba(200, 200, 200, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="Training",
                annotation_position="top left",
            )
        
        if not test_data.empty:
            min_date = test_data['Date'].min()
            max_date = test_data['Date'].max()
            fig.add_vrect(
                x0=min_date,
                x1=max_date,
                fillcolor="rgba(135, 206, 250, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="Test",
                annotation_position="top left",
            )
        
        if not forecast_data.empty:
            min_date = forecast_data['Date'].min()
            max_date = forecast_data['Date'].max()
            fig.add_vrect(
                x0=min_date,
                x1=max_date,
                fillcolor="rgba(144, 238, 144, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="Forecast",
                annotation_position="top left",
            )
        
        # Add significant sentiment annotations
        significant_sentiment = df[abs(df['news_sentiment_score']) > 0.5]
        
        if not significant_sentiment.empty:
            fig.add_trace(
                go.Scatter(
                    x=significant_sentiment['Date'],  # Use datetime objects directly
                    y=significant_sentiment['Actual_Close'].fillna(
                        significant_sentiment['Predicted_Close']),
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=significant_sentiment['news_sentiment_score'].apply(
                            lambda x: 'green' if x > 0 else 'red'),
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    text=significant_sentiment['news_sentiment_score'].round(2).astype(str),
                    hovertext=significant_sentiment['news_text'],
                    hoverinfo='text',
                    name='Significant Sentiment'
                )
            )
        
        # Set layout
        fig.update_layout(
            title=f"{ticker} - Actual vs Forecasted Close Price",
            xaxis=dict(
                title="Date", 
                gridcolor='lightgray',
                type='date',  # Explicitly set the axis type to date
                tickformat='%Y-%m-%d'  # Format the tick labels as YYYY-MM-DD
            ),
            yaxis=dict(title="Price ($)", gridcolor='lightgray'),
            plot_bgcolor='white',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode="x unified"
        )
        
        # Add forecast start line using scatter trace method
        if not forecast_data.empty:
            forecast_start = forecast_data['Date'].min()
            y_min = df['Predicted_Close'].min() * 0.95
            y_max = df['Predicted_Close'].max() * 1.05
            
            fig.add_trace(
                go.Scatter(
                    x=[forecast_start, forecast_start],
                    y=[y_min, y_max],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    name='Forecast Start',
                    showlegend=False
                )
            )
            
            # Add a text annotation using update_layout for annotations
            fig.update_layout(
                annotations=[
                    dict(
                        x=forecast_start,
                        y=y_max,
                        text="Forecast Start",
                        showarrow=False,
                        yshift=10
                    )
                ]
            )
        
        return fig
    
    def plot_sentiment_dist(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Generate a histogram showing distribution of sentiment scores.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            go.Figure: Plotly histogram figure of sentiment distribution
        """
        if df is None or df.empty or 'news_sentiment_score' not in df.columns:
            logger.warning("plot_sentiment_dist: Input DataFrame is empty or missing 'news_sentiment_score'. Returning None.")
            return None
        
        # Filter out rows with missing sentiment scores
        df_sentiment = df[~df['news_sentiment_score'].isna()]
        
        # Create color mapping function for the histogram
        def get_color(val):
            if val < 0:
                return 'rgba(255, 150, 150, 0.6)'  # Soft red
            elif val == 0:
                return 'rgba(180, 180, 180, 0.6)'  # Gray
            else:
                return 'rgba(150, 255, 150, 0.6)'  # Soft green
        
        # Create bins for the histogram
        bin_size = 0.1
        bins = [-1.0 + i * bin_size for i in range(int(2.0 / bin_size) + 1)]
        
        # Create histogram figure
        fig = px.histogram(
            df_sentiment, 
            x='news_sentiment_score',
            nbins=20,
            title='Distribution of News Sentiment Scores',
            labels={'news_sentiment_score': 'Sentiment Score', 'count': 'Frequency'},
            color_discrete_sequence=['rgba(180, 180, 180, 0.6)']
        )
        
        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                title='Sentiment Score',
                tickmode='array',
                tickvals=[-1.0, -0.5, 0, 0.5, 1.0],
                ticktext=['-1.0 (Very Negative)', '-0.5', 'Neutral', '0.5', '1.0 (Very Positive)'],
                gridcolor='lightgray'
            ),
            yaxis=dict(title='Frequency', gridcolor='lightgray'),
            bargap=0.05
        )
        
        # Replace the default trace with our custom colored bars
        if not df_sentiment.empty:
            fig.data = []
            for bin_start in [-1.0, -0.5, 0, 0.5]:
                bin_end = bin_start + 0.5
                mask = (df_sentiment['news_sentiment_score'] >= bin_start) & \
                       (df_sentiment['news_sentiment_score'] < bin_end)
                
                if bin_start < 0:
                    color = 'rgba(255, 150, 150, 0.6)'  # Soft red
                    name = 'Negative'
                elif bin_start == 0:
                    color = 'rgba(180, 180, 180, 0.6)'  # Gray
                    name = 'Neutral'
                else:
                    color = 'rgba(150, 255, 150, 0.6)'  # Soft green
                    name = 'Positive'
                
                sub_df = df_sentiment[mask]
                
                # Only add bins that have data
                if not sub_df.empty:
                    fig.add_trace(
                        go.Histogram(
                            x=sub_df['news_sentiment_score'],
                            name=name,
                            marker_color=color,
                            xbins=dict(start=bin_start, end=bin_end, size=0.1),
                            showlegend=True
                        )
                    )
        
        # Add vertical line for median
        if not df_sentiment.empty:
            median_sentiment = float(df_sentiment['news_sentiment_score'].median())
            fig.add_vline(
                x=median_sentiment, 
                line_width=2, 
                line_dash="dash", 
                line_color="black",
                annotation_text=f"Median: {median_sentiment:.2f}",
                annotation_position="top right"
            )
        
        return fig
    
    def save_charts(self, fig_price: go.Figure, fig_sent: go.Figure) -> tuple:
        """
        Save the generated figures as PNG files in the reports directory.
        
        Args:
            fig_price: Figure object for price chart
            fig_sent: Figure object for sentiment distribution chart
            
        Returns:
            tuple: Paths to the saved PNG files
        """
        # Get the ticker symbol
        ticker = next((file.split('_')[0] for file in os.listdir(self.data_dir) 
                     if file.endswith('_prophet_predictions.csv')), "Stock")
        
        # Generate filenames with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        price_filename = f"{ticker}_price_chart_{timestamp}.png"
        sent_filename = f"{ticker}_sentiment_dist_{timestamp}.png"
        
        price_path = self.report_dir / price_filename
        sent_path = self.report_dir / sent_filename
        
        # Save the figures
        try:
            fig_price.write_image(str(price_path), width=1200, height=800)
            logger.info(f"Price chart saved to {price_path}")
            
            fig_sent.write_image(str(sent_path), width=1000, height=600)
            logger.info(f"Sentiment distribution chart saved to {sent_path}")
            
            return str(price_path), str(sent_path)
        except Exception as e:
            logger.error(f"Error saving charts: {str(e)}")
            
            # Try to save as interactive HTML if PNG export fails
            try:
                price_html = self.report_dir / f"{ticker}_price_chart_{timestamp}.html"
                sent_html = self.report_dir / f"{ticker}_sentiment_dist_{timestamp}.html"
                
                fig_price.write_html(str(price_html))
                fig_sent.write_html(str(sent_html))
                
                logger.info(f"Saved interactive HTML versions instead at {price_html} and {sent_html}")
                return str(price_html), str(sent_html)
            except Exception as e2:
                logger.error(f"Error saving HTML versions: {str(e2)}")
                raise


if __name__ == "__main__":
    # Create and run the visualizer agent
    try:
        agent = VisualizerAgent()
        df = agent.load_data()
        fig_price = agent.plot_price(df)
        fig_sent = agent.plot_sentiment_dist(df)
        saved_paths = agent.save_charts(fig_price, fig_sent)
        
        logger.info("Visualization complete! Charts saved successfully.")
    except Exception as e:
        logger.error(f"Error running visualizer agent: {str(e)}")