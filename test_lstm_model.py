"""Test the PredictorAgent LSTM model functionality."""
import logging
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from models.lstm_model import PredictorAgent
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPredictorAgent(unittest.TestCase):
    """Test cases for PredictorAgent."""
    
    def setUp(self):
        """Initialize test environment."""
        self.agent = PredictorAgent(lookback=30, forecast_days=20)  # Reduced lookback, increased forecast
        self.ticker = "NVDA"
        
    def test_data_loading(self):
        """Test data loading functionality."""
        df = self.agent._load_data(self.ticker)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        logger.info(f"Loaded data shape: {df.shape}")
        
    def test_data_scaling(self):
        """Test data scaling functionality."""
        df = self.agent._load_data(self.ticker)
        scaled_data, dates, columns = self.agent._scale_data(df)
        
        self.assertIsInstance(scaled_data, np.ndarray)
        self.assertIsInstance(dates, pd.Series)
        self.assertTrue(len(columns) > 0)
        logger.info(f"Scaled data shape: {scaled_data.shape}")
        logger.info(f"Number of features: {len(columns)}")
        
    def test_sequence_creation(self):
        """Test sequence creation for LSTM."""
        df = self.agent._load_data(self.ticker)
        scaled_data, _, _ = self.agent._scale_data(df)
        X, y = self.agent._prepare_sequences(scaled_data)
        
        self.assertEqual(X.shape[1], self.agent.lookback)
        self.assertEqual(X.shape[2], scaled_data.shape[1])
        self.assertEqual(X.shape[0], y.shape[0])
        logger.info(f"LSTM input shape: {X.shape}")
        logger.info(f"LSTM target shape: {y.shape}")
        
    def test_full_pipeline(self):
        """Test the complete prediction pipeline."""
        try:
            mae, rmse = self.agent.run(self.ticker)
            self.assertIsInstance(mae, float)
            self.assertIsInstance(rmse, float)
            
            # Check if model files were saved
            model_path = self.agent.models_saved / "lstm_model_latest.pth"
            model_best_path = self.agent.models_saved / "lstm_model_best.pth"
            scaler_path = self.agent.models_saved / "feature_scaler.pkl"
            target_scaler_path = self.agent.models_saved / "target_scaler.pkl"
            
            self.assertTrue(model_path.exists())
            self.assertTrue(model_best_path.exists())
            self.assertTrue(scaler_path.exists())
            self.assertTrue(target_scaler_path.exists())
            
            # Check if predictions file was created
            pred_path = self.agent.data_predictions / f"{self.ticker}_lstm_predictions.csv"
            self.assertTrue(pred_path.exists())
            
            # Load predictions and validate format
            predictions = pd.read_csv(pred_path)
            self.assertIn('Date', predictions.columns)
            self.assertIn('Actual_Close', predictions.columns)
            self.assertIn('Predicted_Close', predictions.columns)
            self.assertIn('news_sentiment_score', predictions.columns)
            
            # Validate predictions structure
            historical_data = predictions[predictions['Actual_Close'].notna()]
            self.assertGreater(len(historical_data), 0, "No historical actual prices found")
            
            test_predictions = historical_data[historical_data['Predicted_Close'].notna()]
            self.assertGreater(len(test_predictions), 0, "No test predictions found")
            
            future_predictions = predictions[predictions['Actual_Close'].isna()]
            self.assertEqual(len(future_predictions), self.agent.forecast_days,
                           "Incorrect number of future predictions")
            
            # Calculate and log metrics
            logger.info("\n" + "="*50)
            logger.info("Model Performance Metrics:")
            logger.info(f"Mean Absolute Error (MAE): ${mae:.2f}")
            logger.info(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
            
            # Validate sentiment scores
            self.assertTrue(all(predictions['news_sentiment_score'].notna()),
                          "Missing news sentiment scores")
            
            # Check metrics file was created
            metric_files = list(self.agent.data_metrics.glob(f"{self.ticker}_metrics_*.json"))
            self.assertTrue(len(metric_files) > 0, "No metrics file created")
            
            logger.info("\nTest completed successfully!")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {str(e)}")
            raise
            
    def test_prediction_accuracy(self):
        """Test prediction accuracy and calculate percentage metrics."""
        # Run the model
        mae, rmse = self.agent.run(self.ticker)
        
        # Load predictions
        pred_path = self.agent.data_predictions / f"{self.ticker}_lstm_predictions.csv"
        predictions = pd.read_csv(pred_path)
        
        # Calculate percentage metrics
        test_data = predictions.dropna(subset=['Actual_Close', 'Predicted_Close'])
        mean_price = test_data['Actual_Close'].mean()
        
        mae_pct = (mae / mean_price) * 100
        rmse_pct = (rmse / mean_price) * 100
        
        # Log percentage metrics
        logger.info("\n" + "="*50)
        logger.info("Percentage Error Metrics:")
        logger.info(f"MAE as % of mean price: {mae_pct:.2f}%")
        logger.info(f"RMSE as % of mean price: {rmse_pct:.2f}%")
        logger.info("Mean price: ${:.2f}".format(mean_price))
        logger.info("="*50)
        
        # Additional accuracy checks
        mape = np.mean(np.abs((test_data['Actual_Close'] - test_data['Predicted_Close']) / 
                             test_data['Actual_Close'])) * 100
        logger.info(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Directional accuracy
        actual_direction = np.sign(test_data['Actual_Close'].diff())
        pred_direction = np.sign(test_data['Predicted_Close'].diff())
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Assert reasonable accuracy
        self.assertLess(mae_pct, 15.0, "MAE percentage too high")
        self.assertLess(rmse_pct, 20.0, "RMSE percentage too high")
        self.assertGreater(directional_accuracy, 50.0, "Directional accuracy below random chance")
            
if __name__ == '__main__':
    unittest.main()