"""
LSTM-based stock price predictor with improved accuracy.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """Improved LSTM model with residual connections."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        
        # Residual connection
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm_out = lstm1_out + lstm2_out
        
        # Get last output and apply dropout
        last_output = self.dropout(lstm_out[:, -1, :])
        
        # Fully connected layers with ReLU
        fc1_out = F.relu(self.fc1(last_output))
        return self.fc2(fc1_out)

class PredictorAgent:
    """Agent for training LSTM model and predicting stock prices."""
    
    def __init__(self, lookback: int = 20, forecast_days: int = 10):
        """Initialize predictor agent with hyperparameters."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = lookback
        self.forecast_days = forecast_days
        
        # Optimized hyperparameters
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.1
        self.learning_rate = 0.0005
        self.batch_size = 16
        self.epochs = 150
        
        # Training parameters
        self.validation_split = 0.15
        self.test_size = 0.15
        
        # Create directories
        self.base_path = Path(".")
        self.models_saved = self.base_path / "models_saved"
        self.data_predictions = self.base_path / "data_predictions"
        self.data_metrics = self.base_path / "data_metrics"
        
        for path in [self.models_saved, self.data_predictions, self.data_metrics]:
            path.mkdir(exist_ok=True)
        
        # Scaler instances
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def _load_data(self, ticker: str) -> pd.DataFrame:
        """Load and preprocess data for the given ticker."""
        data_sentiment = self.base_path / "data" / "sentiment_file"
        sentiment_files = list(data_sentiment.glob(f"{ticker}_*with_sentiment_scored.csv"))
        if not sentiment_files:
            raise FileNotFoundError(f"No sentiment scored files found for {ticker}")
            
        latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading data from {latest_file}")
        
        df = pd.read_csv(latest_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Select only essential features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'RSI_14',
            'news_sentiment_score'
        ]
        
        # Select available columns
        available_columns = ['Date'] + [col for col in feature_columns if col in df.columns]
        df = df[available_columns]
        
        # Forward fill missing values
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].ffill().fillna(0)
        
        # Calculate daily returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df = df.fillna(0)
        
        return df
        
    def _scale_data(self, df: pd.DataFrame) -> tuple:
        """Scale features and target variables separately."""
        # Extract close price and convert to numpy array
        close_price = df['Close'].to_numpy().reshape(-1, 1)
        
        # Select features excluding Date and Close
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        feature_data = df[feature_cols].to_numpy()
        dates = df['Date']
        
        # Scale Close price separately
        self.target_scaler.fit(close_price)
        scaled_close = self.target_scaler.transform(close_price)
        
        # Scale features
        self.feature_scaler.fit(feature_data)
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Combine scaled data with close price first
        scaled_data = np.hstack([scaled_close, scaled_features])
        
        return scaled_data, dates, [col for col in df.columns if col != 'Date']
        
    def _prepare_sequences(self, data: np.ndarray) -> tuple:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            # Get sequence and calculate returns
            sequence = data[i:i + self.lookback]
            target = data[i + self.lookback, 0]  # Close price is first column
            
            # Calculate returns as percentage change
            close_prices = sequence[:, 0]
            returns = np.zeros_like(close_prices)
            returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
            
            # Create sequence with technical features
            sequence_with_features = np.column_stack([
                sequence,
                returns.reshape(-1, 1)  # Add returns as a new feature
            ])
            
            X.append(sequence_with_features)
            y.append(target)
        
        return np.array(X), np.array(y).reshape(-1, 1)
        
    def _create_train_test_split(self, X: np.ndarray, y: np.ndarray, dates: pd.Series) -> tuple:
        """Split data into training and test sets chronologically."""
        total_samples = len(X)
        test_size = int(total_samples * self.test_size)
        val_size = int((total_samples - test_size) * self.validation_split)
        train_size = total_samples - test_size - val_size
        
        # Split indices
        train_idx = slice(0, train_size)
        val_idx = slice(train_size, train_size + val_size)
        test_idx = slice(train_size + val_size, None)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X[train_idx], y[train_idx])
        val_dataset = TimeSeriesDataset(X[val_idx], y[val_idx])
        test_dataset = TimeSeriesDataset(X[test_idx], y[test_idx])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size
        )
        
        test_dates = dates[train_size + val_size + self.lookback:]
        
        return train_loader, val_loader, test_loader, test_dates
        
    def _train_model(self, model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, ticker: str) -> nn.Module:
        """Train the LSTM model."""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        mse_criterion = nn.MSELoss()
        
        def directional_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # MSE component
            mse = F.mse_loss(y_pred, y_true)
            
            # Directional component
            pred_diff = torch.diff(y_pred.squeeze())
            true_diff = torch.diff(y_true.squeeze())
            
            # Calculate directional accuracy
            correct_direction = ((pred_diff * true_diff) > 0).float()
            dir_loss = 1.0 - correct_direction.mean()
            
            # Magnitude-aware component for wrong predictions
            wrong_direction = (correct_direction == 0)
            if wrong_direction.any():
                magnitude_penalty = (torch.abs(pred_diff[wrong_direction]) * 
                                  torch.abs(true_diff[wrong_direction])).mean()
            else:
                magnitude_penalty = torch.tensor(0.0).to(y_pred.device)
            
            # Combined loss with higher weight on directional component
            return 0.3 * mse + 0.5 * dir_loss + 0.2 * magnitude_penalty
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_dir_acc = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                # Calculate loss with directional component
                loss = directional_loss(y_pred, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_dir_acc = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    y_pred = model(X_batch)
                    val_batch_loss = directional_loss(y_pred, y_batch)
                    
                    # Calculate directional accuracy
                    pred_diff = torch.diff(y_pred.squeeze())
                    true_diff = torch.diff(y_batch.squeeze())
                    correct_dirs = ((pred_diff * true_diff) > 0).float().mean()
                    
                    val_loss += val_batch_loss.item()
                    val_dir_acc += correct_dirs.item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches
            val_dir_acc = (val_dir_acc / n_val_batches) * 100
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                torch.save(best_model_state, self.models_saved / "lstm_model_best.pth")
            else:
                patience_counter += 1
                
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"Val Dir Acc: {val_dir_acc:.2f}%")
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model
        
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> tuple:
        """Evaluate model performance."""
        model.eval()
        predictions = []
        actual_values = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = model(X_batch)
                
                # Move to CPU and convert to numpy
                y_pred = y_pred.cpu().numpy()
                y_batch = y_batch.cpu().numpy()
                
                # Inverse transform predictions
                y_pred = self.target_scaler.inverse_transform(y_pred)
                y_batch = self.target_scaler.inverse_transform(y_batch)
                
                predictions.extend(y_pred.flatten())
                actual_values.extend(y_batch.flatten())
        
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        mae = np.mean(np.abs(actual_values - predictions))
        rmse = np.sqrt(np.mean((actual_values - predictions) ** 2))
        
        return mae, rmse, predictions, actual_values
        
    def _forecast_future(self, model: nn.Module, last_sequence: np.ndarray) -> np.ndarray:
        """Generate future predictions."""
        model.eval()
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(self.forecast_days):
                sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                pred = model(sequence_tensor).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = np.array([pred] + [0] * (current_sequence.shape[1] - 1))
        
        # Inverse transform predictions
        predictions_array = self.target_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions_array
        
    def _save_predictions(self, historical_dates: np.ndarray, historical_actual: np.ndarray, 
                         historical_pred: np.ndarray, future_dates: list, 
                         future_pred: np.ndarray, ticker: str):
        """
        Save both historical and future predictions with sentiment scores.
        
        Args:
            historical_dates (np.ndarray): Dates for historical predictions
            historical_actual (np.ndarray): Actual historical prices
            historical_pred (np.ndarray): Predicted historical prices
            future_dates (list): Dates for future predictions
            future_pred (np.ndarray): Future price predictions
            ticker (str): Stock ticker symbol
        """
        # First, get all historical data with actual close prices and sentiment scores
        df_original = self._load_data(ticker)
        df_original = df_original[['Date', 'Close', 'news_sentiment_score']]
        df_original = df_original.rename(columns={'Close': 'Actual_Close'})
        df_original['Date'] = pd.to_datetime(df_original['Date'])
        
        # Initialize predictions DataFrame with None values
        df_original['Predicted_Close'] = pd.Series(dtype=float)
        
        # Add predictions for test period
        test_start_idx = len(historical_dates) - len(historical_actual)
        test_dates = pd.to_datetime(historical_dates[test_start_idx:])
        for date, pred in zip(test_dates, historical_pred.flatten()):
            mask = df_original['Date'] == date
            if any(mask):
                df_original.loc[mask, 'Predicted_Close'] = pred
        
        # Prepare future predictions with explicitly defined dtypes
        future_df = pd.DataFrame({
            'Date': pd.to_datetime(future_dates),
            'Actual_Close': pd.Series([np.nan] * len(future_pred), dtype=float),
            'Predicted_Close': pd.Series(future_pred, dtype=float),
            'news_sentiment_score': pd.Series([df_original['news_sentiment_score'].iloc[-1]] * len(future_pred), dtype=float)
        })
        
        # Combine historical and future data with consistent dtypes
        results_df = pd.concat([df_original, future_df], ignore_index=True)
        results_df = results_df.sort_values('Date')
        
        # Save to CSV
        output_file = self.data_predictions / f"{ticker}_lstm_predictions.csv"
        results_df.to_csv(output_file, index=False, date_format='%Y-%m-%d')
        logger.info(f"Saved predictions to {output_file}")
        
    def run(self, ticker: str = "NVDA"):
        """
        Load processed data, build LSTM, train, evaluate, and save predictions and model.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            tuple: (mae, rmse) evaluation metrics
        """
        try:
            # Load and preprocess data
            df = self._load_data(ticker)
            scaled_data, dates, feature_cols = self._scale_data(df)
            
            # Create sequences
            X, y = self._prepare_sequences(scaled_data)
            
            # Split data
            train_loader, val_loader, test_loader, test_dates = self._create_train_test_split(X, y, dates)
            
            # Initialize model
            input_size = X.shape[2]  # Number of features
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            
            # Train model
            logger.info("Starting model training...")
            model = self._train_model(model, train_loader, val_loader, ticker)
            
            # Evaluate model
            logger.info("Evaluating model...")
            mae, rmse, predictions, actual_values = self._evaluate_model(model, test_loader)
            logger.info(f"Test MAE: {mae:.2f}")
            logger.info(f"Test RMSE: {rmse:.2f}")
            
            # Generate future predictions
            logger.info("Generating future predictions...")
            last_sequence = X[-1]
            future_pred = self._forecast_future(model, last_sequence)
            
            # Generate future dates
            last_date = dates.iloc[-1]
            future_dates = [
                last_date + timedelta(days=i+1)
                for i in range(self.forecast_days)
            ]
            
            # Save predictions to DataFrame
            df_pred = pd.DataFrame({
                'Date': pd.concat([test_dates, pd.Series(future_dates)]),
                'Actual_Close': pd.concat([
                    pd.Series(actual_values), 
                    pd.Series([None] * len(future_dates))
                ]),
                'Predicted_Close': pd.concat([
                    pd.Series(predictions),
                    pd.Series(future_pred)
                ])
            })
            
            # Save to CSV
            output_file = self.data_predictions / f"{ticker}_lstm_predictions.csv"
            df_pred.to_csv(output_file, index=False)
            
            # Save model metrics
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'num_features': int(input_size),
                'lookback': int(self.lookback),
                'forecast_days': int(self.forecast_days)
            }
            
            metrics_file = self.data_metrics / f"{ticker}_metrics_{metrics['timestamp']}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save models and scalers
            torch.save(model.state_dict(), self.models_saved / "lstm_model_latest.pth")
            joblib.dump(self.feature_scaler, self.models_saved / "feature_scaler.pkl")
            joblib.dump(self.target_scaler, self.models_saved / "target_scaler.pkl")
            
            return mae, rmse
            
        except Exception as e:
            logger.error(f"Error in model pipeline: {str(e)}")
            raise