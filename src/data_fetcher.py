"""
Data fetching module for TQQQ historical data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching and preprocessing of TQQQ historical data"""
    
    def __init__(self, symbol: str = "TQQQ"):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def fetch_data(
        self, 
        start_date: str, 
        end_date: str, 
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch historical data for the symbol
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1h, 1d, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {self.symbol} data from {start_date} to {end_date}")
            
            data = self.ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            logger.warning(f"Yahoo Finance failed: {e}")
            logger.info("Falling back to sample data...")
            return self._load_sample_data(start_date, end_date)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data"""
        df = data.copy()
        
        # Volume ratio for volume confirmation
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def get_latest_price(self) -> float:
        """Get the latest price for the symbol"""
        try:
            latest_data = self.ticker.history(period="1d", interval="1m")
            if not latest_data.empty:
                return latest_data['Close'].iloc[-1]
            else:
                raise ValueError("No recent data available")
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            raise
    
    def _load_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load sample data when Yahoo Finance is unavailable"""
        try:
            # Try to load the sample data we created
            sample_file = f'sample_{self.symbol.lower()}_data.csv'
            sample_data = pd.read_csv(sample_file, index_col=0, parse_dates=True)
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            filtered_data = sample_data[(sample_data.index >= start) & (sample_data.index <= end)]
            
            if filtered_data.empty:
                logger.warning("Sample data doesn't cover requested date range, using all available data")
                return sample_data
            
            logger.info(f"Loaded {len(filtered_data)} sample records")
            return filtered_data
            
        except FileNotFoundError:
            logger.error(f"Sample data file not found. Please run 'python create_sample_data.py' first")
            raise ValueError("No data available - Yahoo Finance failed and no sample data found")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the quality of fetched data"""
        if data.empty:
            return False
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.1:  # More than 10% missing
            logger.warning(f"High percentage of missing data: {missing_pct:.2%}")
            return False
        
        # Check for reasonable price ranges
        if data['close'].min() <= 0 or data['close'].max() > 2000:
            logger.warning("Price data seems unreasonable")
            return False
        
        return True


def load_config() -> dict:
    """Load configuration from YAML file"""
    import yaml
    try:
        with open('config/strategy_config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        return {
            'data': {
                'symbol': 'TQQQ',
                'timeframe': '1d',
                'start_date': '2018-01-01',
                'end_date': '2025-01-01'
            }
        }
