"""
Data fetching module for QQQ historical data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching and preprocessing of QQQ historical data"""
    
    def __init__(self, symbol: str = "QQQ"):
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
            logger.error(f"Error fetching data: {e}")
            raise
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        df = data.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Price deviation from mean
        df['price_deviation'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
        if data['close'].min() <= 0 or data['close'].max() > 1000:
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
                'symbol': 'QQQ',
                'timeframe': '1h',
                'start_date': '2020-01-01',
                'end_date': '2024-01-01'
            }
        }
