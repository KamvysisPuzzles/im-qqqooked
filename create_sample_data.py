"""
Create sample QQQ data for testing when Yahoo Finance is down
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_qqq_data(start_date="2023-01-01", end_date="2023-12-31"):
    """Create realistic sample QQQ data for testing"""
    
    # Create date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='H')
    
    # Remove weekends (keep only weekdays)
    dates = dates[dates.weekday < 5]
    
    # Create realistic price data
    np.random.seed(42)  # For reproducible results
    
    # Start with QQQ-like price
    base_price = 350.0
    prices = [base_price]
    
    # Generate realistic price movements
    for i in range(1, len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0.0001, 0.01)  # Small hourly changes
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 50))  # Don't go below $50
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices[:len(dates)]
    
    # Generate realistic OHLC from close prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(data)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(data)))
    data['volume'] = np.random.randint(1000000, 5000000, len(data))
    
    # Add technical indicators
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    bb_std = data['close'].rolling(window=20).std()
    data['bb_middle'] = data['sma_20']
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Price deviation and volatility
    data['price_deviation'] = (data['close'] - data['sma_20']) / data['sma_20']
    data['volatility'] = data['close'].rolling(window=20).std()
    
    # Clean up
    data = data.dropna()
    
    print(f"Created sample QQQ data:")
    print(f"  Records: {len(data)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data

if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_qqq_data()
    
    # Save to CSV for backup
    sample_data.to_csv('sample_qqq_data.csv')
    print("Sample data saved to 'sample_qqq_data.csv'")
    
    # Show sample
    print("\nSample data:")
    print(sample_data.head())
