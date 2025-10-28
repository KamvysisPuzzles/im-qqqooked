"""
Simple test to debug data fetching issues
"""
import yfinance as yf
from datetime import datetime

print("Testing Yahoo Finance data fetching...")

# Test 1: Basic QQQ data
print("\n1. Testing basic QQQ data:")
try:
    ticker = yf.Ticker("QQQ")
    data = ticker.history(period="5d")  # Last 5 days
    print(f"Got {len(data)} records")
    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Specific date range
print("\n2. Testing specific date range:")
try:
    data = ticker.history(start="2023-01-01", end="2023-12-31", interval="1d")
    print(f"Got {len(data)} records for 2023")
    if len(data) > 0:
        print(f"First price: ${data['Close'].iloc[0]:.2f}")
        print(f"Last price: ${data['Close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Hourly data
print("\n3. Testing hourly data:")
try:
    data = ticker.history(start="2024-01-01", end="2024-01-02", interval="1h")
    print(f"Got {len(data)} hourly records")
    if len(data) > 0:
        print(f"Sample prices: {data['Close'].head().tolist()}")
except Exception as e:
    print(f"Error: {e}")

print("\nDone testing!")
