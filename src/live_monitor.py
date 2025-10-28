"""
Live monitoring module for signal generation (manual execution via Telegram alerts)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import time

from src.data_fetcher import DataFetcher
from src.strategy import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class LiveMonitor:
    """Monitor live signals and send alerts for manual execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_params = config.get('strategy', {})
        self.live_params = config.get('live', {})
        
        # Initialize components
        self.fetcher = DataFetcher(config['data']['symbol'])
        self.strategy = TrendFollowingStrategy(config)
        
        # Telegram settings (if enabled)
        self.telegram_enabled = self.live_params.get('telegram_enabled', False)
        self.telegram_bot_token = self.live_params.get('telegram_bot_token', '')
        self.telegram_chat_id = self.live_params.get('telegram_chat_id', '')
        
        # State tracking
        self.last_signal_time = None
        self.current_position = False
        self.last_check_time = None
        
        logger.info("Live Monitor initialized")
    
    def run(self):
        """Main monitoring loop"""
        check_interval = self.live_params.get('signal_check_interval', 3600)  # Default: 1 hour
        
        logger.info(f"Starting live monitoring for {self.fetcher.symbol}")
        logger.info(f"Checking signals every {check_interval/60:.1f} minutes")
        logger.info("="*70)
        
        while True:
            try:
                self.check_signals()
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def check_signals(self):
        """Check for new trading signals"""
        try:
            # Fetch recent data (need enough for indicators)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=200)  # Get enough history
            
            data = self.fetcher.fetch_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if data.empty:
                logger.warning("No data available")
                return
            
            # Generate signals
            data_with_signals = self.strategy.generate_signals(data)
            
            # Get latest signal
            latest_row = data_with_signals.iloc[-1]
            signal = latest_row['signal']
            signal_source = latest_row.get('signal_source', 'Unknown')
            current_price = latest_row['close']
            
            # Check for new signals
            if signal != 0 and self.last_signal_time != latest_row.name:
                timestamp = latest_row.name
                self.send_signal_alert(signal, current_price, timestamp, signal_source)
                self.last_signal_time = timestamp
                
                logger.info(f"New signal detected: {signal} at ${current_price:.2f} via {signal_source}")
            else:
                logger.debug("No new signals")
            
            self.last_check_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
    
    def send_signal_alert(self, signal: int, price: float, timestamp: pd.Timestamp, source: str):
        """Send signal alert via configured method"""
        
        signal_type = "BUY" if signal == 1 else "SELL"
        message = f"""
{'='*50}
TRADING SIGNAL ALERT
{'='*50}

Symbol: {self.fetcher.symbol}
Signal: {signal_type}
Price: ${price:.2f}
Time: {timestamp}
Source: {source}

Action Required: MANUAL EXECUTION
{'='*50}
"""
        
        # Log to console and file
        logger.info(message)
        
        # Send via Telegram if enabled
        if self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id:
            self.send_telegram_message(message)
        else:
            logger.info("Telegram not configured. Signal logged only.")
    
    def send_telegram_message(self, message: str):
        """Send message via Telegram bot"""
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
            else:
                logger.error(f"Failed to send Telegram message: {response.status_code}")
                
        except ImportError:
            logger.warning("requests library not installed. Install with: pip install requests")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")


def main():
    """Entry point for live monitoring"""
    import yaml
    
    # Load configuration
    with open('config/strategy_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize and run monitor
    monitor = LiveMonitor(config)
    monitor.run()


if __name__ == "__main__":
    main()

