"""
Main entry point for QQQ Mean Reversion Strategy
"""
import argparse
import logging
import yaml
from datetime import datetime, timedelta
import os

from src.data_fetcher import DataFetcher, load_config
from src.strategy import MeanReversionStrategy
from src.backtest import BacktestEngine, run_backtest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'results', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_strategy_backtest(config_path: str = 'config/strategy_config.yaml'):
    """Run the strategy backtest"""
    
    logger.info("Starting QQQ Mean Reversion Strategy Backtest")
    
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Run backtest
    try:
        results = run_backtest(config)
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
        print(f"Volatility: {results.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        
        if 'total_trades' in results:
            print(f"\nTrade Statistics:")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Winning Trades: {results['winning_trades']}")
            print(f"Losing Trades: {results['losing_trades']}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Average Win: ${results['avg_win']:.2f}")
            print(f"Average Loss: ${results['avg_loss']:.2f}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Max Profit: ${results['max_profit']:.2f}")
            print(f"Max Loss: ${results['max_loss']:.2f}")
        
        print("="*50)
        
        # Save results
        with open('results/backtest_results.yaml', 'w') as file:
            yaml.dump(results, file, default_flow_style=False)
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise


def run_live_strategy(config_path: str = 'config/strategy_config.yaml'):
    """Run the strategy in live mode (simulation)"""
    
    logger.info("Starting QQQ Mean Reversion Strategy (Live Mode)")
    
    # Load configuration
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Initialize components
    fetcher = DataFetcher(config['data']['symbol'])
    strategy = MeanReversionStrategy(config)
    
    logger.info("Strategy initialized. Monitoring for signals...")
    
    # In a real implementation, this would run continuously
    # For now, we'll just fetch the latest data and check for signals
    try:
        latest_price = fetcher.get_latest_price()
        logger.info(f"Latest QQQ price: ${latest_price:.2f}")
        
        # You would implement continuous monitoring here
        # This is a placeholder for the live trading logic
        
    except Exception as e:
        logger.error(f"Error in live strategy: {e}")


def optimize_parameters(config_path: str = 'config/strategy_config.yaml'):
    """Optimize strategy parameters"""
    
    logger.info("Starting parameter optimization")
    
    # Load base configuration
    try:
        with open(config_path, 'r') as file:
            base_config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Parameter ranges to test
    lookback_periods = [10, 15, 20, 25, 30]
    threshold_multipliers = [1.5, 2.0, 2.5, 3.0]
    
    best_config = None
    best_sharpe = -float('inf')
    results = []
    
    for lookback in lookback_periods:
        for threshold in threshold_multipliers:
            # Create test configuration
            test_config = base_config.copy()
            test_config['strategy']['lookback_period'] = lookback
            test_config['strategy']['threshold_multiplier'] = threshold
            
            try:
                # Run backtest
                result = run_backtest(test_config)
                sharpe = result.get('sharpe_ratio', -float('inf'))
                
                results.append({
                    'lookback_period': lookback,
                    'threshold_multiplier': threshold,
                    'sharpe_ratio': sharpe,
                    'total_return': result.get('total_return', 0),
                    'max_drawdown': result.get('max_drawdown', 0)
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = test_config
                
                logger.info(f"Lookback: {lookback}, Threshold: {threshold}, Sharpe: {sharpe:.3f}")
                
            except Exception as e:
                logger.error(f"Error testing parameters: {e}")
    
    # Save optimization results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/optimization_results.csv', index=False)
    
    if best_config:
        with open('results/best_config.yaml', 'w') as file:
            yaml.dump(best_config, file, default_flow_style=False)
        
        print(f"\nBest configuration found:")
        print(f"Lookback Period: {best_config['strategy']['lookback_period']}")
        print(f"Threshold Multiplier: {best_config['strategy']['threshold_multiplier']}")
        print(f"Sharpe Ratio: {best_sharpe:.3f}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='QQQ Mean Reversion Strategy')
    parser.add_argument('--mode', choices=['backtest', 'live', 'optimize'], 
                       default='backtest', help='Strategy mode')
    parser.add_argument('--config', default='config/strategy_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    if args.mode == 'backtest':
        run_strategy_backtest(args.config)
    elif args.mode == 'live':
        run_live_strategy(args.config)
    elif args.mode == 'optimize':
        optimize_parameters(args.config)


if __name__ == "__main__":
    main()
