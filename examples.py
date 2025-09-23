"""
Example usage and testing scripts for QQQ Mean Reversion Strategy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yaml

from src.data_fetcher import DataFetcher
from src.strategy import MeanReversionStrategy
from src.backtest import BacktestEngine
from src.utils import format_performance_summary, validate_strategy_parameters


def run_sample_backtest():
    """Run a sample backtest with default parameters"""
    
    print("Running Sample Backtest...")
    print("="*50)
    
    # Load configuration
    with open('config/strategy_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate parameters
    errors = validate_strategy_parameters(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Fetch data
    fetcher = DataFetcher(config['data']['symbol'])
    data = fetcher.fetch_data(
        config['data']['start_date'],
        config['data']['end_date'],
        config['data']['timeframe']
    )
    
    print(f"Fetched {len(data)} data points")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Initialize strategy and backtest
    strategy = MeanReversionStrategy(config)
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    results = backtest_engine.run_backtest(data, strategy)
    
    # Print results
    print(format_performance_summary(results))
    
    # Generate plots
    backtest_engine.plot_results(data, 'results/sample_backtest.png')
    
    return results


def test_parameter_sensitivity():
    """Test sensitivity to different parameters"""
    
    print("Testing Parameter Sensitivity...")
    print("="*50)
    
    # Load base configuration
    with open('config/strategy_config.yaml', 'r') as file:
        base_config = yaml.safe_load(file)
    
    # Test different lookback periods
    lookback_periods = [10, 15, 20, 25, 30]
    results_by_lookback = []
    
    fetcher = DataFetcher(base_config['data']['symbol'])
    data = fetcher.fetch_data(
        base_config['data']['start_date'],
        base_config['data']['end_date'],
        base_config['data']['timeframe']
    )
    
    for lookback in lookback_periods:
        config = base_config.copy()
        config['strategy']['lookback_period'] = lookback
        
        strategy = MeanReversionStrategy(config)
        backtest_engine = BacktestEngine(config)
        
        try:
            results = backtest_engine.run_backtest(data, strategy)
            results_by_lookback.append({
                'lookback_period': lookback,
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0)
            })
            print(f"Lookback {lookback}: Return={results.get('total_return', 0):.2%}, "
                  f"Sharpe={results.get('sharpe_ratio', 0):.2f}")
        except Exception as e:
            print(f"Error with lookback {lookback}: {e}")
    
    # Plot sensitivity results
    if results_by_lookback:
        df_results = pd.DataFrame(results_by_lookback)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(df_results['lookback_period'], df_results['total_return'])
        axes[0, 0].set_title('Total Return vs Lookback Period')
        axes[0, 0].set_xlabel('Lookback Period')
        axes[0, 0].set_ylabel('Total Return')
        
        axes[0, 1].plot(df_results['lookback_period'], df_results['sharpe_ratio'])
        axes[0, 1].set_title('Sharpe Ratio vs Lookback Period')
        axes[0, 1].set_xlabel('Lookback Period')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        
        axes[1, 0].plot(df_results['lookback_period'], df_results['max_drawdown'])
        axes[1, 0].set_title('Max Drawdown vs Lookback Period')
        axes[1, 0].set_xlabel('Lookback Period')
        axes[1, 0].set_ylabel('Max Drawdown')
        
        axes[1, 1].plot(df_results['lookback_period'], df_results['win_rate'])
        axes[1, 1].set_title('Win Rate vs Lookback Period')
        axes[1, 1].set_xlabel('Lookback Period')
        axes[1, 1].set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig('results/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()


def analyze_market_regimes():
    """Analyze strategy performance across different market regimes"""
    
    print("Analyzing Market Regimes...")
    print("="*50)
    
    # Load configuration
    with open('config/strategy_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Fetch data
    fetcher = DataFetcher(config['data']['symbol'])
    data = fetcher.fetch_data(
        config['data']['start_date'],
        config['data']['end_date'],
        config['data']['timeframe']
    )
    
    # Detect market regimes
    from src.utils import detect_market_regime
    regimes = detect_market_regime(data)
    
    # Analyze performance by regime
    strategy = MeanReversionStrategy(config)
    backtest_engine = BacktestEngine(config)
    
    regime_results = {}
    
    for regime in ['trending', 'mean_reverting', 'neutral']:
        regime_mask = regimes == regime
        regime_data = data[regime_mask]
        
        if len(regime_data) > 100:  # Minimum data points
            try:
                results = backtest_engine.run_backtest(regime_data, strategy)
                regime_results[regime] = results
                print(f"{regime.capitalize()} regime: "
                      f"Return={results.get('total_return', 0):.2%}, "
                      f"Sharpe={results.get('sharpe_ratio', 0):.2f}")
            except Exception as e:
                print(f"Error analyzing {regime} regime: {e}")
    
    return regime_results


def compare_with_benchmark():
    """Compare strategy performance with benchmark"""
    
    print("Comparing with Benchmark...")
    print("="*50)
    
    # Load configuration
    with open('config/strategy_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Fetch QQQ data
    qqq_fetcher = DataFetcher('QQQ')
    qqq_data = qqq_fetcher.fetch_data(
        config['data']['start_date'],
        config['data']['end_date'],
        config['data']['timeframe']
    )
    
    # Fetch SPY data for benchmark
    spy_fetcher = DataFetcher('SPY')
    spy_data = spy_fetcher.fetch_data(
        config['data']['start_date'],
        config['data']['end_date'],
        config['data']['timeframe']
    )
    
    # Calculate benchmark returns
    spy_returns = spy_data['close'].pct_change().dropna()
    qqq_returns = qqq_data['close'].pct_change().dropna()
    
    # Run strategy backtest
    strategy = MeanReversionStrategy(config)
    backtest_engine = BacktestEngine(config)
    strategy_results = backtest_engine.run_backtest(qqq_data, strategy)
    
    # Calculate benchmark metrics
    benchmark_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0]) - 1
    benchmark_volatility = spy_returns.std() * np.sqrt(252 * 24)
    benchmark_sharpe = (spy_returns.mean() * 252 * 24) / benchmark_volatility
    
    print("Strategy vs Benchmark:")
    print(f"Strategy Return:     {strategy_results.get('total_return', 0):.2%}")
    print(f"Benchmark Return:    {benchmark_return:.2%}")
    print(f"Strategy Sharpe:     {strategy_results.get('sharpe_ratio', 0):.2f}")
    print(f"Benchmark Sharpe:    {benchmark_sharpe:.2f}")
    print(f"Strategy Max DD:     {strategy_results.get('max_drawdown', 0):.2%}")
    
    # Calculate alpha and beta
    aligned_returns = pd.DataFrame({
        'strategy': qqq_returns,
        'benchmark': spy_returns
    }).dropna()
    
    if len(aligned_returns) > 1:
        beta = aligned_returns['strategy'].cov(aligned_returns['benchmark']) / aligned_returns['benchmark'].var()
        alpha = aligned_returns['strategy'].mean() - beta * aligned_returns['benchmark'].mean()
        
        print(f"Beta:                {beta:.2f}")
        print(f"Alpha:               {alpha:.4f}")


if __name__ == "__main__":
    # Run sample backtest
    run_sample_backtest()
    
    # Test parameter sensitivity
    test_parameter_sensitivity()
    
    # Analyze market regimes
    analyze_market_regimes()
    
    # Compare with benchmark
    compare_with_benchmark()
