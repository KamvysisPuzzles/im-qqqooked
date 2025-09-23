"""
Utility functions for the QQQ Mean Reversion Strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    if returns.std() == 0:
        return 0
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown and its duration"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find the peak before max drawdown
    peak_before_dd = peak.loc[:max_dd_date].idxmax()
    
    return max_dd, peak_before_dd, max_dd_date


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio"""
    if max_drawdown == 0:
        return float('inf')
    return annual_return / abs(max_drawdown)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf')
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns.mean() * np.sqrt(252) / downside_deviation


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)"""
    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR)"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_trade_statistics(trades: List[Dict]) -> Dict:
    """Calculate comprehensive trade statistics"""
    if not trades:
        return {}
    
    pnls = [trade['pnl'] for trade in trades if trade.get('pnl') is not None]
    
    if not pnls:
        return {"total_trades": len(trades)}
    
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]
    
    stats = {
        "total_trades": len(trades),
        "completed_trades": len(pnls),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(pnls) if pnls else 0,
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls),
        "median_pnl": np.median(pnls),
        "std_pnl": np.std(pnls),
        "max_profit": max(pnls),
        "max_loss": min(pnls),
        "avg_win": np.mean(winning_trades) if winning_trades else 0,
        "avg_loss": np.mean(losing_trades) if losing_trades else 0,
        "profit_factor": abs(sum(winning_trades)) / abs(sum(losing_trades)) if losing_trades else float('inf'),
        "expectancy": np.mean(pnls),
    }
    
    return stats


def calculate_rolling_metrics(prices: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate rolling performance metrics"""
    returns = prices.pct_change()
    
    rolling_metrics = pd.DataFrame(index=prices.index)
    rolling_metrics['returns'] = returns
    rolling_metrics['cumulative_return'] = (1 + returns).cumprod() - 1
    rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
        lambda x: calculate_sharpe_ratio(x) if len(x) == window else np.nan
    )
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling max drawdown
    rolling_peak = prices.rolling(window).max()
    rolling_drawdown = (prices - rolling_peak) / rolling_peak
    rolling_metrics['rolling_max_drawdown'] = rolling_drawdown.rolling(window).min()
    
    return rolling_metrics


def format_performance_summary(results: Dict) -> str:
    """Format performance results into a readable summary"""
    
    summary = f"""
PERFORMANCE SUMMARY
{'='*50}
Total Return:        {results.get('total_return', 0):>8.2%}
Annualized Return:   {results.get('annualized_return', 0):>8.2%}
Volatility:          {results.get('volatility', 0):>8.2%}
Sharpe Ratio:        {results.get('sharpe_ratio', 0):>8.2f}
Max Drawdown:        {results.get('max_drawdown', 0):>8.2%}
Final Capital:       ${results.get('final_capital', 0):>8,.2f}

TRADE STATISTICS
{'='*50}
Total Trades:        {results.get('total_trades', 0):>8}
Winning Trades:      {results.get('winning_trades', 0):>8}
Losing Trades:       {results.get('losing_trades', 0):>8}
Win Rate:           {results.get('win_rate', 0):>8.2%}
Average Win:         ${results.get('avg_win', 0):>8.2f}
Average Loss:        ${results.get('avg_loss', 0):>8.2f}
Profit Factor:       {results.get('profit_factor', 0):>8.2f}
Max Profit:         ${results.get('max_profit', 0):>8.2f}
Max Loss:           ${results.get('max_loss', 0):>8.2f}
"""
    
    return summary


def validate_strategy_parameters(config: Dict) -> List[str]:
    """Validate strategy configuration parameters"""
    errors = []
    
    strategy_params = config.get('strategy', {})
    risk_params = config.get('risk', {})
    
    # Validate strategy parameters
    if strategy_params.get('lookback_period', 0) <= 0:
        errors.append("lookback_period must be positive")
    
    if strategy_params.get('threshold_multiplier', 0) <= 0:
        errors.append("threshold_multiplier must be positive")
    
    if strategy_params.get('min_holding_period', 0) < 0:
        errors.append("min_holding_period must be non-negative")
    
    if strategy_params.get('max_holding_period', 0) <= strategy_params.get('min_holding_period', 0):
        errors.append("max_holding_period must be greater than min_holding_period")
    
    # Validate risk parameters
    if not 0 < risk_params.get('stop_loss_pct', 0) < 1:
        errors.append("stop_loss_pct must be between 0 and 1")
    
    if not 0 < risk_params.get('take_profit_pct', 0) < 1:
        errors.append("take_profit_pct must be between 0 and 1")
    
    if not 0 < risk_params.get('position_size', 0) <= 1:
        errors.append("position_size must be between 0 and 1")
    
    return errors


def resample_data(data: pd.DataFrame, target_frequency: str) -> pd.DataFrame:
    """Resample data to target frequency"""
    
    if target_frequency == '1h':
        return data
    elif target_frequency == '1d':
        return data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    elif target_frequency == '4h':
        return data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        raise ValueError(f"Unsupported frequency: {target_frequency}")


def detect_market_regime(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Detect market regime (trending vs mean-reverting)"""
    
    # Calculate trend strength using ADX-like metric
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window).mean()
    
    # Calculate trend direction
    price_change = data['close'].pct_change()
    trend_strength = price_change.rolling(window).std()
    
    # Normalize to 0-1 scale
    regime_score = (trend_strength / atr).rolling(window).mean()
    
    # Classify regimes
    regimes = pd.Series(index=data.index, dtype=str)
    regimes[regime_score > regime_score.quantile(0.7)] = 'trending'
    regimes[regime_score < regime_score.quantile(0.3)] = 'mean_reverting'
    regimes = regimes.fillna('neutral')
    
    return regimes
