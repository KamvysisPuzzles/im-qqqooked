"""
Backtesting framework for 3EMA + MACD-V + Aroon Trend Following Strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from src.strategy import TrendFollowingStrategy, Trade, Position
from src.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for the trend-following strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backtest_params = config.get('backtest', {})
        self.metrics_params = config.get('metrics', {})
        
        # Backtest parameters
        self.initial_capital = self.backtest_params.get('initial_capital', 100000)
        self.commission = self.backtest_params.get('commission', 0.001)
        self.slippage = self.backtest_params.get('slippage', 0.0005)
        
        # Metrics parameters
        self.benchmark_symbol = self.metrics_params.get('benchmark', 'SPY')
        self.risk_free_rate = self.metrics_params.get('risk_free_rate', 0.02)
        
        # State
        self.current_capital = self.initial_capital
        self.portfolio_value = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, data: pd.DataFrame, strategy: TrendFollowingStrategy) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data with technical indicators
            strategy: Trend following strategy instance
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data)
        
        # Initialize portfolio tracking
        portfolio_values = []
        timestamps = []
        
        for i, (timestamp, row) in enumerate(data_with_signals.iterrows()):
            current_price = row['close']
            signal_value = row['signal']
            volatility = row['volatility']
            
            # Convert signal to Signal enum
            if signal_value == 1:
                signal = strategy.Signal.BUY
            elif signal_value == -1:
                signal = strategy.Signal.SELL
            else:
                signal = strategy.Signal.HOLD
            
            # Check for exit conditions
            if strategy.current_position != Position.FLAT and strategy.current_trade:
                should_exit, exit_reason = strategy.should_exit_position(
                    current_price, 
                    strategy.current_trade.entry_price,
                    strategy.current_trade.entry_time,
                    timestamp,
                    signal
                )
                
                if should_exit:
                    # Apply slippage
                    exit_price = current_price * (1 - self.slippage) if strategy.current_position == Position.LONG else current_price * (1 + self.slippage)
                    
                    completed_trade = strategy.close_position(exit_price, timestamp, exit_reason)
                    if completed_trade:
                        self.trades.append(completed_trade)
                        self._update_capital(completed_trade)
            
            # Check for entry conditions (trend following only goes LONG)
            if strategy.should_enter_position(signal, current_price, timestamp, data_with_signals):
                # Apply slippage for long entry
                entry_price = current_price * (1 + self.slippage)
                
                # Get signal source
                signal_source = row.get('signal_source', 'Unknown')
                
                new_trade = strategy.execute_trade(signal, entry_price, timestamp, self.current_capital, signal_source)
                if new_trade:
                    self.trades.append(new_trade)
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price, strategy)
            portfolio_values.append(portfolio_value)
            timestamps.append(timestamp)
        
        # Store results
        self.equity_curve = pd.DataFrame({
            'timestamp': timestamps,
            'portfolio_value': portfolio_values
        }).set_index('timestamp')
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(data_with_signals)
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        return results
    
    def _update_capital(self, trade: Trade):
        """Update capital after trade completion"""
        if trade.pnl is not None:
            # Apply commission
            commission_cost = abs(trade.quantity * trade.entry_price * self.commission) + \
                            abs(trade.quantity * trade.exit_price * self.commission)
            
            self.current_capital += trade.pnl - commission_cost
    
    def _calculate_portfolio_value(self, current_price: float, strategy: TrendFollowingStrategy) -> float:
        """Calculate current portfolio value"""
        if strategy.current_position == Position.FLAT:
            return self.current_capital
        
        # Calculate unrealized P&L
        if strategy.current_trade:
            if strategy.current_position == Position.LONG:
                unrealized_pnl = (current_price - strategy.current_trade.entry_price) * strategy.current_trade.quantity
            else:  # SHORT
                unrealized_pnl = (strategy.current_trade.entry_price - current_price) * strategy.current_trade.quantity
            
            return self.current_capital + unrealized_pnl
        
        return self.current_capital
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if self.equity_curve.empty:
            return {"error": "No equity curve data available"}
        
        # Basic metrics
        total_return = (self.equity_curve['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Calculate returns
        returns = self.equity_curve['portfolio_value'].pct_change().dropna()
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24)  # Annualized for hourly data
        sharpe_ratio = (returns.mean() * 252 * 24 - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = self.equity_curve['portfolio_value'].expanding().max()
        drawdown = (self.equity_curve['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade analysis
        completed_trades = [t for t in self.trades if t.pnl is not None]
        
        trade_metrics = {}
        if completed_trades:
            pnls = [t.pnl for t in completed_trades]
            trade_metrics = {
                'total_trades': len(completed_trades),
                'winning_trades': len([p for p in pnls if p > 0]),
                'losing_trades': len([p for p in pnls if p < 0]),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
                'avg_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
                'profit_factor': abs(sum([p for p in pnls if p > 0])) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf'),
                'max_profit': max(pnls),
                'max_loss': min(pnls)
            }
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 * 24 / len(self.equity_curve)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.equity_curve['portfolio_value'].iloc[-1],
            **trade_metrics
        }
    
    def plot_results(self, data: pd.DataFrame, save_path: str = None):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 14))
        
        # Plot 1: Price and EMAs
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='TQQQ Price', color='black', linewidth=1.5, alpha=0.8)
        
        # Plot EMAs if they exist
        for ema_col in ['ema12', 'ema89', 'ema125']:
            if ema_col in data.columns:
                ax1.plot(data.index, data[ema_col], label=ema_col.upper(), alpha=0.6, linestyle='--')
        
        # Mark trade entries and exits
        for trade in self.trades:
            if trade.pnl is not None:  # Completed trades
                color = 'green' if trade.pnl > 0 else 'red'
                ax1.scatter(trade.entry_time, trade.entry_price, color=color, s=50, alpha=0.7)
                ax1.scatter(trade.exit_time, trade.exit_price, color=color, s=50, alpha=0.7, marker='x')
        
        ax1.set_title('TQQQ Price with Trading Signals', fontweight='bold')
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value
        ax2 = axes[1]
        ax2.plot(self.equity_curve.index, self.equity_curve['portfolio_value'], label='Portfolio Value', color='darkgreen', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time', fontweight='bold')
        ax2.set_ylabel('Value ($)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        peak = self.equity_curve['portfolio_value'].expanding().max()
        drawdown = (self.equity_curve['portfolio_value'] - peak) / peak
        ax3.fill_between(self.equity_curve.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(self.equity_curve.index, drawdown, color='red', linewidth=1.5, alpha=0.7)
        ax3.set_title('Drawdown Analysis', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trade P&L distribution
        ax4 = axes[3]
        completed_trades = [t for t in self.trades if t.pnl is not None]
        if completed_trades:
            pnls = [t.pnl for t in completed_trades]
            ax4.hist(pnls, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.axvline(0, color='red', linestyle='--', linewidth=2)
            ax4.set_xlabel('P&L ($)', fontweight='bold')
            ax4.set_ylabel('Frequency', fontweight='bold')
            ax4.set_title('Trade P&L Distribution', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        ax3.set_xlabel('Date', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_trade_report(self) -> pd.DataFrame:
        """Generate detailed trade report"""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            if trade.pnl is not None:  # Only completed trades
                trade_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'position_type': trade.position_type.name,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl / (trade.entry_price * trade.quantity),
                    'holding_hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600,
                    'exit_reason': trade.exit_reason
                })
        
        return pd.DataFrame(trade_data)


def run_backtest(config: Dict, start_date: str = None, end_date: str = None) -> Dict:
    """
    Run a complete backtest
    
    Args:
        config: Configuration dictionary
        start_date: Override start date
        end_date: Override end date
        
    Returns:
        Backtest results
    """
    # Load data
    data_config = config['data']
    if start_date:
        data_config['start_date'] = start_date
    if end_date:
        data_config['end_date'] = end_date
    
    fetcher = DataFetcher(data_config['symbol'])
    data = fetcher.fetch_data(
        data_config['start_date'],
        data_config['end_date'],
        data_config['timeframe']
    )
    
    # Initialize strategy and backtest engine
    strategy = TrendFollowingStrategy(config)
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    results = backtest_engine.run_backtest(data, strategy)
    
    # Generate plots
    backtest_engine.plot_results(data, 'results/backtest_results.png')
    
    # Generate trade report
    trade_report = backtest_engine.generate_trade_report()
    if not trade_report.empty:
        trade_report.to_csv('results/trade_report.csv', index=False)
    
    return results
