"""
QQQ Mean Reversion Strategy Implementation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class Position(Enum):
    """Position types"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Trade record"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_type: Position
    quantity: float
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


class MeanReversionStrategy:
    """QQQ Mean Reversion Strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_params = config.get('strategy', {})
        self.risk_params = config.get('risk', {})
        
        # Strategy parameters
        self.lookback_period = self.strategy_params.get('lookback_period', 20)
        self.threshold_multiplier = self.strategy_params.get('threshold_multiplier', 2.0)
        self.min_holding_period = self.strategy_params.get('min_holding_period', 4)
        self.max_holding_period = self.strategy_params.get('max_holding_period', 48)
        
        # Risk parameters
        self.stop_loss_pct = self.risk_params.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.risk_params.get('take_profit_pct', 0.03)
        self.position_size = self.risk_params.get('position_size', 0.1)
        self.max_positions = self.risk_params.get('max_positions', 1)
        
        # State variables
        self.current_position = Position.FLAT
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion logic
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signals added
        """
        df = data.copy()
        
        # Calculate mean reversion signals
        df['signal'] = 0
        df['position'] = Position.FLAT.value
        
        for i in range(self.lookback_period, len(df)):
            current_price = df.iloc[i]['close']
            sma = df.iloc[i]['sma_20']
            volatility = df.iloc[i]['volatility']
            
            # Calculate thresholds
            upper_threshold = sma + (self.threshold_multiplier * volatility)
            lower_threshold = sma - (self.threshold_multiplier * volatility)
            
            # Generate signals
            if current_price < lower_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = Signal.BUY.value
            elif current_price > upper_threshold:
                df.iloc[i, df.columns.get_loc('signal')] = Signal.SELL.value
            else:
                df.iloc[i, df.columns.get_loc('signal')] = Signal.HOLD.value
        
        return df
    
    def should_enter_position(self, signal: Signal, current_price: float, 
                            sma: float, volatility: float) -> bool:
        """Determine if we should enter a position"""
        
        # Don't enter if already in position
        if self.current_position != Position.FLAT:
            return False
        
        # Check signal strength
        if signal == Signal.BUY:
            deviation = (current_price - sma) / sma
            return deviation < -self.threshold_multiplier * (volatility / sma)
        elif signal == Signal.SELL:
            deviation = (current_price - sma) / sma
            return deviation > self.threshold_multiplier * (volatility / sma)
        
        return False
    
    def should_exit_position(self, current_price: float, entry_price: float,
                           entry_time: pd.Timestamp, current_time: pd.Timestamp) -> Tuple[bool, str]:
        """Determine if we should exit current position"""
        
        if self.current_position == Position.FLAT:
            return False, ""
        
        # Calculate holding period in hours
        holding_hours = (current_time - entry_time).total_seconds() / 3600
        
        # Check minimum holding period
        if holding_hours < self.min_holding_period:
            return False, ""
        
        # Check maximum holding period
        if holding_hours >= self.max_holding_period:
            return True, "max_holding_period"
        
        # Calculate P&L
        if self.current_position == Position.LONG:
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"
        
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit"
        
        # Check mean reversion exit
        if self._check_mean_reversion_exit(current_price, entry_price):
            return True, "mean_reversion"
        
        return False, ""
    
    def _check_mean_reversion_exit(self, current_price: float, entry_price: float) -> bool:
        """Check if price has reverted to mean"""
        if self.current_position == Position.LONG:
            return current_price >= entry_price * 1.01  # 1% profit
        else:  # SHORT
            return current_price <= entry_price * 0.99  # 1% profit
    
    def calculate_position_size(self, account_value: float, current_price: float,
                             volatility: float) -> float:
        """Calculate position size based on risk parameters"""
        
        # Kelly criterion inspired position sizing
        risk_amount = account_value * self.position_size
        
        # Adjust for volatility
        volatility_adjustment = min(1.0, 0.02 / volatility)  # Cap at 2% volatility
        
        position_value = risk_amount * volatility_adjustment
        quantity = position_value / current_price
        
        return quantity
    
    def execute_trade(self, signal: Signal, price: float, timestamp: pd.Timestamp,
                     account_value: float, volatility: float) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        if signal == Signal.BUY and self.current_position == Position.FLAT:
            # Enter long position
            quantity = self.calculate_position_size(account_value, price, volatility)
            self.current_position = Position.LONG
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                position_type=Position.LONG,
                quantity=quantity
            )
            logger.info(f"Entered LONG position at {price:.2f}, quantity: {quantity:.2f}")
            return self.current_trade
            
        elif signal == Signal.SELL and self.current_position == Position.FLAT:
            # Enter short position (if enabled)
            quantity = self.calculate_position_size(account_value, price, volatility)
            self.current_position = Position.SHORT
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                position_type=Position.SHORT,
                quantity=quantity
            )
            logger.info(f"Entered SHORT position at {price:.2f}, quantity: {quantity:.2f}")
            return self.current_trade
        
        return None
    
    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str) -> Optional[Trade]:
        """Close current position"""
        
        if self.current_position == Position.FLAT or self.current_trade is None:
            return None
        
        # Calculate P&L
        if self.current_position == Position.LONG:
            pnl = (price - self.current_trade.entry_price) * self.current_trade.quantity
        else:  # SHORT
            pnl = (self.current_trade.entry_price - price) * self.current_trade.quantity
        
        # Update trade record
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = price
        self.current_trade.pnl = pnl
        self.current_trade.exit_reason = reason
        
        logger.info(f"Closed {self.current_position.name} position at {price:.2f}, "
                   f"P&L: {pnl:.2f}, Reason: {reason}")
        
        # Store completed trade
        completed_trade = self.current_trade
        self.trades.append(completed_trade)
        
        # Reset position
        self.current_position = Position.FLAT
        self.current_trade = None
        
        return completed_trade
    
    def get_strategy_summary(self) -> Dict:
        """Get summary statistics of the strategy"""
        if not self.trades:
            return {"total_trades": 0}
        
        completed_trades = [t for t in self.trades if t.pnl is not None]
        
        if not completed_trades:
            return {"total_trades": len(self.trades), "completed_trades": 0}
        
        pnls = [t.pnl for t in completed_trades]
        
        return {
            "total_trades": len(self.trades),
            "completed_trades": len(completed_trades),
            "winning_trades": len([p for p in pnls if p > 0]),
            "losing_trades": len([p for p in pnls if p < 0]),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "max_profit": max(pnls),
            "max_loss": min(pnls),
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
            "profit_factor": abs(sum([p for p in pnls if p > 0])) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
        }
