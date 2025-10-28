"""
3EMA + MACD-V + Aroon Trend Following Strategy Implementation
"""
import pandas as pd
import numpy as np
import talib
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
    signal_source: Optional[str] = None  # Which indicator triggered the signal


class TrendFollowingStrategy:
    """3EMA + MACD-V + Aroon Trend Following Strategy for TQQQ"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_params = config.get('strategy', {})
        self.risk_params = config.get('risk', {})
        
        # Triple EMA parameters
        self.ema1 = self.strategy_params.get('ema1', 12)
        self.ema2 = self.strategy_params.get('ema2', 89)
        self.ema3 = self.strategy_params.get('ema3', 125)
        
        # MACD-V parameters
        self.macd_fast = self.strategy_params.get('macd_fast', 25)
        self.macd_slow = self.strategy_params.get('macd_slow', 30)
        self.macd_signal = self.strategy_params.get('macd_signal', 85)
        self.volume_threshold = self.strategy_params.get('volume_threshold', 0.0)
        
        # Aroon parameters
        self.aroon_length = self.strategy_params.get('aroon_length', 66)
        
        # Risk parameters
        self.stop_loss_pct = self.risk_params.get('stop_loss_pct', None)  # Disabled
        self.take_profit_pct = self.risk_params.get('take_profit_pct', None)  # Disabled
        self.position_size = self.risk_params.get('position_size', 0.1)
        self.max_positions = self.risk_params.get('max_positions', 1)
        self.max_holding_days = self.risk_params.get('max_holding_days', None)  # Disabled
        
        # State variables
        self.current_position = Position.FLAT
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using 3EMA + MACD-V + Aroon ensemble (OR logic)
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signals added
        """
        df = data.copy()
        
        # Initialize signals
        df['signal_3ema'] = 0
        df['signal_macdv'] = 0
        df['signal_aroon'] = 0
        df['signal'] = 0
        df['signal_source'] = ''
        
        # Calculate indicators if not already present
        self._calculate_indicators(df)
        
        # Generate signals for each indicator
        for i in range(1, len(df)):
            # 3EMA signals
            signal_3ema = self._get_3ema_signal(df, i)
            
            # MACD-V signals
            signal_macdv = self._get_macdv_signal(df, i)
            
            # Aroon signals
            signal_aroon = self._get_aroon_signal(df, i)
            
            # Combine with OR logic - any indicator can trigger
            if signal_3ema != 0:
                df.iloc[i, df.columns.get_loc('signal_3ema')] = signal_3ema
                df.iloc[i, df.columns.get_loc('signal')] = signal_3ema
                df.iloc[i, df.columns.get_loc('signal_source')] = '3EMA'
            elif signal_macdv != 0:
                df.iloc[i, df.columns.get_loc('signal_macdv')] = signal_macdv
                df.iloc[i, df.columns.get_loc('signal')] = signal_macdv
                df.iloc[i, df.columns.get_loc('signal_source')] = 'MACD-V'
            elif signal_aroon != 0:
                df.iloc[i, df.columns.get_loc('signal_aroon')] = signal_aroon
                df.iloc[i, df.columns.get_loc('signal')] = signal_aroon
                df.iloc[i, df.columns.get_loc('signal_source')] = 'Aroon'
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame):
        """Calculate all technical indicators"""
        # EMAs
        df[f'ema{self.ema1}'] = df['close'].ewm(span=self.ema1, adjust=False).mean()
        df[f'ema{self.ema2}'] = df['close'].ewm(span=self.ema2, adjust=False).mean()
        df[f'ema{self.ema3}'] = df['close'].ewm(span=self.ema3, adjust=False).mean()
        
        # MACD
        close_array = df['close'].values
        macd, signal_line, _ = talib.MACD(close_array, 
                                          fastperiod=self.macd_fast,
                                          slowperiod=self.macd_slow,
                                          signalperiod=self.macd_signal)
        df['macd'] = pd.Series(macd, index=df.index)
        df['macd_signal'] = pd.Series(signal_line, index=df.index)
        
        # Volume ratio
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Aroon
        high_array = df['high'].values
        low_array = df['low'].values
        aroon_up, aroon_down = talib.AROON(high_array, low_array, timeperiod=self.aroon_length)
        df['aroon_up'] = pd.Series(aroon_up, index=df.index)
        df['aroon_down'] = pd.Series(aroon_down, index=df.index)
    
    def _get_3ema_signal(self, df: pd.DataFrame, i: int) -> int:
        """Check 3EMA crossover signals"""
        if i == 0:
            return 0
        
        ema1 = df[f'ema{self.ema1}'].iloc[i]
        ema2 = df[f'ema{self.ema2}'].iloc[i]
        ema3 = df[f'ema{self.ema3}'].iloc[i]
        
        ema1_prev = df[f'ema{self.ema1}'].iloc[i-1]
        ema2_prev = df[f'ema{self.ema2}'].iloc[i-1]
        ema3_prev = df[f'ema{self.ema3}'].iloc[i-1]
        
        # Buy signals: any EMA crosses above another
        if (ema1 > ema2 and ema1_prev <= ema2_prev) or \
           (ema1 > ema3 and ema1_prev <= ema3_prev) or \
           (ema2 > ema3 and ema2_prev <= ema3_prev):
            return Signal.BUY.value
        
        # Sell signals: any EMA crosses below another
        elif (ema1 < ema2 and ema1_prev >= ema2_prev) or \
             (ema1 < ema3 and ema1_prev >= ema3_prev) or \
             (ema2 < ema3 and ema2_prev >= ema3_prev):
            return Signal.SELL.value
        
        return 0
    
    def _get_macdv_signal(self, df: pd.DataFrame, i: int) -> int:
        """Check MACD-V signals"""
        if i == 0 or pd.isna(df['macd'].iloc[i]) or pd.isna(df['macd_signal'].iloc[i]):
            return 0
        
        macd = df['macd'].iloc[i]
        macd_signal = df['macd_signal'].iloc[i]
        macd_prev = df['macd'].iloc[i-1]
        signal_prev = df['macd_signal'].iloc[i-1]
        
        volume_ratio = df['volume_ratio'].iloc[i]
        volume_confirm = volume_ratio > self.volume_threshold
        
        # Bullish cross
        if macd > macd_signal and macd_prev <= signal_prev and volume_confirm:
            return Signal.BUY.value
        
        # Bearish cross
        elif macd < macd_signal and macd_prev >= signal_prev and volume_confirm:
            return Signal.SELL.value
        
        return 0
    
    def _get_aroon_signal(self, df: pd.DataFrame, i: int) -> int:
        """Check Aroon crossover signals"""
        if i == 0 or pd.isna(df['aroon_up'].iloc[i]) or pd.isna(df['aroon_down'].iloc[i]):
            return 0
        
        aroon_up = df['aroon_up'].iloc[i]
        aroon_down = df['aroon_down'].iloc[i]
        aroon_up_prev = df['aroon_up'].iloc[i-1]
        aroon_down_prev = df['aroon_down'].iloc[i-1]
        
        # Buy signal: Aroon Up crosses above Aroon Down
        if aroon_up > aroon_down and aroon_up_prev <= aroon_down_prev:
            return Signal.BUY.value
        
        # Sell signal: Aroon Up crosses below Aroon Down
        elif aroon_up < aroon_down and aroon_up_prev >= aroon_down_prev:
            return Signal.SELL.value
        
        return 0
    
    def should_enter_position(self, signal: Signal, current_price: float,
                            timestamp: pd.Timestamp, df: pd.DataFrame) -> bool:
        """Determine if we should enter a position (trend-following)"""
        
        # Don't enter if already in position
        if self.current_position != Position.FLAT:
            return False
        
        # For trend following, we only take LONG positions based on signals
        # Check if we have a valid BUY signal
        if signal == Signal.BUY:
            return True
        
        return False
    
    def should_exit_position(self, current_price: float, entry_price: float,
                           entry_time: pd.Timestamp, current_time: pd.Timestamp,
                           signal: Signal) -> Tuple[bool, str]:
        """Determine if we should exit current position (pure trend-following, exits only on opposite signal)"""
        
        if self.current_position == Position.FLAT:
            return False, ""
        
        # Exit on opposite signal (trend reversal)
        # This is pure trend-following - no stop loss, take profit, or time-based exits
        if self.current_position == Position.LONG and signal == Signal.SELL:
            return True, "opposite_signal"
        
        return False, ""
    
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
                     account_value: float, signal_source: str = None) -> Optional[Trade]:
        """Execute a trade based on signal"""
        
        if signal == Signal.BUY and self.current_position == Position.FLAT:
            # Enter long position (trend-following only goes LONG)
            quantity = self.calculate_position_size(account_value, price, 0.02)  # Use default volatility
            self.current_position = Position.LONG
            self.current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price,
                exit_price=None,
                position_type=Position.LONG,
                quantity=quantity,
                signal_source=signal_source
            )
            logger.info(f"Entered LONG position at {price:.2f}, quantity: {quantity:.2f}, source: {signal_source}")
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
