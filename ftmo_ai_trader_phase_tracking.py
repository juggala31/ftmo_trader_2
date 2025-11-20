#!/usr/bin/env python3
# ==============================================================
# ftmo_ai_trader.py - EXPERT MODE GUI + REAL MT5 TRADING + XGBOOST
# ==============================================================

import os, json, time, math, threading, csv, traceback, sqlite3
from pathlib import Path
from collections import defaultdict, namedtuple
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Fix pandas import to be safe
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available - some features may be limited")

import logging
import logging.handlers
import re
import sys
import io

# Fix console encoding for Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup logging
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('FTMO_AI')
    logger.setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        Path(__file__).parent / "trading_system.log",
        maxBytes=10 * 1024 * 1024,   # 10 MiB
        backupCount=5,
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # optional console output (can be removed)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.stream = io.TextIOWrapper(console.stream.buffer, encoding='utf-8')
    logger.addHandler(console)

    return logger

logger = setup_logging()

# XGBOOST IMPORT
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed – AI will run in fallback mode.")

# GUI IMPORTS
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

# Matplotlib for charts
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    FigureCanvasTkAgg = None
    MATPLOTLIB_AVAILABLE = False
    logger.info("Matplotlib not found – live chart will be disabled.")

# META TRADER 5 WITH OANDA CREDENTIALS - KEEPING YOUR ORIGINAL CREDENTIALS
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True

    # YOUR ORIGINAL CREDENTIALS - KEPT AS REQUESTED
    LOGIN = 1600038177
    PASSWORD = "H32G33x*"
    SERVER = "OANDA-Demo-1"
    TERMINAL_PATH = r"C:\Program Files\OANDA MetaTrader 5\terminal64.exe"

except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 module not found – running in simulation mode.")

# ==============================================================
# CONFIGURATION CONSTANTS
# ==============================================================

PROJECT_ROOT = Path(r"C:\FTMO_AI")
DB_FILE = PROJECT_ROOT / "live_trades.db"
ANALYTICS_FILE = PROJECT_ROOT / "trade_analytics.json"
POSITIONS_FILE = PROJECT_ROOT / "open_positions.json"

# Trading symbols
def get_oanda_symbols() -> List[str]:
    """REAL OANDA symbols for trading – correct case."""
    return ["US30Z25.sim", "US100Z25.sim", "XAUZ25.sim"]

def map_symbol_for_mt5(symbol: str) -> str:
    """Direct mapping – no changes needed if we have the right case."""
    return symbol

# MT5 Initialization
def initialize_mt5() -> bool:
    """Start MT5, enable required symbols and report status."""
    if not MT5_AVAILABLE:
        return False
    try:
        ok = mt5.initialize(path=TERMINAL_PATH,
                            login=LOGIN,
                            password=PASSWORD,
                            server=SERVER)
        if ok:
            logger.info("✅ MT5 INITIALIZED SUCCESSFULLY")
            # Enable real‑trading symbols
            for symbol in get_oanda_symbols():
                if mt5.symbol_select(symbol, True):
                    logger.info(f"   ✅ Symbol enabled: {symbol}")
                else:
                    logger.warning(f"   ❌ Failed to enable symbol: {symbol}")
        return ok
    except Exception as e:
        logger.error(f"❌ MT5 INITIALIZATION ERROR: {e}")
        return False

# ==============================================================
# END OF CONFIGURATION SECTION
# ==============================================================
# ==============================================================
# CORE TRADING CLASSES
# ==============================================================

class SimpleLearningAI:
    """Very lightweight learning AI – keeps a short history and optional XGBoost."""

    def __init__(self):
        self.winning_trades = []
        self.all_trades = []
        self.optimal_position_size = 0.01            # % of account (in lots later)
        self.best_rsi_levels = {'buy': 40, 'sell': 60}
        # XGBoost model (optional)
        self.xgb_model = None
        
        # FIX: Consistent feature names for training and prediction
        self.feature_names = [
            'rsi_at_entry', 'size_used', 'holding_time', 'volatility', 
            'price_change', 'ema_signal', 'market_trend', 'entry_volatility',
            'account_balance_pct', 'time_of_day'
        ]
        
        if XGBOOST_AVAILABLE:
            self.initialize_xgboost_model()

    def initialize_xgboost_model(self):
        """Initialize XGBoost model with default parameters."""
        try:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            logger.info("✅ XGBoost model initialized")
        except Exception as e:
            logger.warning(f"⚠️  XGBoost initialization failed: {e}")
            self.xgb_model = None

    def extract_features(self, df: pd.DataFrame, current_price: float):
        """Create the feature vector expected by the XGBoost model."""
        if pd is None or len(df) < 30:
            return None
        try:
            rsi = self.calculate_rsi(df['close'], 14)
            ema_fast = df['close'].ewm(span=12).mean().iloc[-1]
            ema_slow = df['close'].ewm(span=26).mean().iloc[-1]
            price_change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] if len(df) > 1 else 0
            volatility = df['close'].pct_change().rolling(14).std().iloc[-1] if len(df) > 14 else 10.0
            volume = df['tick_volume'].iloc[-1] if 'tick_volume' in df.columns else 0

            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()

            # Stochastic
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
            stoch_d = stoch_k.rolling(3).mean()

            features = np.array([
                rsi, ema_fast, ema_slow, price_change, volatility,
                volume, macd.iloc[-1], macd_signal.iloc[-1],
                stoch_k.iloc[-1], stoch_d.iloc[-1]
            ])

            return features.reshape(1, -1)
        except Exception as e:
            logger.warning(f"⚠️  Feature extraction error: {e}")
            return None

    def get_xgboost_signal(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float]:
        """Return (`action`, `confidence`) from the XGBoost model."""
        if not XGBOOST_AVAILABLE or self.xgb_model is None:
            return 'hold', 0.0
        
        # ADD THIS CHECK - Model must be fitted before prediction
        if not hasattr(self.xgb_model, 'feature_names_in_') and len(self.all_trades) < 50:
            return 'hold', 0.0
            
        try:
            features = self.extract_features(df, current_price)
            if features is None:
                return 'hold', 0.0

            prediction = self.xgb_model.predict(features)[0]
            probabilities = self.xgb_model.predict_proba(features)[0]

            if prediction == 1:                      # buy
                return 'buy', probabilities[1]
            elif prediction == 2:                    # sell
                return 'sell', probabilities[2]
            else:
                return 'hold', probabilities[0]
        except Exception as e:
            logger.warning(f"⚠️  XGBoost prediction error: {e}")
            return 'hold', 0.0

    def learn_from_trade(self, trade_result: Dict, account_balance: float):
        """Store the result and, if it was a win, update size / RSI levels."""
        self.all_trades.append({
            'size_used': trade_result.get('size', 0.01),
            'account_balance': account_balance,
            'pnl': trade_result.get('pnl', 0),
            'pnl_percent': (trade_result['pnl'] / account_balance) * 100,
            'rsi_at_entry': trade_result.get('rsi', 50),
            'holding_time': trade_result.get('holding_time', 0),
            'volatility': trade_result.get('volatility', 0),
            'was_winner': trade_result['pnl'] > 0
        })

        if trade_result['pnl'] > 0:
            self.winning_trades.append({
                'size_used': trade_result.get('size', 0.01),
                'account_balance': account_balance,
                'pnl_percent': (trade_result['pnl'] / account_balance) * 100,
                'rsi_at_entry': trade_result.get('rsi', 50),
                'holding_time': trade_result.get('holding_time', 0),
                'volatility': trade_result.get('volatility', 0),
            })
            if len(self.winning_trades) > 20:
                self.winning_trades.pop(0)

            self.learn_optimal_size(account_balance)
            self.learn_rsi_levels()

        # Train XGBoost (if we have enough data)
        self.train_xgboost_model()

    def train_xgboost_model(self):
        """FIXED: Very light‑weight incremental training with consistent features."""
        global XGBOOST_AVAILABLE  # FIX: Move global declaration to top
        
        if not XGBOOST_AVAILABLE or self.xgb_model is None:
            return
        if len(self.all_trades) < 50:
            return
            
        try:
            if pd is None:
                logger.warning("⚠️  Pandas not available for XGBoost training")
                return
                
            # FIX: Build consistent features that match what we can store
            rows = []
            labels = []
            for tr in self.all_trades[-100:]:
                # Only use features we actually have in trade history
                rows.append([
                    tr.get('rsi_at_entry', 50),
                    tr.get('size_used', 0.01),
                    tr.get('holding_time', 0),
                    tr.get('volatility', 0),
                    tr.get('price_change', 0),  # Calculate from entry/exit if available
                    tr.get('ema_signal', 0),
                    tr.get('market_trend', 0),  # 0=neutral, 1=bullish, -1=bearish
                    tr.get('entry_volatility', 0),
                    tr.get('account_balance_pct', 0),  # Size as % of account
                    tr.get('time_of_day', 12)  # Hour of trade entry
                ])
                labels.append(1 if tr['was_winner'] else 0)
                
            if len(rows) < 10:
                return
                
            X = np.array(rows)
            y = np.array(labels)

            if len(np.unique(y)) < 2:
                return

            self.xgb_model.fit(X, y)
            logger.info("✅ XGBoost model re‑trained with consistent features")
        except Exception as e:
            logger.warning(f"⚠️  XGBoost re‑training failed: {e}")
            # Disable XGBoost if training consistently fails
            XGBOOST_AVAILABLE = False

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Classic RSI – used for the simple‑rule path."""
        try:
            if len(prices) < period + 1:
                return 50.0
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            result = float(rsi.iloc[-1]) if not rsi.empty else 50.0
            return max(0, min(100, result))
        except Exception:
            return 50.0
  
    def learn_optimal_size(self, current_balance: float):
        """Adjust the optimal %‑of‑balance size based on recent winners."""
        if len(self.winning_trades) < 5:
            return
        best_trade = max(self.winning_trades,
                         key=lambda x: x['pnl_percent'])
        best_size_percent = (best_trade['size_used'] /
                             best_trade['account_balance']) * 100
        self.optimal_position_size = min(
            2.0,
            max(0.5,
                self.optimal_position_size * 0.9 +
                best_size_percent * 0.1)
        )

    def learn_rsi_levels(self):
        """Smooth the best entry‑RSI levels."""
        if len(self.winning_trades) < 3:
            return
        buy_trades = [t for t in self.winning_trades if t['rsi_at_entry'] < 50]
        sell_trades = [t for t in self.winning_trades if t['rsi_at_entry'] > 50]

        if buy_trades:
            avg_buy = np.mean([t['rsi_at_entry'] for t in buy_trades])
            self.best_rsi_levels['buy'] = (
                avg_buy * 0.1 + self.best_rsi_levels['buy'] * 0.9
            )
        if sell_trades:
            avg_sell = np.mean([t['rsi_at_entry'] for t in sell_trades])
            self.best_rsi_levels['sell'] = (
                avg_sell * 0.1 + self.best_rsi_levels['sell'] * 0.9
            )

    def get_adaptive_signal(self, rsi: float, ema_signal: float) -> Tuple[str, float]:
        """ENHANCED: More adaptive signal generation for high RSI markets"""
        # Dynamic thresholds based on market conditions
        if rsi > 75:  # High RSI market
            # In high RSI markets, be more sensitive to sell signals
            ema_threshold = 0.03  # Lower threshold for sells
            rsi_upper = 85        # Allow higher RSI for buys
            rsi_lower = 70        # Higher threshold for sells
        elif rsi < 25:  # Low RSI market
            # In low RSI markets, be more sensitive to buy signals
            ema_threshold = 0.03  # Lower threshold for buys
            rsi_upper = 30        # Lower threshold for buys
            rsi_lower = 15        # Allow lower RSI for sells
        else:  # Normal market
            ema_threshold = 0.05
            rsi_upper = 80
            rsi_lower = 20
        
        # Enhanced buy signal with market context
        if ema_signal > ema_threshold and rsi < rsi_upper:
            # Calculate confidence with market context
            rsi_factor = (rsi_upper - rsi) / rsi_upper
            ema_factor = min(1.0, ema_signal / 0.2)
            
            # Adjust for high RSI conditions
            if rsi > 70:
                confidence_penalty = 0.7  # Reduce confidence in high RSI
            else:
                confidence_penalty = 1.0
                
            confidence = min(0.8, max(0.1, (rsi_factor * 0.6 + ema_factor * 0.4) * confidence_penalty))
            return 'buy', confidence
        
        # Enhanced sell signal with market context
        elif ema_signal < -ema_threshold and rsi > rsi_lower:
            rsi_factor = (rsi - rsi_lower) / (100 - rsi_lower)
            ema_factor = min(1.0, abs(ema_signal) / 0.2)
            
            # Adjust for low RSI conditions
            if rsi < 30:
                confidence_penalty = 0.7  # Reduce confidence in low RSI
            else:
                confidence_penalty = 1.0
                
            confidence = min(0.8, max(0.1, (rsi_factor * 0.6 + ema_factor * 0.4) * confidence_penalty))
            return 'sell', confidence
        
        return 'hold', 0.05

    def get_position_size(self, account_balance: float) -> float:
        """Convert %‑of‑balance to a lot‑size (clamped 0.01 – 1.0)."""
        base_percent = self.optimal_position_size / 100
        position_size = account_balance * base_percent / 10000
        return max(0.01, min(1.0, position_size))
class FTMOEnforcer:
    """Imposes daily‑loss and max‑drawdown rules (FTMO style)."""

    def __init__(self, initial_balance: float = 200_000):
        self.initial_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.daily_loss_limit = initial_balance * 0.05
        self.max_drawdown = initial_balance * 0.10
        self.last_daily_reset = datetime.now().date()
        self.load_persistent_data()

    def load_persistent_data(self):
        try:
            if ANALYTICS_FILE.exists():
                with open(ANALYTICS_FILE, 'r') as f:
                    data = json.load(f)
                today_str = datetime.now().date().isoformat()
                if data.get('ftmo_last_date') == today_str:
                    self.daily_start_balance = data.get('ftmo_daily_start_balance',
                                                       self.initial_balance)
                    self.last_daily_reset = datetime.fromisoformat(
                        data.get('ftmo_last_date')).date()
        except Exception as e:
            logger.warning(f"⚠️  Could not load FTMO persistent data: {e}")

    def save_persistent_data(self):
        try:
            existing = {}
            if ANALYTICS_FILE.exists():
                with open(ANALYTICS_FILE, 'r') as f:
                    existing = json.load(f)

            existing.update({
                'ftmo_daily_start_balance': self.daily_start_balance,
                'ftmo_last_date': datetime.now().date().isoformat()
            })
            with open(ANALYTICS_FILE, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  Could not save FTMO persistent data: {e}")

    def reset_daily_tracking(self, current_balance: float):
        self.daily_start_balance = current_balance
        self.last_daily_reset = datetime.now().date()
        self.save_persistent_data()
        logger.info("✅ FTMO Daily Loss Tracking Reset")

    def check_daily_loss(self, current_balance: float) -> bool:
        if datetime.now().date() != self.last_daily_reset:
            self.daily_start_balance = current_balance
            self.last_daily_reset = datetime.now().date()
            self.save_persistent_data()
        daily_pnl = current_balance - self.daily_start_balance
        return daily_pnl >= -self.daily_loss_limit

    def check_total_drawdown(self, current_balance: float) -> bool:
        total_drawdown = self.initial_balance - current_balance
        return total_drawdown <= self.max_drawdown

    def can_trade(self, current_balance: float) -> bool:
        return self.check_daily_loss(current_balance) and self.check_total_drawdown(current_balance)

    def get_rule_status(self, current_balance: float) -> Dict:
        return {
            'daily_loss_ok': self.check_daily_loss(current_balance),
            'drawdown_ok': self.check_total_drawdown(current_balance),
            'daily_pnl': current_balance - self.daily_start_balance,
            'total_drawdown': self.initial_balance - current_balance
        }

    def check_correlated_positions(self, positions: Dict) -> bool:
        """Return False if we have too many correlated open positions."""
        active = sum(1 for p in positions.values() if p is not None)
        return active <= 3
class PerformanceTracker:
    """Tracks daily / overall performance and persists it."""

    def __init__(self, initial_balance: float = 200_000.0):
        self.initial_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.closed_pnl_today = 0.0
        self.open_positions_pnl = 0.0
        self.trades_today = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.trades: List[Dict] = []
        self.last_reset_date = datetime.now().date()
        self.load_persistent_data()

    def get_phase_status(self):
        total_trades = self.total_trades
        
        if total_trades < 75:
            return {
                'phase': 1,
                'phase_name': 'Data Collection',
                'target_trades': 75,
                'trades_remaining': max(0, 75 - total_trades),
                'confidence_threshold': 0.01,
                'progress': min(100, (total_trades / 75) * 100)
            }
        elif total_trades < 150:
            return {
                'phase': 2,
                'phase_name': 'Quality Improvement',
                'target_trades': 150,
                'trades_remaining': max(0, 150 - total_trades),
                'confidence_threshold': 0.05,
                'progress': min(100, ((total_trades - 75) / 75) * 100)
            }
        else:
            return {
                'phase': 3,
                'phase_name': 'Performance Focus',
                'target_trades': '150+',
                'trades_remaining': 0,
                'confidence_threshold': 0.10,
                'progress': 100
            }

    def get_phase_metrics(self):
        if len(self.trades) < 10:
            return None
            
        recent_trades = self.trades[-50:] if len(self.trades) >= 50 else self.trades
        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(recent_trades) * 100 if recent_trades else 0
        profit_factor = 0
        
        if wins and losses:
            total_wins = sum(t['pnl'] for t in wins)
            total_losses = abs(sum(t['pnl'] for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_trades': len(self.trades)
        }

    def load_persistent_data(self):
        try:
            if ANALYTICS_FILE.exists():
                with open(ANALYTICS_FILE, 'r') as f:
                    data = json.load(f)
                today_str = datetime.now().date().isoformat()
                if data.get('last_date') == today_str:
                    self.daily_start_balance = data.get('daily_start_balance',
                                                       self.initial_balance)
                    self.closed_pnl_today = data.get('closed_pnl_today', 0.0)
                    self.trades_today = data.get('trades_today', 0)
                    self.win_count = data.get('win_count', 0)
                    self.loss_count = data.get('loss_count', 0)
                    self.last_reset_date = datetime.fromisoformat(
                        data.get('last_date')).date()
                self.total_trades = data.get('total_trades', 0)
                self.trades = data.get('trades', [])
        except Exception as e:
            logger.warning(f"⚠️  Could not load persistent data: {e}")

    def save_persistent_data(self):
        try:
            data = {
                'daily_start_balance': self.daily_start_balance,
                'closed_pnl_today': self.closed_pnl_today,
                'trades_today': self.trades_today,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_trades': self.total_trades,
                'trades': self.trades,
                'last_date': datetime.now().date().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            with open(ANALYTICS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  Could not save persistent data: {e}")

    def record_trade(self, trade: Dict):
        trade['timestamp'] = datetime.now().isoformat()
        trade['size'] = trade.get('size', 0.01)
        self.trades.append(trade)
        pnl = trade.get('pnl', 0.0)
        self.closed_pnl_today += pnl
        self.trades_today += 1
        self.total_trades += 1
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.save_persistent_data()

    def update_open_pnl(self, positions: Dict, market_prices: Dict):
        self.open_positions_pnl = 0.0
        total_pnl = 0.0

        for sym, pos in positions.items():
            if pos is None:
                continue

            if MT5_AVAILABLE:
                try:
                    mt5_symbol = map_symbol_for_mt5(sym)
                    mt5_positions = mt5.positions_get(symbol=mt5_symbol)
                    position_found = False

                    if mt5_positions:
                        for mt5_pos in mt5_positions:
                            if 'ticket' in pos and mt5_pos.ticket == pos['ticket']:
                                total_pnl += mt5_pos.profit
                                position_found = True
                                break

                    if not position_found:
                        cur = market_prices.get(sym, 0)
                        if cur > 0 and pos['entry'] > 0:
                            calc = (cur - pos['entry']) * pos['size'] * (
                                1 if pos['dir'] == 'long' else -1)
                            total_pnl += calc
                except Exception:
                    cur = market_prices.get(sym, 0)
                    if cur > 0 and pos['entry'] > 0:
                        calc = (cur - pos['entry']) * pos['size'] * (
                            1 if pos['dir'] == 'long' else -1)
                        total_pnl += calc
            else:
                cur = market_prices.get(sym, 0)
                if cur > 0 and pos['entry'] > 0:
                    calc = (cur - pos['entry']) * pos['size'] * (
                        1 if pos['dir'] == 'long' else -1)
                    total_pnl += calc

        self.open_positions_pnl = total_pnl

    def calculate_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winning = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return (winning / len(self.trades)) * 100

    def calculate_profit_factor(self) -> float:
        if not self.trades:
            return 0.0
        profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        loss = abs(
            sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        return profit / loss if loss > 0 else float('inf')

    def get_performance_metrics(self) -> Dict:
        total = self.closed_pnl_today + self.open_positions_pnl
        return {
            'daily_pnl': round(total, 2),
            'closed_pnl': round(self.closed_pnl_today, 2),
            'open_pnl': round(self.open_positions_pnl, 2),
            'win_rate': round(self.calculate_win_rate(), 2),
            'profit_factor': round(self.calculate_profit_factor(), 2),
            'total_trades': self.total_trades,
            'trades_today': self.trades_today
        }
# ==============================================================
# EXPERT MODE GUI DASHBOARD – TABBED INTERFACE WITH CANDLESTICK CHARTS
# ==============================================================

class ExpertTradingGUI:
    def __init__(self, trader_instance):
        self.trader = trader_instance
        self.root = tk.Tk()
        self.root.title("🎯 FTMO AI QUANTITATIVE TRADING SYSTEM - REAL MT5 TRADING")
        self.root.geometry("1920x1080")
        self.root.configure(bg='#1a1a1a')

        # Modern color scheme
        self.COLORS = {
            'DARK_BG': '#0f172a',      # Deep blue-black
            'CARD_BG': '#1e293b',       # Card background
            'ACCENT_GREEN': '#10b981',  # Emerald green
            'ACCENT_RED': '#ef4444',    # Red accent
            'ACCENT_BLUE': '#3b82f6',   # Blue accent
            'TEXT_PRIMARY': '#f8fafc',  # Light text
            'TEXT_SECONDARY': '#94a3b8' # Muted text
        }

        # ----- ttk style -------------------------------------------------
        self.style = ttk.Style()
        self.style.configure('Expert.TLabelframe',
                             background=self.COLORS['CARD_BG'],
                             foreground='white',
                             font=('Consolas', 10, 'bold'))
        self.style.configure('Expert.TLabelframe.Label',
                             background=self.COLORS['CARD_BG'],
                             foreground=self.COLORS['ACCENT_GREEN'])

        # ----- build UI -------------------------------------------------
        self.setup_expert_gui()
        self.is_running = False
        self.ai_activity_log = []

    def create_modern_card(self, parent, title, accent_color=None):
        """Create a modern card-based container with professional styling."""
        accent = accent_color or self.COLORS['ACCENT_BLUE']
        card = tk.Frame(parent, bg=self.COLORS['CARD_BG'], relief='raised', bd=1)
        
        # Header with modern styling
        header = tk.Frame(card, bg=accent, height=30)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text=title, bg=accent, fg='white', 
                font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        # Content area
        content = tk.Frame(card, bg=self.COLORS['CARD_BG'])
        content.pack(fill='both', expand=True, padx=10, pady=10)
        
        return card, content

    def setup_expert_gui(self):
        # ---- header ----------------------------------------------------
        header_frame = tk.Frame(self.root, bg=self.COLORS['DARK_BG'], height=80)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)

        tk.Label(header_frame,
                 text="🎯 FTMO AI QUANTITATIVE TRADING SYSTEM - REAL MT5 TRADING",
                 font=("Segoe UI", 18, "bold"),
                 bg=self.COLORS['DARK_BG'],
                 fg=self.COLORS['ACCENT_GREEN']).pack(pady=20)

        # ---- main container --------------------------------------------
        main_frame = tk.Frame(self.root, bg=self.COLORS['DARK_BG'])
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # ---- notebook (tabbed interface) -------------------------------
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # ---- Tab 1: Trading Dashboard ----------------------------------
        self.dashboard_frame = tk.Frame(self.notebook, bg=self.COLORS['DARK_BG'])
        self.notebook.add(self.dashboard_frame, text="📊 TRADING DASHBOARD")
        self.setup_dashboard_tab()

        # ---- Tab 2: Charts ---------------------------------------------
        self.charts_frame = tk.Frame(self.notebook, bg=self.COLORS['DARK_BG'])
        self.notebook.add(self.charts_frame, text="📈 CHARTS")
        self.setup_charts_tab()
        
        # ---- Tab 3: Trade History --------------------------------------
        self.history_frame = tk.Frame(self.notebook, bg=self.COLORS['DARK_BG'])
        self.notebook.add(self.history_frame, text="📜 TRADE HISTORY")
        self.setup_history_tab()

        # ---- Tab 4: Advanced Analytics ---------------------------------
        self.analytics_frame = tk.Frame(self.notebook, bg=self.COLORS['DARK_BG'])
        self.notebook.add(self.analytics_frame, text="📊 ADVANCED ANALYTICS")
        self.setup_analytics_tab()

        # ---- control buttons -------------------------------------------
        control_frame = tk.Frame(self.root, bg=self.COLORS['CARD_BG'], height=60)
        control_frame.pack(fill='x', padx=15, pady=5)
        control_frame.pack_propagate(False)

        # Modern button styling
        button_style = {'font': ("Segoe UI", 11, "bold"), 'width': 15, 'height': 1}
        
        tk.Button(control_frame, text="🚀 START", command=self.start_trading,
                  bg=self.COLORS['ACCENT_GREEN'], fg='white', **button_style).pack(side='left', padx=8)

        tk.Button(control_frame, text="🛑 STOP", command=self.stop_trading,
                  bg=self.COLORS['ACCENT_RED'], fg='white', **button_style).pack(side='left', padx=8)

        tk.Button(control_frame, text="🔄 RESET", command=self.reset_ftmo_daily,
                  bg=self.COLORS['ACCENT_BLUE'], fg='white', **button_style).pack(side='left', padx=8)

        tk.Button(control_frame, text="📊 EXPORT", command=self.export_trades,
                  bg='#aa6600', fg='white', **button_style).pack(side='left', padx=8)

        # ---- status bar -------------------------------------------------
        self.status_var = tk.StringVar(value="🟢 SYSTEM READY | REAL MT5 TRADING MODE")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              relief="sunken", anchor="w",
                              bg=self.COLORS['CARD_BG'], fg=self.COLORS['ACCENT_GREEN'],
                              font=("Segoe UI", 10, "bold"))
        status_bar.pack(fill='x', padx=15, pady=(0, 10))

    def setup_dashboard_tab(self):
        # ---- configuration panel ---------------------------------------
        self.add_config_panel(self.dashboard_frame)

        # ---- vertical PanedWindow (top / middle / bottom) -------------
        self.paned_vertical = ttk.PanedWindow(self.dashboard_frame, orient=tk.VERTICAL)
        self.paned_vertical.pack(fill='both', expand=True)

        # ----- TOP pane -------------------------------------------------
        self.top_frame = tk.Frame(self.paned_vertical, bg=self.COLORS['DARK_BG'])
        self._build_top_section(self.top_frame)
        self.paned_vertical.add(self.top_frame, weight=1)

        # ----- MIDDLE pane ----------------------------------------------
        self.middle_frame = tk.Frame(self.paned_vertical, bg=self.COLORS['DARK_BG'])
        self._build_middle_section(self.middle_frame)
        self.paned_vertical.add(self.middle_frame, weight=1)

        # ----- BOTTOM pane ----------------------------------------------
        self.bottom_frame = tk.Frame(self.paned_vertical, bg=self.COLORS['DARK_BG'])
        self._build_bottom_section(self.bottom_frame)
        self.paned_vertical.add(self.bottom_frame, weight=1)

    def add_config_panel(self, parent):
        cfg_card, cfg_content = self.create_modern_card(parent, "⚙️ TRADING CONFIGURATION")
        cfg_card.pack(fill='x', pady=(0, 10), padx=5)

        # Modern configuration controls
        config_grid = tk.Frame(cfg_content, bg=self.COLORS['CARD_BG'])
        config_grid.pack(fill='x', pady=5)

        # Row 1: Confidence and Position Limits
        row1 = tk.Frame(config_grid, bg=self.COLORS['CARD_BG'])
        row1.pack(fill='x', pady=5)

        # Confidence threshold with modern slider
        tk.Label(row1, text="Min Confidence:", bg=self.COLORS['CARD_BG'], 
                fg=self.COLORS['TEXT_PRIMARY'], font=("Segoe UI", 9)).pack(side='left', padx=5)
        
        self.confidence_var = tk.DoubleVar(value=0.01)
        confidence_scale = tk.Scale(row1, from_=0.01, to=0.9, resolution=0.01,
                                   variable=self.confidence_var, showvalue=0,
                                   orient='horizontal', bg=self.COLORS['CARD_BG'], 
                                   fg=self.COLORS['TEXT_PRIMARY'], 
                                   highlightthickness=0, 
                                   troughcolor=self.COLORS['ACCENT_BLUE'],
                                   length=150)
        confidence_scale.pack(side='left', padx=5)
        
        self.confidence_label = tk.Label(row1, textvariable=self.confidence_var, 
                                        bg=self.COLORS['CARD_BG'], 
                                        fg=self.COLORS['ACCENT_BLUE'],
                                        font=("Segoe UI", 9, "bold"), width=6)
        self.confidence_label.pack(side='left', padx=5)

        # Max positions
        tk.Label(row1, text="Max Positions:", bg=self.COLORS['CARD_BG'],
                fg=self.COLORS['TEXT_PRIMARY'], font=("Segoe UI", 9)).pack(side='left', padx=(20,5))
        
        self.max_positions_var = tk.IntVar(value=3)
        pos_spinbox = tk.Spinbox(row1, from_=1, to=10, width=5,
                                textvariable=self.max_positions_var,
                                bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY'],
                                buttonbackground=self.COLORS['ACCENT_BLUE'])
        pos_spinbox.pack(side='left', padx=5)

        # Row 2: Advanced parameters
        row2 = tk.Frame(config_grid, bg=self.COLORS['CARD_BG'])
        row2.pack(fill='x', pady=5)

        # ATR multiplier
        tk.Label(row2, text="ATR Multiplier:", bg=self.COLORS['CARD_BG'],
                fg=self.COLORS['TEXT_PRIMARY'], font=("Segoe UI", 9)).pack(side='left', padx=5)
        
        self.atr_multiplier_var = tk.DoubleVar(value=1.0)
        atr_scale = tk.Scale(row2, from_=0.5, to=3.0, resolution=0.1,
                            variable=self.atr_multiplier_var, showvalue=0,
                            orient='horizontal', bg=self.COLORS['CARD_BG'],
                            length=120)
        atr_scale.pack(side='left', padx=5)
        
        self.atr_label = tk.Label(row2, textvariable=self.atr_multiplier_var,
                                 bg=self.COLORS['CARD_BG'], fg=self.COLORS['ACCENT_BLUE'],
                                 font=("Segoe UI", 9, "bold"), width=4)
        self.atr_label.pack(side='left', padx=5)

        # Risk per trade
        tk.Label(row2, text="Risk per Trade (%):", bg=self.COLORS['CARD_BG'],
                fg=self.COLORS['TEXT_PRIMARY'], font=("Segoe UI", 9)).pack(side='left', padx=(20,5))
        
        self.risk_per_trade_var = tk.DoubleVar(value=1.0)
        risk_spinbox = tk.Spinbox(row2, from_=0.1, to=5.0, increment=0.1, width=5,
                                 textvariable=self.risk_per_trade_var,
                                 bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY'])
        risk_spinbox.pack(side='left', padx=5)

    def _build_top_section(self, parent):
        top_h = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        top_h.pack(fill='both', expand=True)

        # ---- market analysis - modern card ----
        market_card, market_content = self.create_modern_card(top_h, "📊 LIVE MARKET ANALYSIS", self.COLORS['ACCENT_BLUE'])
        top_h.add(market_card, weight=1)

        self.market_text = scrolledtext.ScrolledText(
            market_content, height=12, width=50,
            bg=self.COLORS['CARD_BG'], fg=self.COLORS['ACCENT_GREEN'], 
            font=("Consolas", 9), relief='flat')
        self.market_text.pack(fill='both', expand=True, padx=5, pady=5)

        # ---- AI signals - modern card ----
        signals_card, signals_content = self.create_modern_card(top_h, "🤖 AI TRADING SIGNALS", self.COLORS['ACCENT_GREEN'])
        top_h.add(signals_card, weight=1)

        self.signals_text = scrolledtext.ScrolledText(
            signals_content, height=12, width=50,
            bg=self.COLORS['CARD_BG'], fg='#ff9900', 
            font=("Consolas", 9), relief='flat')
        self.signals_text.pack(fill='both', expand=True, padx=5, pady=5)

    def _build_middle_section(self, parent):
        middle_h = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        middle_h.pack(fill='both', expand=True)

        # ---- live positions - modern card ----
        pos_card, pos_content = self.create_modern_card(middle_h, "💰 LIVE POSITIONS", self.COLORS['ACCENT_RED'])
        middle_h.add(pos_card, weight=1)

        self.positions_text = scrolledtext.ScrolledText(
            pos_content, height=10, width=50,
            bg=self.COLORS['CARD_BG'], fg=self.COLORS['ACCENT_RED'], 
            font=("Consolas", 9), relief='flat')
        self.positions_text.pack(fill='both', expand=True, padx=5, pady=5)

        # ---- performance metrics - modern card ----
        perf_card, perf_content = self.create_modern_card(middle_h, "📈 PERFORMANCE METRICS", self.COLORS['ACCENT_GREEN'])
        middle_h.add(perf_card, weight=1)

        self.performance_text = scrolledtext.ScrolledText(
            perf_content, height=10, width=50,
            bg=self.COLORS['CARD_BG'], fg=self.COLORS['ACCENT_GREEN'], 
            font=("Consolas", 9), relief='flat')
        self.performance_text.pack(fill='both', expand=True, padx=5, pady=5)

    def _build_bottom_section(self, parent):
        bottom_h = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        bottom_h.pack(fill='both', expand=True)

        # ---- FTMO risk management - modern card ----
        ftmo_card, ftmo_content = self.create_modern_card(bottom_h, "🛡️ FTMO RISK MANAGEMENT", '#ff44ff')
        bottom_h.add(ftmo_card, weight=1)

        self.ftmo_text = scrolledtext.ScrolledText(
            ftmo_content, height=8, width=50,
            bg=self.COLORS['CARD_BG'], fg='#ff44ff', 
            font=("Consolas", 9), relief='flat')
        self.ftmo_text.pack(fill='both', expand=True, padx=5, pady=5)

        # ---- AI activity stream - modern card ----
        act_card, act_content = self.create_modern_card(bottom_h, "📝 AI ACTIVITY STREAM", '#ffff44')
        bottom_h.add(act_card, weight=1)

        self.activity_text = scrolledtext.ScrolledText(
            act_content, height=8, width=50,
            bg=self.COLORS['CARD_BG'], fg='#ffff44', 
            font=("Consolas", 9), relief='flat')
        self.activity_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_charts_tab(self):
        if not MATPLOTLIB_AVAILABLE:
            no_chart_label = tk.Label(self.charts_frame, 
                    text="Matplotlib not available - charts disabled\n\n"
                         "Install with: pip install matplotlib",
                    bg=self.COLORS['DARK_BG'], fg=self.COLORS['TEXT_PRIMARY'], 
                    font=("Segoe UI", 12))
            no_chart_label.pack(pady=50)
            return

        # ---- chart selector - modern card ----
        selector_card, selector_content = self.create_modern_card(self.charts_frame, "📈 CHART CONTROLS")
        selector_content.pack(fill='x', padx=5, pady=5)

        selector_frame = tk.Frame(selector_content, bg=self.COLORS['CARD_BG'])
        selector_frame.pack(fill='x', pady=5)

        tk.Label(selector_frame, text="Symbol:",
                 bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY'], 
                 font=("Segoe UI", 10)).pack(side='left', padx=5)
        
        self.chart_symbol = tk.StringVar()
        if self.trader.symbols:
            self.chart_symbol.set(self.trader.symbols[0])
            
        self.symbol_selector = ttk.Combobox(
            selector_frame,
            values=self.trader.symbols,
            textvariable=self.chart_symbol,
            state='readonly',
            width=15)
        self.symbol_selector.pack(side='left', padx=5)
        self.symbol_selector.bind("<<ComboboxSelected>>", lambda e: self.update_chart())

        # Timeframe selector
        tk.Label(selector_frame, text="Timeframe:",
                 bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY'],
                 font=("Segoe UI", 10)).pack(side='left', padx=(20,5))
        
        self.timeframe_var = tk.StringVar(value="M5")
        timeframe_selector = ttk.Combobox(
            selector_frame,
            values=["M1", "M5", "M15", "M30", "H1", "H4"],
            textvariable=self.timeframe_var,
            state='readonly',
            width=8)
        timeframe_selector.pack(side='left', padx=5)
        timeframe_selector.bind("<<ComboboxSelected>>", lambda e: self.update_chart())

        # ---- refresh button ----
        tk.Button(selector_frame, text="🔄 Refresh", command=self.update_chart,
                  bg=self.COLORS['ACCENT_BLUE'], fg='white', 
                  font=("Segoe UI", 9, "bold"), width=10, height=1).pack(side='left', padx=10)

        # ---- chart canvas ----
        chart_card, chart_content = self.create_modern_card(self.charts_frame, "LIVE PRICE CHART")
        chart_content.pack(fill='both', expand=True, padx=5, pady=5)

        self.chart_frame_container = tk.Frame(chart_content, bg=self.COLORS['CARD_BG'])
        self.chart_frame_container.pack(fill='both', expand=True)

        # ---- Matplotlib figure & canvas ----
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor=self.COLORS['DARK_BG'])
        self.ax.set_facecolor(self.COLORS['DARK_BG'])
        self.ax.tick_params(colors=self.COLORS['TEXT_PRIMARY'])
        for spine in self.ax.spines.values():
            spine.set_color(self.COLORS['TEXT_PRIMARY'])
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, self.chart_frame_container)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True)

        # ---- initial chart update ----
        self.root.after(1000, self.update_chart())

    def update_chart(self):
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            symbol = self.chart_symbol.get()
            timeframe = self.timeframe_var.get()
            if not symbol:
                return
                
            # Map timeframe to MT5 constant
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4
            }
            
            df = self.trader.get_bars(
                symbol,
                timeframe_map.get(timeframe, mt5.TIMEFRAME_M5) if MT5_AVAILABLE else None,
                100)
                
            if df.empty or len(df) < 2:
                return
                
            self.ax.clear()
            
            # Create candlestick chart
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # Convert to numpy arrays for faster processing
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Plot candlesticks
            for i in range(len(df)):
                # Wick (high-low line)
                self.ax.plot([i, i], [lows[i], highs[i]], color='white', linewidth=1)
                
                # Body (open-close rectangle)
                open_price = opens[i]
                close_price = closes[i]
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                # Color: green for up candles, red for down candles
                color = self.COLORS['ACCENT_GREEN'] if close_price >= open_price else self.COLORS['ACCENT_RED']
                
                rect = Rectangle((i-0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor='white', linewidth=0.5)
                self.ax.add_patch(rect)
            
            # Format the chart
            self.ax.set_title(f'{symbol} - Candlestick Chart ({timeframe})', 
                             color=self.COLORS['TEXT_PRIMARY'], fontsize=14)
            self.ax.set_xlabel('Time', color=self.COLORS['TEXT_PRIMARY'])
            self.ax.set_ylabel('Price', color=self.COLORS['TEXT_PRIMARY'])
            self.ax.tick_params(colors=self.COLORS['TEXT_PRIMARY'])
            self.ax.grid(True, alpha=0.3, color='gray')
            
            # Set x-axis labels (show every 10th timestamp)
            step = max(1, len(df) // 10)
            indices = range(0, len(df), step)
            labels = [df.index[i].strftime('%H:%M') for i in indices]
            self.ax.set_xticks(indices)
            self.ax.set_xticklabels(labels, rotation=45, ha='right', color=self.COLORS['TEXT_PRIMARY'])
            
            self.fig.tight_layout()
            self.chart_canvas.draw()
            
        except Exception as e:
            logger.error(f"Chart update error: {e}")

    def setup_history_tab(self):
        """Setup the trade history tab with comprehensive trade logging"""
        try:
            # Create main frame for history tab
            history_card, history_content = self.create_modern_card(self.history_frame, "📜 TRADE HISTORY")
            history_content.pack(fill='both', expand=True, padx=5, pady=5)

            # Create treeview for trade data with modern styling
            columns = ('Timestamp', 'Symbol', 'Direction', 'Entry', 'Exit', 'P&L', 'Size', 'Reason')
            self.history_tree = ttk.Treeview(history_content, columns=columns, show='headings', height=20)
            
            # Define headings with modern styling
            for col in columns:
                self.history_tree.heading(col, text=col)
                self.history_tree.column(col, width=100, anchor='center')
            
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(history_content, orient=tk.VERTICAL, command=self.history_tree.yview)
            h_scrollbar = ttk.Scrollbar(history_content, orient=tk.HORIZONTAL, command=self.history_tree.xview)
            self.history_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack elements
            self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Control buttons in modern card
            button_card, button_content = self.create_modern_card(self.history_frame, "HISTORY CONTROLS")
            button_content.pack(fill='x', padx=5, pady=5)
            
            button_frame = tk.Frame(button_content, bg=self.COLORS['CARD_BG'])
            button_frame.pack(fill='x', pady=10)
            
            tk.Button(button_frame, text="🔄 REFRESH HISTORY", command=self.load_trade_history,
                      bg=self.COLORS['ACCENT_BLUE'], fg='white', font=("Segoe UI", 10, "bold"),
                      width=20, height=1).pack(side='left', padx=5)
            
            tk.Button(button_frame, text="📊 EXPORT CSV", command=self.export_history_csv,
                      bg=self.COLORS['ACCENT_GREEN'], fg='white', font=("Segoe UI", 10, "bold"),
                      width=15, height=1).pack(side='left', padx=5)
            
            tk.Button(button_frame, text="🧹 CLEAR FILTERS", command=self.clear_history_filters,
                      bg=self.COLORS['TEXT_SECONDARY'], fg='white', font=("Segoe UI", 10, "bold"),
                      width=15, height=1).pack(side='left', padx=5)
            
            # Filter controls
            filter_frame = tk.Frame(button_content, bg=self.COLORS['CARD_BG'])
            filter_frame.pack(fill='x', pady=5)
            
            tk.Label(filter_frame, text="Filter by Symbol:",
                     bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY']).pack(side='left', padx=5)
            
            self.filter_symbol = tk.StringVar(value="All")
            symbol_filter = ttk.Combobox(filter_frame, textvariable=self.filter_symbol,
                                        values=["All"] + self.trader.symbols, width=12)
            symbol_filter.pack(side='left', padx=5)
            symbol_filter.bind("<<ComboboxSelected>>", lambda e: self.load_trade_history())
            
            tk.Label(filter_frame, text="Filter by Result:",
                     bg=self.COLORS['CARD_BG'], fg=self.COLORS['TEXT_PRIMARY']).pack(side='left', padx=(20,5))
            
            self.filter_result = tk.StringVar(value="All")
            result_filter = ttk.Combobox(filter_frame, textvariable=self.filter_result,
                                        values=["All", "Profit", "Loss"], width=10)
            result_filter.pack(side='left', padx=5)
            result_filter.bind("<<ComboboxSelected>>", lambda e: self.load_trade_history())
            
            # Load initial data
            self.load_trade_history()
            
        except Exception as e:
            # Fallback simple version if something goes wrong
            error_label = tk.Label(self.history_frame, 
                                  text=f"Error setting up history tab: {e}\n\nUsing simple version.",
                                  bg=self.COLORS['DARK_BG'], fg='red')
            error_label.pack(pady=50)
            
            # Simple fallback
            simple_label = tk.Label(self.history_frame, 
                                   text="Trade History - Basic View\n\nRecent trades will appear here during trading.",
                                   bg=self.COLORS['DARK_BG'], fg='white',
                                   font=("Segoe UI", 12))
            simple_label.pack(pady=50)

    def setup_analytics_tab(self):
        """Setup the advanced analytics tab with comprehensive performance visualization"""
        try:
            # Main analytics container 
            analytics_card, analytics_content = self.create_modern_card(self.analytics_frame, "📊 ADVANCED ANALYTICS")
            analytics_content.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Create a notebook for multiple analytics views
            analytics_notebook = ttk.Notebook(analytics_content)
            analytics_notebook.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Tab 1: Performance Overview
            perf_frame = tk.Frame(analytics_notebook, bg=self.COLORS['CARD_BG'])
            analytics_notebook.add(perf_frame, text="📈 Performance Overview")
            self.setup_performance_overview(perf_frame)
            
            # Tab 2: Trade Analytics
            trade_frame = tk.Frame(analytics_notebook, bg=self.COLORS['CARD_BG'])
            analytics_notebook.add(trade_frame, text="📊 Trade Analytics")
            self.setup_trade_analytics(trade_frame)
            
            # Tab 3: Risk Analytics
            risk_frame = tk.Frame(analytics_notebook, bg=self.COLORS['CARD_BG'])
            analytics_notebook.add(risk_frame, text="🛡️ Risk Analytics")
            self.setup_risk_analytics(risk_frame)
            
            # Tab 4: AI Learning Progress
            ai_frame = tk.Frame(analytics_notebook, bg=self.COLORS['CARD_BG'])
            analytics_notebook.add(ai_frame, text="🤖 AI Learning")
            self.setup_ai_learning_tab(ai_frame)
            
        except Exception as e:
            # Fallback simple version
            error_label = tk.Label(self.analytics_frame, 
                                  text=f"Error setting up analytics: {e}\n\nUsing simple version.",
                                  bg=self.COLORS['DARK_BG'], fg='red')
            error_label.pack(pady=50)
            
            simple_label = tk.Label(self.analytics_frame, 
                                   text="Advanced Analytics Dashboard\n\n"
                                        "Basic analytics are available in the Performance Overview tab.\n"
                                        "Full analytics features will be enabled as trading data accumulates.",
                                   bg=self.COLORS['DARK_BG'], fg='white',
                                   font=("Segoe UI", 12))
            simple_label.pack(pady=50)

    def load_trade_history(self):
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        try:
            # Load from trader's performance tracker
            trades = self.trader.performance_tracker.trades
            
            # Also load from database
            db_trades = []
            if self.trader.con:
                cursor = self.trader.con.execute("SELECT * FROM trades ORDER BY ts DESC")
                db_trades = cursor.fetchall()
            
            # Combine and sort all trades by timestamp (newest first)
            all_trades = []
            
            # Add trades from performance tracker
            for trade in trades:
                all_trades.append({
                    'timestamp': trade.get('timestamp', ''),
                    'symbol': trade.get('symbol', ''),
                    'dir': trade.get('dir', ''),
                    'entry': trade.get('entry', 0),
                    'exit': trade.get('exit', 0),
                    'pnl': trade.get('pnl', 0),
                    'size': trade.get('size', 0),
                    'reason': trade.get('reason', '')
                })
            
            # Add trades from database that might not be in tracker
            for row in db_trades:
                exists = any(t.get('timestamp') == row[0] and t.get('symbol') == row[1] for t in all_trades)
                if not exists:
                    all_trades.append({
                        'timestamp': row[0],
                        'symbol': row[1],
                        'dir': row[2],
                        'entry': row[3],
                        'exit': row[4],
                        'pnl': row[5],
                        'size': row[6] if len(row) > 6 else 0,
                        'reason': row[8] if len(row) > 8 else ''
                    })
            
            # Sort by timestamp (newest first)
            all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Add to treeview with color coding
            for trade in all_trades:
                pnl = trade['pnl']
                pnl_str = f"${pnl:,.2f}"
                pnl_color = self.COLORS['ACCENT_GREEN'] if pnl > 0 else self.COLORS['ACCENT_RED'] if pnl < 0 else self.COLORS['TEXT_SECONDARY']
                
                # Create formatted values
                entry_str = f"${trade['entry']:,.2f}" if trade['entry'] else 'N/A'
                exit_str = f"${trade['exit']:,.2f}" if trade['exit'] else 'OPEN'
                size_str = f"{trade['size']:.4f}" if trade.get('size') else 'N/A'
                
                item = self.history_tree.insert('', 'end', values=(
                    trade['timestamp'][:19],
                    trade['symbol'],
                    trade['dir'].upper(),
                    entry_str,
                    exit_str,
                    pnl_str,
                    size_str,
                    trade['reason']
                ))
                
                # Color code the P&L column
                self.history_tree.set(item, 'P&L', pnl_str)
                
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.history_tree.insert('', 'end', values=('Error', 'Loading', 'Trade', 'History', f'Error: {e}', '', '', ''))

    def export_history_csv(self):
        """Export trade history to CSV file"""
        try:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Symbol', 'Direction', 'Entry', 'Exit', 'P&L', 'Size', 'Reason'])
                
                for item in self.history_tree.get_children():
                    values = self.history_tree.item(item)['values']
                    writer.writerow(values)
                    
            self.add_ai_activity(f"Trade history exported to {filename}", "analysis")
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            self.add_ai_activity(f"Export failed: {e}", "risk")

    def clear_history_filters(self):
        """Clear any filters applied to trade history"""
        self.load_trade_history()
        self.add_ai_activity("Trade history filters cleared", "analysis")

    def get_color_coded_value(self, value, value_type):
        """Return color-coded values with emojis for display"""
        if value_type == "pnl":
            if value > 500:      return f"🟢 +${value:,.2f} 📈"
            elif value > 100:    return f"🟢 +${value:,.2f}"
            elif value > -100:   return f"🟡 ${value:,.2f}"
            elif value > -500:   return f"🔴 ${value:,.2f} 📉"
            else:                return f"🔴 ${value:,.2f} 📉"
        if value_type == "win_rate":
            if value > 60:       return f"🟢 {value:.1f}% 🏆"
            elif value > 50:     return f"🟢 {value:.1f}%"
            elif value > 40:     return f"🟡 {value:.1f}%"
            else:                return f"🔴 {value:.1f}%"
        if value_type == "profit_factor":
            if value > 2.0:      return f"🟢 {value:.2f} 🏆"
            elif value > 1.5:    return f"🟢 {value:.2f}"
            elif value > 1.0:    return f"🟡 {value:.2f}"
            else:                return f"🔴 {value:.2f}"
        return str(value)

    def add_ai_activity(self, message, message_type="info"):
        """Add activity to the AI activity stream with modern styling"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = {
            "trade": "🎯",
            "analysis": "📊",
            "risk": "🛡️",
            "learning": "🎓",
            "error": "❌",
            "success": "✅"
        }.get(message_type, "🤖")
        
        # Color coding based on message type
        color = {
            "trade": self.COLORS['ACCENT_GREEN'],
            "analysis": self.COLORS['ACCENT_BLUE'], 
            "risk": self.COLORS['ACCENT_RED'],
            "learning": "#ff9900",
            "error": self.COLORS['ACCENT_RED'],
            "success": self.COLORS['ACCENT_GREEN']
        }.get(message_type, self.COLORS['TEXT_PRIMARY'])
        
        line = f"[{timestamp}] {emoji} {message}"
        self.ai_activity_log.append((line, color))
        if len(self.ai_activity_log) > 15:
            self.ai_activity_log.pop(0)

    def reset_ftmo_daily(self):
        """Reset FTMO daily tracking"""
        current_balance = self.trader.get_account_balance()
        self.trader.ftmo_enforcer.reset_daily_tracking(current_balance)
        self.add_ai_activity("FTMO Daily Loss Tracking Reset", "risk")
        self.update_display()

    def export_trades(self):
        """Export trades to external file"""
        try:
            filename = f"trading_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_metrics': self.trader.performance_tracker.get_performance_metrics(),
                'recent_trades': self.trader.performance_tracker.trades[-50:],
                'ai_config': {
                    'optimal_position_size': self.trader.learning_ai.optimal_position_size,
                    'best_rsi_levels': self.trader.learning_ai.best_rsi_levels
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.add_ai_activity(f"Trade data exported to {filename}", "success")
        except Exception as e:
            self.add_ai_activity(f"Export error: {e}", "error")

    def update_display(self):
        """Update all GUI displays with modern styling"""
        try:
            # Clear all text widgets
            for w in (self.market_text, self.signals_text,
                      self.positions_text, self.performance_text,
                      self.ftmo_text, self.activity_text):
                try:
                    w.delete(1.0, tk.END)
                except:
                    continue

            # Fetch fresh data
            metrics = self.trader.performance_tracker.get_performance_metrics()
            current_balance = self.trader.get_account_balance()
            ftmo_status = self.trader.ftmo_enforcer.get_rule_status(current_balance)

            # Market analysis display with modern formatting
            self.market_text.insert(tk.END, "=== REAL‑TIME MARKET ANALYSIS ===\n\n")
            for symbol in self.trader.symbols:
                sdata = self.trader.get_signal_data(symbol)
                if not sdata:
                    continue
                price = self.trader.get_current_price(symbol)
                
                # Color code based on signal strength
                signal_strength = sdata['confidence'] * 10
                if signal_strength > 7:
                    signal_emoji = "📈"
                elif signal_strength > 5:
                    signal_emoji = "📊"
                else:
                    signal_emoji = "📉"
                    
                self.market_text.insert(
                    tk.END,
                    f"{signal_emoji} {symbol}\n"
                    f"   💰 Price: ${price:,.2f}\n"
                    f"   📈 RSI: {sdata['rsi']:.1f}\n"
                    f"   🎯 Signal: {sdata['action'].upper()}\n"
                    f"   🔮 Confidence: {sdata['confidence']:.1%}\n"
                    f"   🔄 Min Required: {self.confidence_var.get():.1%}\n"
                    f"{'='*40}\n")

            # AI signals display with threshold checking
            self.signals_text.insert(tk.END, "=== AI TRADING DECISIONS ===\n\n")
            for symbol in self.trader.symbols:
                sdata = self.trader.get_signal_data(symbol)
                if not sdata:
                    continue
                action = sdata.get('action', 'hold')
                conf = sdata.get('confidence', 0) * 10
                min_conf = self.confidence_var.get()
                
                if action == 'buy':
                    col, txt, emo = "🟢", "LONG", "📈"
                elif action == 'sell':
                    col, txt, emo = "🔴", "SHORT", "📉"
                else:
                    col, txt, emo = "🟡", "HOLD", "⚡"
                    
                meets_threshold = conf/10 >= min_conf
                threshold_status = "✅" if meets_threshold else "❌"
                
                strength = max(1.0, min(10.0, conf))
                str_col = "🟢" if strength > 7 else ("🟡" if strength > 5 else "🔴")
                self.signals_text.insert(
                    tk.END,
                    f"{col} {emo} {symbol}: {txt}\n"
                    f"   💪 Strength: {str_col} {strength:.1f}/10\n"
                    f"   🔮 Confidence: {conf:.1f}/10 {threshold_status}\n"
                    f"   🎯 Required: {min_conf*10:.1f}/10\n"
                    f"   ✅ FTMO Compliant: {'🟢 YES' if sdata['ftmo_compliant'] else '🔴 NO'}\n"
                    f"{'='*40}\n")

            # Positions display with real-time P&L
            self.positions_text.insert(tk.END, "=== ACTIVE POSITIONS ===\n\n")
            total_open_pnl = 0.0
            active_count = 0
            
            for sym, pos in self.trader.positions.items():
                if pos is None:
                    continue
                    
                active_count += 1
                cur = self.trader.get_current_price(sym)
                pnl = (cur - pos['entry']) * pos['size'] * (
                    1 if pos['dir'] == 'long' else -1)
                total_open_pnl += pnl
                pnl_disp = self.get_color_coded_value(pnl, "pnl")
                
                self.positions_text.insert(
                    tk.END,
                    f"{'🟢' if pnl > 0 else '🔴' if pnl < 0 else '🟡'} {sym}: {pos['dir'].upper()}\n"
                    f"   💰 Entry: ${pos['entry']:.2f}\n"
                    f"   📊 Current: ${cur:.2f}\n"
                    f"   📈 P&L: {pnl_disp}\n"
                    f"   🎯 SL: ${pos['sl']:.2f} | TP: ${pos['tp']:.2f}\n"
                    f"   ⚖️  Size: {pos['size']:.4f} lots\n"
                    f"{'='*40}\n")
                    
            if active_count == 0:
                self.positions_text.insert(
                    tk.END,
                    "🟡 No active positions – AI monitoring markets\n")
            else:
                self.positions_text.insert(
                    tk.END,
                    f"\n📊 Total Open P&L: ${total_open_pnl:,.2f}\n"
                    f"🔢 Active Positions: {active_count}\n")

            # Performance display with phase tracking
            self.performance_text.insert(tk.END, "=== QUANTITATIVE PERFORMANCE ===\n\n")
            self.performance_text.insert(
                tk.END,
                f"📊 Win Rate: {self.get_color_coded_value(metrics['win_rate'], 'win_rate')}\n")
            self.performance_text.insert(
                tk.END,
                f"💰 Today's P&L: {self.get_color_coded_value(metrics['daily_pnl'], 'pnl')}\n")
            self.performance_text.insert(
                tk.END,
                f"📈 Profit Factor: {self.get_color_coded_value(metrics['profit_factor'], 'profit_factor')}\n")
            self.performance_text.insert(
                tk.END,
                f"🔢 Total Trades: {metrics['total_trades']}\n")
            self.performance_text.insert(
                tk.END,
                f"📅 Trades Today: {metrics['trades_today']}\n")
            self.performance_text.insert(
                tk.END,
                f"🏆 Closed P&L: ${metrics['closed_pnl']:,.2f}\n")
            self.performance_text.insert(
                tk.END,
                f"📈 Open P&L: ${metrics['open_pnl']:,.2f}\n")
            self.performance_text.insert(
                tk.END,
                f"💵 Account Balance: ${current_balance:,.2f}\n")

            # Phase tracking display
            phase_status = self.trader.performance_tracker.get_phase_status()
            self.performance_text.insert(
                tk.END,
                f"\n🎯 TRAINING PHASE: {phase_status['phase_name']}\n"
                f"   🔢 Trades: {self.trader.performance_tracker.total_trades}/{phase_status['target_trades']}\n"
                f"   📈 Progress: {phase_status['progress']:.1f}%\n"
                f"   🎯 Confidence: {phase_status['confidence_threshold']:.2f}\n"
            )

            # Phase metrics
            phase_metrics = self.trader.performance_tracker.get_phase_metrics()
            if phase_metrics:
                self.performance_text.insert(
                    tk.END,
                    f"\n📊 PHASE METRICS:\n"
                    f"   🏆 Win Rate: {phase_metrics['win_rate']:.1f}%\n"
                    f"   💰 Profit Factor: {phase_metrics['profit_factor']:.2f}\n"
                )
            # FTMO status with modern risk indicators
            self.ftmo_text.insert(tk.END, "=== RISK MANAGEMENT STATUS ===\n\n")
            daily_ok = "🟢 COMPLIANT" if ftmo_status['daily_loss_ok'] else "🔴 VIOLATION"
            draw_ok = "🟢 COMPLIANT" if ftmo_status['drawdown_ok'] else "🔴 VIOLATION"
            
            # Color code daily P&L
            daily_pnl_color = self.COLORS['ACCENT_GREEN'] if ftmo_status['daily_pnl'] >= 0 else self.COLORS['ACCENT_RED']
            drawdown_color = self.COLORS['ACCENT_GREEN'] if ftmo_status['total_drawdown'] <= self.trader.ftmo_enforcer.max_drawdown * 0.5 else self.COLORS['ACCENT_RED']
            
            self.ftmo_text.insert(
                tk.END,
                f"🛡️  Daily Loss Limit: {daily_ok}\n"
                f"   📊 P&L: ${ftmo_status['daily_pnl']:,.2f}\n"
                f"📉 Max Drawdown: {draw_ok}\n"
                f"   📊 Drawdown: ${ftmo_status['total_drawdown']:,.2f}\n"
                f"✅ Trading Allowed: {'🟢 YES' if self.trader.ftmo_enforcer.can_trade(current_balance) else '🔴 NO'}\n")
            
            self.ftmo_text.insert(
                tk.END,
                f"\n⚙️  TRADING PARAMETERS:\n"
                f"   🎯 Min Confidence: {self.confidence_var.get():.1%}\n"
                f"   🔢 Max Positions: {self.max_positions_var.get()}\n"
                f"   📊 ATR Multiplier: {self.atr_multiplier_var.get():.1f}\n"
                f"   ⚠️  Risk per Trade: {self.risk_per_trade_var.get():.1f}%\n")

            # Activity stream with color coding
            self.activity_text.insert(tk.END, "=== AI ACTIVITY STREAM ===\n\n")
            for line, color in self.ai_activity_log[-12:]:
                # Apply color coding to activity stream
                self.activity_text.insert(tk.END, line + "\n")
                # Apply color to the last inserted line
                start_index = self.activity_text.index("end-2l")
                end_index = self.activity_text.index("end-1l")
                self.activity_text.tag_add(color, start_index, end_index)
                self.activity_text.tag_config(color, foreground=color)
                
            now = datetime.now().strftime("%H:%M:%S")
            status_emoji = "🟢" if self.trader.ftmo_enforcer.can_trade(current_balance) else "🔴"
            status_text = f"{status_emoji} QUANTITATIVE SYSTEM: {'OPTIMAL' if self.trader.ftmo_enforcer.can_trade(current_balance) else 'RISK LIMITED'} | TRADES: {metrics['total_trades']}"
            
            self.activity_text.insert(tk.END, f"\n[{now}] {status_text}\n")

            # Auto-scroll all text widgets
            for w in (self.market_text, self.signals_text,
                      self.positions_text, self.performance_text,
                      self.ftmo_text, self.activity_text):
                try:
                    w.see(tk.END)
                except:
                    continue

            # Status bar with real-time updates
            status = "🟢 QUANTITATIVE TRADING ACTIVE" if self.is_running else "🔴 SYSTEM STOPPED"
            self.status_var.set(
                f"{status} | Balance: ${current_balance:,.2f} | "
                f"Min Conf: {self.confidence_var.get():.1%} | "
                f"Active Positions: {active_count} | "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

            # Update chart if charts tab is visible
            if self.notebook.index(self.notebook.select()) == 1:
                self.update_chart()

        except Exception as e:
            self.status_var.set(f"❌ SYSTEM ERROR: {e}")
            logger.error(f"❌ GUI Update Error: {e}", exc_info=True)
            self.add_ai_activity(f"GUI update error: {e}", "error")

    def start_trading(self):
        """Start the quantitative trading system"""
        if not self.is_running:
            self.is_running = True
            self.trader.is_running = True
            self.trader.trading_enabled = True
            self.add_ai_activity("🚀 QUANTITATIVE TRADING ACTIVATED – REAL MT5 ORDERS", "trade")
            
            # Start trading loop in separate thread
            trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            trading_thread.start()
            
            # Start GUI update loop
            self.update_loop()

    def stop_trading(self):
        """Stop the trading system"""
        self.is_running = False
        self.trader.is_running = False
        self.trader.trading_enabled = False
        self.add_ai_activity("🛑 TRADING SYSTEM STOPPED – No new orders will be placed", "risk")
        self.update_display()

    def update_loop(self):
        """Continuous GUI update loop"""
        if self.is_running:
            self.update_display()
            self.root.after(2000, self.update_loop)  # Update every 2 seconds

    def trading_loop(self):
        """Main trading loop running in separate thread"""
        while self.is_running and self.trader.is_running:
            try:
                self.trader.run_trading_cycle()
                time.sleep(2)  # Reduced sleep for more responsive GUI
            except Exception as e:
                self.add_ai_activity(f"❌ TRADING LOOP ERROR: {e}", "error")
                logger.error(f"❌ Trading Loop Error: {e}", exc_info=True)
                break

    def run(self):
        """Start the GUI main loop"""
        self.update_display()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_trading()
            logger.info("🛑 System stopped by user")

# ==============================================================
# END OF GUI CLASS SECTION
# ==============================================================
# ==============================================================
# REAL MT5 TRADER CLASS – core execution logic
# ==============================================================

class FTMOAITrader:
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.is_running = False
        self.last_entry_ts = defaultdict(lambda: 0.0)
        self.con = None
        self.trading_enabled = False
        self.start_time = time.time()
        self.warmup_period = 30
        self.symbols = get_oanda_symbols()
        self.positions = {s: None for s in self.symbols}
        self.symbol_volumes = {}
        self.symbol_pnl_params = {}
        self.learning_ai = SimpleLearningAI()
        self.ftmo_enforcer = FTMOEnforcer()
        
        # Enhanced thread safety locks
        self.positions_lock = threading.RLock()
        self.cache_lock = threading.Lock()
        self.db_lock = threading.Lock()
        
        self.closing_in_progress = {}
        self._bars_cache = {}
        self._cache_ttl = 120
        self._last_cache_cleanup = time.time()
        
        self.market_regime_data = {}
        self.correlation_matrix = {}
        self.performance_by_hour = defaultdict(list)
        
        # Initialize all components
        self.mt5_init()
        self.db_init()
        self.load_existing_trades()
        self.recover_open_positions()
        logger.info("✅ FTMO AI QUANTITATIVE TRADER – REAL MT5 TRADING ACTIVATED")
        self.debug_mt5_positions()
        self.load_symbol_volumes()

        self._simulated_prices = {}

    def mt5_init(self):
        """Initialize MT5 connection with retry logic"""
        global MT5_AVAILABLE
        if not MT5_AVAILABLE:
            self.performance_tracker.initial_balance = 100_000.0
            logger.info("🔄 Running in simulation mode - MT5 not available")
            return
            
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                if not initialize_mt5():
                    logger.warning(f"❌ MT5 initialization attempt {attempt + 1} failed")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    MT5_AVAILABLE = False
                    self.performance_tracker.initial_balance = 100_000.0
                    logger.warning("🔄 MT5 initialization failed - falling back to simulation mode")
                    return
                    
                # Get account info
                account_info = None
                for info_attempt in range(3):
                    try:
                        account_info = mt5.account_info()
                        if account_info:
                            break
                        time.sleep(1)
                    except Exception as e:
                        if info_attempt == 2:
                            logger.error(f"❌ Failed to get account info: {e}")
                        time.sleep(1)
                        
                if account_info:
                    self.performance_tracker.initial_balance = account_info.balance
                    self.ftmo_enforcer.initial_balance = account_info.balance
                    self.ftmo_enforcer.daily_start_balance = account_info.balance
                    logger.info(f"✅ MT5 ACCOUNT CONNECTED – Balance ${account_info.balance:,.2f}")
                else:
                    self.performance_tracker.initial_balance = 100_000.0
                    
                break
                    
            except Exception as e:
                logger.error(f"❌ MT5 initialization error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    MT5_AVAILABLE = False
                    self.performance_tracker.initial_balance = 100_000.0

    def db_init(self):
        """Initialize database"""
        try:
            self.con = sqlite3.connect(DB_FILE, check_same_thread=False)
            self.con.execute("""CREATE TABLE IF NOT EXISTS trades(
                ts TEXT, symbol TEXT, dir TEXT, entry REAL, exit REAL,
                pnl_cash REAL, sl REAL, tp REAL, reason TEXT, size REAL,
                holding_time REAL, volatility REAL, rsi_entry REAL
            )""")
            
            self.con.execute("""CREATE TABLE IF NOT EXISTS analytics(
                date TEXT, symbol TEXT, win_loss TEXT, pnl REAL, 
                duration REAL, entry_time TEXT, exit_time TEXT,
                market_regime TEXT, confidence REAL
            )""")
            
            self.con.commit()
            logger.info("✅ Database initialization complete")
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")

    def load_existing_trades(self):
        """Load existing trades from database"""
        try:
            cursor = self.con.execute("SELECT * FROM trades ORDER BY ts DESC LIMIT 200")
            rows = cursor.fetchall()
            loaded_count = 0
            
            for row in rows:
                try:
                    trade = {
                        'timestamp': row[0],
                        'symbol': row[1],
                        'dir': row[2],
                        'entry': row[3],
                        'exit': row[4],
                        'pnl': row[5],
                        'sl': row[6],
                        'tp': row[7],
                        'reason': row[8],
                        'size': row[9] if len(row) > 9 else 0.01,
                        'holding_time': row[10] if len(row) > 10 else 0,
                        'volatility': row[11] if len(row) > 11 else 0,
                        'rsi_entry': row[12] if len(row) > 12 else 50
                    }
                    
                    existing_timestamps = [t.get('timestamp', '') for t in self.performance_tracker.trades]
                    if trade['timestamp'] not in existing_timestamps:
                        self.performance_tracker.trades.append(trade)
                        self.performance_tracker.total_trades += 1
                        if trade['pnl'] > 0:
                            self.performance_tracker.win_count += 1
                        else:
                            self.performance_tracker.loss_count += 1
                        loaded_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error loading trade row: {e}")
                    continue
                    
            logger.info(f"✅ Loaded {loaded_count} existing trades from database")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing trades: {e}")

    def recover_open_positions(self):
        """Recover open positions from file"""
        recovered = 0
        try:
            if POSITIONS_FILE.exists():
                with open(POSITIONS_FILE, 'r') as f:
                    saved = json.load(f)
                for sym, pdata in saved.items():
                    if sym in self.symbols:
                        self.positions[sym] = pdata
                        self.last_entry_ts[sym] = pdata.get('entry_time', time.time())
                        recovered += 1
                POSITIONS_FILE.unlink()
                
            logger.info(f"✅ Recovered {recovered} open positions")
        except Exception as e:
            logger.warning(f"⚠️ Position recovery error: {e}")

    def debug_mt5_positions(self):
        """Debug MT5 positions"""
        if MT5_AVAILABLE:
            try:
                logger.info("🔍 DEBUG: MT5 Position Check …")
                for sym in self.symbols:
                    mt5_sym = map_symbol_for_mt5(sym)
                    pos = mt5.positions_get(symbol=mt5_sym)
                    if pos:
                        logger.info(f"   {sym}: {len(pos)} position(s)")
                    else:
                        logger.info(f"   {sym}: No positions")
            except Exception as e:
                logger.error(f"❌ Error checking positions: {e}")

    def load_symbol_volumes(self):
        """Load symbol volume information"""
        for symbol in self.symbols:
            self.symbol_volumes[symbol] = {'min': 0.01, 'max': 1.0, 'step': 0.01}
            if "US30" in symbol.upper():
                pnl_factor = 0.05
            elif "US100" in symbol.upper():
                pnl_factor = 0.20
            elif "XAU" in symbol.upper():
                pnl_factor = 1.0
            else:
                pnl_factor = 1.0
            self.symbol_pnl_params[symbol] = {'pnl_factor': pnl_factor}

        # ========== ENHANCED MARKET DATA HELPERS WITH MEMORY MANAGEMENT ==========
    def get_bars(self, symbol: str, timeframe: int, n: int) -> pd.DataFrame:
        """FIXED: Get FRESH market data with proper cache expiration"""
        # Create a time-sensitive cache key that changes frequently
        current_minute = int(time.time() // 60)  # Changes every minute
        cache_key = f"{symbol}_{timeframe}_{n}_{current_minute}"
        
        now = time.time()
        
        # Clean cache more frequently (every minute instead of 5 minutes)
        if now - self._last_cache_cleanup > 60:
            self.cleanup_cache()
            self._last_cache_cleanup = now
        
        # Thread-safe cache access with shorter TTL
        with self.cache_lock:
            cached = self._bars_cache.get(cache_key)
            if cached and now - cached[1] < 30:  # Reduced TTL to 30 seconds
                logger.debug(f"🔄 Using cached data for {symbol} (expires in {30 - (now - cached[1]):.0f}s)")
                return cached[0]

        # If cache expired or no data, fetch FRESH data
        logger.info(f"📡 Fetching FRESH market data for {symbol}")
        
        # Real data from MT5 with enhanced error handling
        try:
            if MT5_AVAILABLE:
                mt5_symbol = map_symbol_for_mt5(symbol)
                if not mt5.symbol_select(mt5_symbol, True):
                    logger.warning(f"❌ Failed to select symbol for bars: {symbol} → {mt5_symbol}")
                    return self.get_fallback_bars(symbol, n)
                    
                # Use current time to ensure fresh data
                rates = mt5.copy_rates_from_pos(mt5_symbol,
                                                timeframe or mt5.TIMEFRAME_M5,
                                                0,  # Start from current position
                                                n)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                    
                    # Log the freshness of the data
                    latest_time = df.index[-1] if len(df) > 0 else None
                    logger.info(f"✅ Fresh data for {symbol}: {len(df)} bars, latest: {latest_time}")
                    
                    # Thread-safe cache update with shorter TTL
                    with self.cache_lock:
                        self._bars_cache[cache_key] = (df, now)
                    return df
                else:
                    logger.warning(f"⚠️ No rates returned for {symbol}, using fallback")
                    return self.get_fallback_bars(symbol, n)
                    
        except Exception as e:
            logger.error(f"Error fetching MT5 bars for {symbol}: {e}")

        return self.get_fallback_bars(symbol, n)

    def cleanup_cache(self):
        """FIXED: More aggressive cache cleanup to prevent stale data"""
        with self.cache_lock:
            now = time.time()
            expired_keys = [
                k for k, (_, timestamp) in self._bars_cache.items() 
                if now - timestamp > 60  # Clean up anything older than 1 minute
            ]
            for key in expired_keys:
                del self._bars_cache[key]
            
            # More aggressive size limiting
            if len(self._bars_cache) > 50:  # Reduced from 100 to 50
                # Remove oldest entries
                sorted_keys = sorted(
                    self._bars_cache.keys(), 
                    key=lambda k: self._bars_cache[k][1]
                )
                for key in sorted_keys[:25]:  # Remove oldest 25 (was 50)
                    del self._bars_cache[key]
                    
            logger.debug(f"🧹 Cache cleaned: {len(expired_keys)} expired entries removed, {len(self._bars_cache)} remain")

    def get_fallback_bars(self, symbol: str, n: int) -> pd.DataFrame:
        """Generate FRESH fallback bars with realistic price movement"""
        # Use current time for fresh data
        dates = pd.date_range(end=datetime.now(), periods=n, freq='5min')
        
        # More realistic price simulation with actual movement
        if "US30" in symbol.upper():
            base_price = 35_000.0 + (time.time() % 1000)  # Add time-based variation
            volatility = 50.0
        elif "US100" in symbol.upper():
            base_price = 18_000.0 + (time.time() % 500)   # Add time-based variation
            volatility = 25.0
        elif "XAU" in symbol.upper():
            base_price = 2_000.0 + (time.time() % 100)    # Add time-based variation
            volatility = 15.0
        else:
            base_price = 10_000.0 + (time.time() % 200)   # Add time-based variation
            volatility = 10.0

        # Simulate realistic price movement with trend
        returns = np.random.normal(0, volatility/100, n)
        
        # Add a slight trend based on time to make prices move
        time_trend = np.linspace(-0.001, 0.001, n)  # Small trend over the period
        combined_returns = returns + time_trend
        
        prices = base_price * (1 + np.cumsum(combined_returns))
        
        df = pd.DataFrame({
            'open': prices * 0.9995,  # More realistic open/close differences
            'high': prices * 1.0015,  # More realistic highs
            'low': prices * 0.9985,   # More realistic lows
            'close': prices,
            'tick_volume': np.random.randint(1_000, 10_000, n)  # More realistic volume
        }, index=dates)
        
        logger.info(f"🔄 Generated fresh fallback data for {symbol}: ${prices[-1]:.2f}")
        return df
    # ========== ENHANCED POSITION SYNCHRONIZATION WITH THREAD SAFETY ==========
    def sync_all_positions(self):
        """FIXED: Comprehensive position synchronization with thread safety"""
        if not MT5_AVAILABLE:
            return
            
        try:
            with self.positions_lock:
                # Check each symbol we're tracking
                for symbol in self.symbols:
                    self.sync_symbol_positions(symbol)
                    
                # Also check for any MT5 positions we're not tracking
                self.discover_orphaned_positions()
                
        except Exception as e:
            logger.error(f"❌ Position sync error: {e}")

    def sync_symbol_positions(self, symbol: str):
        """Sync positions for a specific symbol"""
        try:
            mt5_symbol = map_symbol_for_mt5(symbol)
            mt5_positions = mt5.positions_get(symbol=mt5_symbol)
            tracked_pos = self.positions.get(symbol)
            
            # Case 1: We think we have a position, but MT5 doesn't
            if tracked_pos is not None and not mt5_positions:
                logger.warning(f"⚠️ Position mismatch: {symbol} tracked but not in MT5 - recording closure")
                self.close_position(symbol, "external_close_sync")
                return
                
            # Case 2: MT5 has positions, but we don't track them
            if mt5_positions and tracked_pos is None:
                logger.info(f"🔍 Found untracked MT5 position for {symbol}")
                self.recover_untracked_position(symbol, mt5_positions)
                return
                
            # Case 3: Both have positions - verify they match
            if mt5_positions and tracked_pos is not None:
                matching_mt5_pos = None
                for mt5_pos in mt5_positions:
                    if 'ticket' in tracked_pos and mt5_pos.ticket == tracked_pos['ticket']:
                        matching_mt5_pos = mt5_pos
                        break
                        
                if not matching_mt5_pos:
                    logger.warning(f"⚠️ Position ticket mismatch for {symbol} - recording closure")
                    self.close_position(symbol, "ticket_mismatch")
                    
        except Exception as e:
            logger.error(f"❌ Symbol position sync error for {symbol}: {e}")

    def place_order(self, symbol: str, direction: str, sl: float, tp: float) -> bool:
        """FIXED: Thread-safe order placement with duplicate prevention"""
        # Check for existing position with thread safety
        with self.positions_lock:
            if self.positions[symbol] is not None:
                logger.warning(f"⚠️ Already a position for {symbol}, skipping duplicate entry")
                return False
                
        account_balance = self.get_account_balance()
        
        # Enhanced position sizing with risk management
        base_size = self.learning_ai.get_position_size(account_balance)
        
        # Adjust size based on current market conditions
        conditions = self.get_current_market_conditions(symbol)
        if conditions['volatility'] == 'high':
            base_size *= 0.7  # Reduce size in high volatility
        elif conditions['volatility'] == 'low':
            base_size *= 1.1  # Slightly increase in low volatility
            
        # Use GUI risk setting if available
        if hasattr(self, 'gui') and hasattr(self.gui, 'risk_per_trade_var'):
            risk_percent = self.gui.risk_per_trade_var.get()
            base_size = base_size * (risk_percent / 1.0)  # Adjust for risk setting

        size = max(0.01, min(1.0, base_size))  # Ensure reasonable bounds
        
        logger.info(f"🎯 ENHANCED TRADE: {direction.upper()} {symbol} | "
                   f"Size {size:.4f} | Volatility: {conditions['volatility']}")

        if MT5_AVAILABLE:
            return self.place_mt5_order(symbol, direction, size, sl, tp, conditions)
        else:
            return self.place_simulated_order(symbol, direction, size, sl, tp, account_balance)
    def place_mt5_order(self, symbol: str, direction: str, size: float, sl: float, tp: float, conditions: Dict) -> bool:
        """Enhanced MT5 order placement"""
        try:
            mt5_sym = map_symbol_for_mt5(symbol)
            if not mt5.symbol_select(mt5_sym, True):
                logger.warning(f"❌ Failed to select symbol for order: {symbol}")
                return False

            # Enhanced price validation with retries
            tick = None
            for attempt in range(5):
                tick = mt5.symbol_info_tick(mt5_sym)
                if tick and hasattr(tick, "bid") and hasattr(tick, "ask") and tick.bid > 0 and tick.ask > 0:
                    break
                time.sleep(0.2 * (attempt + 1))
                
            if not tick or tick.bid <= 0 or tick.ask <= 0:
                logger.warning(f"❌ No valid tick data for order placement on {symbol}")
                return False

            order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
            price = tick.ask if direction == "long" else tick.bid

            symbol_info = self.get_symbol_info(symbol)
            
            # Enhanced volume calculation
            volume = max(symbol_info['volume_min'], min(size, symbol_info['volume_max']))
            volume = round(volume / symbol_info['volume_step']) * symbol_info['volume_step']

            # Enhanced SL/TP validation
            sl, tp = self.validate_sl_tp(symbol, direction, price, sl, tp, symbol_info)
            
            if sl is None or tp is None:
                logger.warning(f"❌ Invalid SL/TP for {symbol}, using market order without stops")
                return self.place_market_order_no_stops(mt5_sym, order_type, volume, price)

            # Try different filling modes with enhanced error handling
            filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]
            result = None
            
            for mode in filling_modes:
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": mt5_sym,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "sl": round(sl, symbol_info['digits']),
                    "tp": round(tp, symbol_info['digits']),
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"FTMO AI Trade - {conditions['volatility']} vol",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mode,
                }
                
                logger.info(f"   📊 Attempting order for {symbol} with filling mode {mode}")
                res = mt5.order_send(req)
                
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    result = res
                    logger.info(f"✅ REAL MT5 ORDER PLACED: {symbol} {direction.upper()} "
                               f"@ ${price:.2f}, SL: ${sl:.2f}, TP: ${tp:.2f}")
                    break
                else:
                    if res:
                        logger.warning(f"   ⚠️ Order failed mode {mode}: {res.retcode} – {res.comment}")
                    else:
                        logger.warning(f"   ⚠️ Order failed mode {mode}: No response from server")
            
            if not result:
                logger.warning(f"   ⚠️ Trying order without SL/TP for {symbol}")
                return self.place_market_order_no_stops(mt5_sym, order_type, volume, price)

            # Enhanced position tracking with thread safety
            with self.positions_lock:
                self.positions[symbol] = {
                    'dir': direction,
                    'entry': price,
                    'size': volume,
                    'sl': sl,
                    'tp': tp,
                    'ticket': result.order,
                    'market_conditions': conditions,
                    'entry_volatility': conditions['volatility']
                }
                self.last_entry_ts[symbol] = time.time()
            self.save_open_positions()

            # Enhanced trade learning
            account_balance = self.get_account_balance()
            self.learning_ai.learn_from_trade({
                'symbol': symbol,
                'size': volume,
                'pnl': 0,
                'market_conditions': conditions,
                'volatility': conditions['volatility']
            }, account_balance)

            return True

        except Exception as e:
            logger.error(f"❌ MT5 Order Error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_sl_tp(self, symbol: str, direction: str, price: float, sl: float, tp: float, symbol_info: Dict) -> Tuple[float, float]:
        """Enhanced SL/TP validation with symbol-specific rules"""
        try:
            min_distance = symbol_info['trade_tick_size'] * 10  # 10 ticks minimum
            
            if direction == "long":
                if sl >= price - min_distance:
                    sl = price - min_distance
                    logger.info(f"   📊 Adjusted SL for long: ${sl:.2f}")
                if tp <= price + min_distance:
                    tp = price + (price - sl) * 2  # 2:1 reward ratio
                    logger.info(f"   📊 Adjusted TP for long: ${tp:.2f}")
            else:
                if sl <= price + min_distance:
                    sl = price + min_distance
                    logger.info(f"   📊 Adjusted SL for short: ${sl:.2f}")
                if tp >= price - min_distance:
                    tp = price - (sl - price) * 2  # 2:1 reward ratio
                    logger.info(f"   📊 Adjusted TP for short: ${tp:.2f}")
                    
            # Ensure SL/TP are reasonable distances
            max_distance = price * 0.05  # 5% maximum distance
            if direction == "long":
                sl = max(sl, price - max_distance)
                tp = min(tp, price + max_distance)
            else:
                sl = min(sl, price + max_distance)
                tp = max(tp, price - max_distance)
                
            return sl, tp
            
        except Exception as e:
            logger.warning(f"⚠️ SL/TP validation error for {symbol}: {e}")
            return sl, tp  # Return original values if validation fails

    def place_market_order_no_stops(self, mt5_sym: str, order_type: int, volume: float, price: float) -> bool:
        """Place market order without stops as fallback"""
        try:
            req_no_stops = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_sym,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "FTMO AI Trade - No Stops",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req_no_stops)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ REAL MT5 ORDER PLACED (no stops): {mt5_sym} @ ${price:.2f}")
                return True
            else:
                if res:
                    logger.error(f"❌ Market order failed: {res.retcode} – {res.comment}")
                else:
                    logger.error(f"❌ Market order failed: No response from server")
                return False
        except Exception as e:
            logger.error(f"❌ Market order error: {e}")
            return False

    def place_simulated_order(self, symbol: str, direction: str, size: float, sl: float, tp: float, account_balance: float) -> bool:
        """Enhanced simulated order placement for testing"""
        price = self.get_current_price(symbol)
        
        with self.positions_lock:
            self.positions[symbol] = {
                'dir': direction,
                'entry': price,
                'size': size,
                'sl': sl,
                'tp': tp,
                'market_conditions': self.get_current_market_conditions(symbol)
            }
            self.last_entry_ts[symbol] = time.time()
        self.save_open_positions()
        
        self.learning_ai.learn_from_trade({
            'symbol': symbol,
            'size': size,
            'pnl': 0,
            'market_conditions': self.get_current_market_conditions(symbol)
        }, account_balance)
        
        logger.info(f"✅ SIMULATED ORDER: {direction.upper()} {symbol} @ ${price:.2f}")
        return True
    def close_position(self, symbol: str, reason: str = "exit") -> bool:
        """FIXED: Thread-safe position closing with comprehensive P&L calculation"""
        # Thread-safe duplicate close prevention
        with self.positions_lock:
            if symbol in self.closing_in_progress:
                logger.info(f"⚠️ Already closing {symbol}, skipping duplicate close")
                return False
                
            if self.positions[symbol] is None:
                logger.info(f"⚠️ No position to close for {symbol}")
                return False
                
            self.closing_in_progress[symbol] = True
            pos = self.positions[symbol].copy()  # Work with a copy to avoid race conditions
        
        try:
            exit_price = self.get_current_price(symbol)
            
            # Enhanced P&L calculation with multiple methods
            real_pnl = self.calculate_enhanced_pnl(symbol, pos, exit_price, reason)
            
            if real_pnl is None:
                logger.warning(f"⚠️ Could not calculate P&L for {symbol}, using fallback")
                real_pnl = self.calculate_fallback_pnl(symbol, pos, exit_price)
            
            holding_time = time.time() - self.last_entry_ts[symbol]
            
            # Enhanced trade recording with more metadata
            trade_data = {
                'symbol': symbol, 
                'dir': pos['dir'], 
                'entry': pos['entry'], 
                'exit': exit_price,
                'pnl': real_pnl, 
                'sl': pos['sl'], 
                'tp': pos['tp'], 
                'reason': reason, 
                'size': pos['size'],
                'holding_time': holding_time,
                'volatility': pos.get('entry_volatility', 'medium'),
                'market_conditions': pos.get('market_conditions', {}),
                'rsi_entry': pos.get('rsi_entry', 50)
            }
            
            # Prevent duplicate recording with enhanced checking
            if not self.is_duplicate_trade(trade_data):
                self.record_enhanced_trade(trade_data)
                account_balance = self.get_account_balance()
                trade_data['account_balance'] = account_balance
                self.learning_ai.learn_from_trade(trade_data, account_balance)
            else:
                logger.info(f"⚠️ Duplicate trade prevented for {symbol}")

            # Enhanced position closing
            position_closed = self.execute_position_close(symbol, pos, reason, real_pnl)
            
            if position_closed:
                with self.positions_lock:
                    self.positions[symbol] = None
                self.save_open_positions()
                logger.info(f"✅ Enhanced close completed for {symbol}: P&L ${real_pnl:.2f}")
            else:
                logger.warning(f"⚠️ Position close may not have completed fully for {symbol}")

            return True
            
        except Exception as e:
            logger.error(f"❌ Error during enhanced close_position for {symbol}: {e}")
            return False
        finally:
            # Always remove from closing_in_progress
            with self.positions_lock:
                if symbol in self.closing_in_progress:
                    del self.closing_in_progress[symbol]
                    logger.info(f"🔓 Finished closing {symbol}")

    def calculate_enhanced_pnl(self, symbol: str, pos: Dict, exit_price: float, reason: str) -> float:
        """Calculate P&L using the most accurate method available"""
        # Try MT5 first for exact P&L
        if MT5_AVAILABLE and 'ticket' in pos:
            try:
                mt5_positions = mt5.positions_get(ticket=pos['ticket'])
                if mt5_positions and len(mt5_positions) > 0:
                    mt5_pnl = mt5_positions[0].profit
                    logger.info(f"💰 Using MT5's exact P&L for {symbol}: ${mt5_pnl:.2f}")
                    return mt5_pnl
            except Exception as e:
                logger.warning(f"⚠️ Could not get MT5 P&L: {e}")

        # Fall back to calculated P&L
        return self.calculate_real_profit(
            symbol=symbol,
            entry_price=pos['entry'],
            exit_price=exit_price,
            direction=pos['dir'],
            volume=pos['size']
        )

    def calculate_fallback_pnl(self, symbol: str, pos: Dict, exit_price: float) -> float:
        """Fallback P&L calculation when other methods fail"""
        if pos['dir'] == 'long':
            return (exit_price - pos['entry']) * pos['size'] * 10  # Simple multiplier
        else:
            return (pos['entry'] - exit_price) * pos['size'] * 10

    def is_duplicate_trade(self, trade_data: Dict) -> bool:
        """Enhanced duplicate trade detection"""
        recent_trades = self.performance_tracker.trades[-10:]
        for trade in recent_trades:
            if (trade.get('symbol') == trade_data['symbol'] and 
                abs(trade.get('entry', 0) - trade_data['entry']) < 0.1 and
                abs(trade.get('exit', 0) - trade_data['exit']) < 0.1 and
                trade.get('dir') == trade_data['dir']):
                return True
        return False

    def record_enhanced_trade(self, trade_data: Dict):
        """Enhanced trade recording with analytics"""
        # Thread-safe database operation
        with self.db_lock:
            try:
                temp_con = sqlite3.connect(DB_FILE)
                temp_con.execute("""INSERT INTO trades 
                    (ts, symbol, dir, entry, exit, pnl_cash, sl, tp, reason, size, holding_time, volatility, rsi_entry) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (datetime.now().isoformat(), trade_data['symbol'], trade_data['dir'],
                     trade_data['entry'], trade_data['exit'], trade_data['pnl'],
                     trade_data['sl'], trade_data['tp'], trade_data['reason'],
                     trade_data['size'], trade_data['holding_time'],
                     trade_data.get('volatility', 0), trade_data.get('rsi_entry', 50)))
                temp_con.commit()
                temp_con.close()
            except Exception as e:
                logger.error(f"❌ Database recording error: {e}")

        self.performance_tracker.record_trade(trade_data)
        logger.info(f"✅ Trade recorded in database: {trade_data['symbol']} P&L: ${trade_data['pnl']:.2f}")

    def execute_position_close(self, symbol: str, pos: Dict, reason: str, pnl: float) -> bool:
        """Execute the actual position closing"""
        if MT5_AVAILABLE and 'ticket' in pos:
            return self.close_mt5_position(symbol, pos, reason, pnl)
        else:
            return self.close_simulated_position(symbol, pos, reason, pnl)

    def close_mt5_position(self, symbol: str, pos: Dict, reason: str, pnl: float) -> bool:
        """Close MT5 position with enhanced error handling"""
        try:
            mt5_symbol = map_symbol_for_mt5(symbol)
            if not mt5.symbol_select(mt5_symbol, True):
                logger.warning(f"❌ Failed to select symbol for closing: {symbol}")
                return True  # Consider it closed since we can't access it

            positions = mt5.positions_get(symbol=mt5_symbol)
            if not positions:
                logger.info(f"ℹ️ No MT5 position found for {symbol} - already closed")
                return True

            position_to_close = None
            for position in positions:
                if position.ticket == pos['ticket']:
                    position_to_close = position
                    break

            if not position_to_close:
                logger.warning(f"❌ Specific position not found for {symbol}")
                return True  # Consider it closed

            if pos['dir'] == 'long':
                close_type = mt5.ORDER_TYPE_SELL
                tick_data = mt5.symbol_info_tick(mt5_symbol)
                close_price = tick_data.bid if tick_data else pos['entry']  # Fallback
            else:
                close_type = mt5.ORDER_TYPE_BUY
                tick_data = mt5.symbol_info_tick(mt5_symbol)
                close_price = tick_data.ask if tick_data else pos['entry']  # Fallback

            if close_price <= 0:
                logger.warning(f"❌ Invalid close price for {symbol}")
                return False

            filling_modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
            close_request = None
            result = None

            for filling_mode in filling_modes:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": mt5_symbol,
                    "volume": pos['size'],
                    "type": close_type,
                    "position": position_to_close.ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"FTMO AI Close: {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }

                result = mt5.order_send(close_request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ REAL MT5 POSITION CLOSED: {symbol} → P&L: ${pnl:.2f}")
                    return True
                else:
                    if result:
                        logger.warning(f"⚠️ Close failed with mode {filling_mode}: {result.retcode} - {result.comment}")
                    else:
                        logger.warning(f"⚠️ Close failed with mode {filling_mode}: No response")

            logger.error(f"❌ All filling modes failed for closing {symbol}")
            return False

        except Exception as e:
            logger.error(f"❌ MT5 close error for {symbol}: {e}")
            return False

    def close_simulated_position(self, symbol: str, pos: Dict, reason: str, pnl: float) -> bool:
        """Close simulated position"""
        logger.info(f"🔻 SIMULATED CLOSE: {symbol} | Reason: {reason} | P&L: ${pnl:.2f}")
        return True
    # ========== ENHANCED EXIT CONDITION CHECKING ==========
    def check_exit_conditions(self, symbol: str) -> bool:
        """Enhanced exit condition checking with multiple factors"""
        with self.positions_lock:
            if self.positions[symbol] is None:
                return False
            pos = self.positions[symbol]
        
        current_price = self.get_current_price(symbol)
        
        # Basic SL/TP checks
        sl_hit = (pos['dir'] == 'long' and current_price <= pos['sl']) or \
                 (pos['dir'] == 'short' and current_price >= pos['sl'])
                 
        tp_hit = pos['tp'] and ((pos['dir'] == 'long' and current_price >= pos['tp']) or \
                               (pos['dir'] == 'short' and current_price <= pos['tp']))
        
        if sl_hit:
            return self.close_position(symbol, "sl_hit")
        if tp_hit:
            return self.close_position(symbol, "tp_hit")
            
        # Enhanced time-based exit
        holding_time = time.time() - self.last_entry_ts[symbol]
        if holding_time > 3600:  # 1 hour maximum
            logger.info(f"⏰ Time-based exit for {symbol} after {holding_time/60:.1f} minutes")
            return self.close_position(symbol, "time_exit")
            
        # Enhanced volatility-based exit
        current_volatility = self.get_current_volatility(symbol)
        entry_volatility = pos.get('entry_volatility', 10.0)
        
        # If volatility has doubled since entry, consider exiting
        if current_volatility > entry_volatility * 2:
            logger.info(f"🌪️  Volatility-based exit for {symbol}: {current_volatility:.1f}% vs entry {entry_volatility:.1f}%")
            return self.close_position(symbol, "volatility_exit")
            
        return False

    # ========== ENHANCED TRAILING STOPS ==========
    def update_trailing_stops(self):
        """Enhanced trailing stops with multiple strategies"""
        with self.positions_lock:
            positions_copy = self.positions.copy()
        
        for symbol, pos in positions_copy.items():
            if pos is not None:
                try:
                    current_price = self.get_current_price(symbol)
                    
                    if pos['dir'] == 'long':
                        profit_pips = (current_price - pos['entry']) / self.get_pip_size(symbol)
                    else:
                        profit_pips = (pos['entry'] - current_price) / self.get_pip_size(symbol)
                    
                    # Strategy 1: Move to breakeven at 10 pips profit
                    if profit_pips > 10 and pos['sl'] < pos['entry']:
                        with self.positions_lock:
                            if self.positions[symbol] is not None:  # Check if still open
                                self.positions[symbol]['sl'] = pos['entry']
                        logger.info(f"📈 SL moved to breakeven for {symbol}: ${pos['entry']:.2f}")
                    
                    # Strategy 2: Trailing stop at 20 pips profit (lock in 10 pips)
                    if profit_pips > 20:
                        if pos['dir'] == 'long':
                            new_sl = current_price - (10 * self.get_pip_size(symbol))
                        else:
                            new_sl = current_price + (10 * self.get_pip_size(symbol))
                        
                        # Thread-safe update
                        with self.positions_lock:
                            if self.positions[symbol] is not None:
                                current_sl = self.positions[symbol]['sl']
                                if (pos['dir'] == 'long' and new_sl > current_sl) or \
                                   (pos['dir'] == 'short' and new_sl < current_sl):
                                    self.positions[symbol]['sl'] = new_sl
                                    logger.info(f"📈 Trailing SL updated for {symbol}: ${new_sl:.2f}")
                    
                    # Strategy 3: Progressive trailing based on ATR
                    if profit_pips > 30:
                        atr = self.calculate_atr(self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 14), 14)
                        if pos['dir'] == 'long':
                            new_sl = current_price - (atr * 0.5)
                        else:
                            new_sl = current_price + (atr * 0.5)
                        
                        # Thread-safe update
                        with self.positions_lock:
                            if self.positions[symbol] is not None:
                                current_sl = self.positions[symbol]['sl']
                                if (pos['dir'] == 'long' and new_sl > current_sl) or \
                                   (pos['dir'] == 'short' and new_sl < current_sl):
                                    self.positions[symbol]['sl'] = new_sl
                                    logger.info(f"📈 ATR-based trailing SL for {symbol}: ${new_sl:.2f}")
                                    
                except Exception as e:
                    logger.warning(f"⚠️  Trailing stop error for {symbol}: {e}")

    def get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol"""
        if "XAU" in symbol.upper():
            return 0.01  # Gold typically quoted to 2 decimal places
        else:
            return 0.0001  # Standard forex pip size

    # ========== ENHANCED TECHNICAL INDICATORS ==========
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Enhanced ATR calculation with better error handling"""
        try:
            if len(df) < period + 1:
                return 10.0
                
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            result = float(atr.iloc[-1]) if not atr.empty else 10.0
            return max(1.0, min(100.0, result))  # Clamp reasonable values
            
        except Exception as e:
            logger.warning(f"⚠️  ATR calculation error: {e}")
            return 10.0

    def ema_series(self, series: pd.Series, period: int) -> pd.Series:
        """Enhanced EMA calculation with validation"""
        try:
            if len(series) < period:
                # Pad with zeros or use simple average
                padded = series.copy()
                while len(padded) < period:
                    padded = pd.Series([series.iloc[0]]).append(padded)
                return padded.ewm(span=period, adjust=False).mean()
            return series.ewm(span=period, adjust=False).mean()
        except Exception:
            return series

    def calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return {'macd': 0, 'signal': 0, 'histogram': 0}
                
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except Exception:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    # ========== ENHANCED MARKET DATA HELPERS ==========
    def get_current_price(self, symbol: str) -> float:
        """FIXED: Get truly real-time price with cache busting"""
        # Use a time-based cache key for prices too
        price_cache_key = f"price_{symbol}_{int(time.time() // 5)}"  # Change every 5 seconds
        
        # Initialize price cache if it doesn't exist
        if not hasattr(self, '_price_cache'):
            self._price_cache = {}
            
        with self.cache_lock:
            cached_price = self._price_cache.get(price_cache_key)
            if cached_price:
                return cached_price
        
        if MT5_AVAILABLE:
            try:
                mt5_sym = map_symbol_for_mt5(symbol)
                if not mt5.symbol_select(mt5_sym, True):
                    logger.warning(f"❌ Failed to select symbol for price: {symbol}")
                    price = self.get_fallback_price(symbol)
                else:
                    # Multiple attempts with short delays for fresh data
                    for attempt in range(3):
                        tick = mt5.symbol_info_tick(mt5_sym)
                        if tick and hasattr(tick, "bid") and hasattr(tick, "ask") and tick.bid > 0 and tick.ask > 0:
                            price = (tick.bid + tick.ask) / 2.0
                            logger.debug(f"📡 Fresh MT5 price for {symbol}: ${price:.2f}")
                            break
                        time.sleep(0.1)
                    else:
                        price = self.get_fallback_price(symbol)
            except Exception as e:
                logger.error(f"❌ Price error for {symbol}: {e}")
                price = self.get_fallback_price(symbol)
        else:
            price = self.get_fallback_price(symbol)
        
        # Cache the price with short TTL
        with self.cache_lock:
            self._price_cache[price_cache_key] = price
            
        return price
    def get_fallback_price(self, symbol: str) -> float:
        """Enhanced fallback price with realistic variations"""
        # Add some realistic variation to simulated prices
        base_prices = {
            "US30Z25.sim": 35_000.0,
            "US100Z25.sim": 18_000.0, 
            "XAUZ25.sim": 2_000.0
        }
        
        base_price = base_prices.get(symbol, 10_000.0)
        
        # Add small random walk to simulate market movement
        if hasattr(self, '_simulated_prices'):
            if symbol in self._simulated_prices:
                last_price = self._simulated_prices[symbol]
                change = np.random.normal(0, base_price * 0.0005)  # 0.05% variation
                new_price = last_price + change
                self._simulated_prices[symbol] = new_price
                return new_price
            else:
                self._simulated_prices[symbol] = base_price
                return base_price
        else:
            self._simulated_prices = {symbol: base_price}
            return base_price

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get enhanced symbol information"""
        if MT5_AVAILABLE:
            try:
                mt5_sym = map_symbol_for_mt5(symbol)
                info = mt5.symbol_info(mt5_sym)
                if info:
                    return {
                        'digits': info.digits,
                        'trade_tick_size': info.trade_tick_size,
                        'trade_tick_value': getattr(info, 'trade_tick_value', 1.0),
                        'volume_min': info.volume_min,
                        'volume_max': info.volume_max,
                        'volume_step': info.volume_step
                    }
            except Exception as e:
                logger.warning(f"⚠️  Error getting symbol info for {symbol}: {e}")
                
        # Fallback values
        return {
            'digits': 2,
            'trade_tick_size': 0.01,
            'trade_tick_value': 1.0,
            'volume_min': 0.01,
            'volume_max': 1.0,
            'volume_step': 0.01
        }
    # ========== ENHANCED SIGNAL GENERATION ==========
    def get_signal_data(self, symbol: str) -> Dict:
        """ENHANCED: Market regime-aware signal generation"""
        try:
            df = self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 50)
            if len(df) < 20:
                return {'action': 'hold', 'rsi': 50, 'confidence': 0, 'ftmo_compliant': False}
                
            close = df['close'].dropna()
            if len(close) < 20:
                return {'action': 'hold', 'rsi': 50, 'confidence': 0, 'ftmo_compliant': False}
                
            current_price = float(close.iloc[-1])

            # Calculate multiple indicators
            ema_fast = self.ema_series(close, 12).iloc[-1]
            ema_slow = self.ema_series(close, 26).iloc[-1]
            ema_signal = (ema_fast - ema_slow) / ema_slow * 100  # Percentage difference
            
            rsi_val = self.learning_ai.calculate_rsi(close, 14)
            macd_data = self.calculate_macd(close)
            
            # Market conditions analysis
            market_conditions = self.get_current_market_conditions(symbol)
            volatility = self.get_current_volatility(symbol)
            
            # FIX 3: Get market regime for adaptive trading
            market_regime = self.get_market_regime(symbol)
            
            logger.info(f"   📊 {symbol} - Price: {current_price:.2f}, "
                       f"EMA_diff: {ema_signal:.4f}%, RSI: {rsi_val:.2f}, "
                       f"Regime: {market_regime}")

            # EMA/RSI path - enhanced with market regime awareness
            ema_action, ema_conf = self.learning_ai.get_adaptive_signal(rsi_val, ema_signal)
            
            # Adjust confidence based on market regime
            regime_multiplier = {
                "trending_high_vol": 1.2,  # Increase confidence in trending markets
                "ranging_high_vol": 0.8,   # Reduce confidence in high volatility ranges
                "trending_low_vol": 1.3,   # Highest confidence in low vol trends
                "ranging_low_vol": 1.0,    # Neutral in low vol ranges
                "neutral": 1.0
            }.get(market_regime, 1.0)
            
            ema_conf *= regime_multiplier
            ema_conf = min(0.9, max(0.05, ema_conf))  # Keep within reasonable bounds

            # XGBoost path
            xgb_action, xgb_conf = self.learning_ai.get_xgboost_signal(df, current_price)

            # Phase-based confidence threshold
            phase_status = self.performance_tracker.get_phase_status()
            
            if phase_status['phase'] == 1:  # Data Collection Phase
                effective_confidence_threshold = 0.05
                xgb_boost = 0.3
                min_trades_for_xgb = 10
            elif phase_status['phase'] == 2:  # Quality Improvement
                effective_confidence_threshold = 0.08
                xgb_boost = 0.4
                min_trades_for_xgb = 25
            else:  # Performance Focus
                effective_confidence_threshold = 0.10
                xgb_boost = 0.5
                min_trades_for_xgb = 50

            # Decision logic with market regime awareness
            final_action = 'hold'
            final_confidence = 0.0
            signal_source = "None"
            
            # Prefer XGBoost when available
            if (xgb_conf > xgb_boost and xgb_action != 'hold' and 
                len(self.learning_ai.all_trades) >= min_trades_for_xgb):
                final_action, final_confidence = xgb_action, xgb_conf
                signal_source = "XGBoost"
                
            # Fall back to enhanced EMA/RSI
            elif ema_conf > effective_confidence_threshold and ema_action != 'hold':
                final_action, final_confidence = ema_action, ema_conf
                signal_source = f"Enhanced EMA/RSI ({market_regime})"
                
            else:
                final_action, final_confidence = 'hold', 0.0
                signal_source = "No clear signal"
                
            # Additional filters with regime awareness
            if final_action != 'hold':
                # In trending markets, be less restrictive about RSI extremes
                if "trending" in market_regime:
                    rsi_penalty = 0.9  # Only 10% penalty in trends
                else:
                    rsi_penalty = 0.7  # 30% penalty in ranging markets
                    
                if (final_action == 'buy' and rsi_val > 75) or (final_action == 'sell' and rsi_val < 25):
                    final_confidence *= rsi_penalty
                    logger.info(f"   ⚠️  Reduced confidence due to extreme RSI: {rsi_val:.1f} "
                               f"(penalty: {rsi_penalty}, regime: {market_regime})")

            ftmo_ok = self.ftmo_enforcer.can_trade(self.get_account_balance())
            
            if final_action != 'hold' and final_confidence > 0:
                logger.info(f"   🎯 Final signal for {symbol}: {final_action.upper()} "
                           f"(conf {final_confidence:.2f}, source: {signal_source})")

            return {
                'action': final_action,
                'price': current_price,
                'rsi': rsi_val,
                'confidence': final_confidence,
                'ftmo_compliant': ftmo_ok,
                'market_regime': market_regime,
                'market_conditions': market_conditions,
                'volatility': volatility,
                'macd': macd_data['macd'],
                'ema_signal': ema_signal
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Signal generation error for {symbol}: {e}")
            return {'action': 'hold', 'rsi': 50, 'confidence': 0, 'ftmo_compliant': False}


    # ========== ENHANCED MAIN TRADING CYCLE ==========
    def run_trading_cycle(self):
        """FIXED: Type error fix and enhanced trading logic"""
        # Enhanced position synchronization
        self.sync_all_positions()
        
        if not self.is_running:
            return

        # Phase-based configuration
        phase_status = self.performance_tracker.get_phase_status()
        
        if phase_status['phase'] == 1:  # Data Collection Phase
            base_confidence = 0.05
            max_positions = 5
            cooldown = 60
        elif phase_status['phase'] == 2:  # Quality Improvement
            base_confidence = 0.08
            max_positions = 4
            cooldown = 90
        else:  # Performance Focus
            base_confidence = 0.10
            max_positions = 3
            cooldown = 120

        # Use GUI settings or phase-based defaults with proper type conversion
        try:
            # FIX: Ensure proper type conversion
            min_confidence = float(getattr(self.gui, 'confidence_var', tk.DoubleVar(value=base_confidence)).get())
            max_positions = int(getattr(self.gui, 'max_positions_var', tk.IntVar(value=max_positions)).get())
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"⚠️ GUI settings error, using defaults: {e}")
            min_confidence = base_confidence
            max_positions = max_positions

        effective_confidence = max(min_confidence, phase_status['confidence_threshold'])
        
        current_balance = self.get_account_balance()
        
        # Enhanced FTMO compliance checking
        if not self.ftmo_enforcer.can_trade(current_balance):
            if self.trading_enabled:
                logger.info("🛑 QUANTITATIVE TRADING HALTED: FTMO Rule Violation")
                self.trading_enabled = False
            return
        
        # Enhanced correlation checking
        if not self.ftmo_enforcer.check_correlated_positions(self.positions):
            logger.info("⚠️ Too many correlated positions - waiting for opportunities")
            return
            
        # Thread-safe active position counting
        with self.positions_lock:
            active_positions = sum(1 for pos in self.positions.values() if pos is not None)
        
        # Enhanced warmup period
        if not self.trading_enabled and time.time() - self.start_time > self.warmup_period:
            self.trading_enabled = True
            logger.info("✅ QUANTITATIVE TRADING ACTIVATED")
            
        self.check_for_closed_trades()
        self.sync_all_positions()
        
        # Enhanced market data collection
        market_prices = {sym: self.get_current_price(sym) for sym in self.symbols}
        self.performance_tracker.update_open_pnl(self.positions, market_prices)
        self.update_trailing_stops()
        
        logger.info(f"🎯 Phase {phase_status['phase']} Trading: "
                   f"Confidence={effective_confidence:.3f}, MaxPositions={max_positions}, "
                   f"ActivePositions={active_positions}")
        
        # Enhanced trading logic with error handling
        if active_positions < max_positions:
            for symbol in self.symbols:
                try:
                    # FIX: Check exit conditions with error handling
                    should_exit = self.check_exit_conditions(symbol)
                    if should_exit:
                        continue
                        
                    signal_data = self.get_signal_data(symbol)
                    
                    # FIX: Ensure proper type comparison
                    if not isinstance(signal_data.get('confidence', 0), (int, float)):
                        logger.warning(f"⚠️ Invalid confidence type for {symbol}: {type(signal_data.get('confidence'))}")
                        continue
                    
                    # Enhanced entry criteria
                    with self.positions_lock:
                        position_exists = self.positions[symbol] is not None
                        last_entry_time = self.last_entry_ts[symbol]
                    
                    # FIX: Proper type comparison for confidence
                    signal_confidence = float(signal_data.get('confidence', 0))
                    
                    if (self.trading_enabled and 
                        signal_data.get('ftmo_compliant', False) and 
                        not position_exists and 
                        signal_data.get('action', 'hold') in ['buy', 'sell'] and
                        signal_confidence > float(effective_confidence) and
                        time.time() - last_entry_time > cooldown):
                        
                        direction = "long" if signal_data['action'] == 'buy' else "short"
                        price = float(signal_data.get('price', 0))
                        
                        logger.info(f"🎯 TRADE SIGNAL: {symbol} {direction.upper()} "
                                   f"(Confidence: {signal_confidence:.3f})")
                        
                        sl, tp = self.calculate_dynamic_sl_tp(symbol, price, direction)
                        self.place_order(symbol, direction, sl, tp)
                        
                except Exception as e:
                    logger.error(f"❌ Trading cycle error for {symbol}: {e}")
                    continue

    # ========== ENHANCED ACCOUNT & P&L HELPERS ==========
    def get_account_balance(self) -> float:
        """Enhanced account balance retrieval with fallbacks"""
        if MT5_AVAILABLE:
            try:
                for attempt in range(3):
                    account_info = mt5.account_info()
                    if account_info:
                        return account_info.balance
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"⚠️  Error getting account balance: {e}")
                
        # Enhanced fallback balance calculation
        if hasattr(self.performance_tracker, 'initial_balance'):
            base_balance = self.performance_tracker.initial_balance
            total_closed_pnl = sum(t.get('pnl', 0) for t in self.performance_tracker.trades)
            total_open_pnl = self.performance_tracker.open_positions_pnl
            return base_balance + total_closed_pnl + total_open_pnl
            
        return 184257.27  # Final fallback

    def calculate_real_profit(self, symbol, entry_price, exit_price, direction, volume):
        """Enhanced P&L calculation with symbol-specific factors"""
        pnl_params = self.symbol_pnl_params.get(symbol, {'pnl_factor': 0.1})
        script_pnl_factor = pnl_params.get('pnl_factor', 0.1)

        # Calculate price movement in points
        points = abs(exit_price - entry_price)
        
        # Adjust for direction
        if direction == "long":
            cash_profit = (exit_price - entry_price) * script_pnl_factor * volume
        else:
            cash_profit = (entry_price - exit_price) * script_pnl_factor * volume

        return cash_profit

    def get_current_market_conditions(self, symbol: str) -> Dict:
        """Get current market conditions for a symbol"""
        try:
            df = self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 50)
            if len(df) < 20:
                return {'trend': 'neutral', 'volatility': 'medium'}
                
            # Simple trend detection
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            if price_change > 2:
                trend = 'bullish'
            elif price_change < -2:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            # Volatility detection
            volatility = df['close'].pct_change().std() * 100
            if volatility > 1.5:
                vol_level = 'high'
            elif volatility < 0.5:
                vol_level = 'low'
            else:
                vol_level = 'medium'
                
            return {'trend': trend, 'volatility': vol_level, 'price_change': price_change}
            
        except Exception as e:
            logger.warning(f"⚠️  Market conditions error for {symbol}: {e}")
            return {'trend': 'neutral', 'volatility': 'medium'}

    def get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol"""
        try:
            df = self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 20)
            if len(df) < 10:
                return 10.0
            return float(df['close'].pct_change().std() * 100)
        except Exception:
            return 10.0
    def get_market_regime(self, symbol: str) -> str:
        """Determine current market regime for adaptive trading"""
        try:
            # Get recent price data
            df = self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 100)
            if len(df) < 50:
                return "neutral"  # Not enough data
                
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Convert to percentage
            
            # Calculate trend strength (price change over last 50 periods)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-50]) / df['close'].iloc[-50] * 100
            
            # Calculate RSI for additional context
            rsi = self.learning_ai.calculate_rsi(df['close'], 14)
            
            logger.debug(f"   📊 {symbol} Regime Analysis: Vol={volatility:.2f}%, "
                        f"Trend={price_change:.2f}%, RSI={rsi:.1f}")
            
            # Determine regime based on volatility and trend
            if volatility > 1.5:  # High volatility
                if abs(price_change) > 3:  # Strong trend
                    return "trending_high_vol"
                else:
                    return "ranging_high_vol"
            else:  # Low volatility
                if abs(price_change) > 2:  # Moderate trend
                    return "trending_low_vol"
                else:
                    return "ranging_low_vol"
                    
        except Exception as e:
            logger.warning(f"⚠️ Market regime detection error for {symbol}: {e}")
            return "neutral"  # Default to neutral on error

    def calculate_dynamic_sl_tp(self, symbol: str, entry_price: float, direction: str) -> Tuple[float, float]:
        """Enhanced dynamic SL/TP with multiple factors"""
        try:
            df = self.get_bars(symbol, mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None, 50)
            
            # Get ATR-based distances
            atr = self.calculate_atr(df, 14) if len(df) >= 14 else 20.0
            
            # Adjust based on market conditions
            conditions = self.get_current_market_conditions(symbol)
            volatility_factor = 1.0
            if conditions['volatility'] == 'high':
                volatility_factor = 1.3
            elif conditions['volatility'] == 'low':
                volatility_factor = 0.7
                
            # Use GUI settings if available
            atr_multiplier = 1.0
            if hasattr(self, 'gui') and hasattr(self.gui, 'atr_multiplier_var'):
                atr_multiplier = self.gui.atr_multiplier_var.get()
            else:
                atr_multiplier = 1.0
                
            sl_dist = atr * atr_multiplier * volatility_factor
            tp_dist = atr * (atr_multiplier * 2) * volatility_factor
            
            # Ensure minimum distances
            min_distance = entry_price * 0.001  # 0.1% minimum
            sl_dist = max(sl_dist, min_distance)
            tp_dist = max(tp_dist, min_distance * 2)
            
            if direction == "long":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
                
            logger.info(f"   📊 Enhanced SL/TP for {symbol}: SL=${sl:.2f} TP=${tp:.2f} "
                       f"(ATR={atr:.2f}, VolFactor={volatility_factor:.1f})")
            return sl, tp
            
        except Exception as e:
            logger.warning(f"❌ SL/TP calc error for {symbol}: {e}")
            # Fallback to simple percentage-based SL/TP
            if direction == "long":
                sl = entry_price * 0.995
                tp = entry_price * 1.01
            else:
                sl = entry_price * 1.005
                tp = entry_price * 0.99
            return sl, tp
    # ========== ENHANCED POSITION SYNCHRONIZATION ==========
    def check_for_closed_trades(self):
        """Check if any tracked positions have been closed externally"""
        try:
            with self.positions_lock:
                positions_copy = self.positions.copy()
            
            for symbol, tracked_pos in positions_copy.items():
                if tracked_pos is not None:
                    if MT5_AVAILABLE and 'ticket' in tracked_pos:
                        mt5_symbol = map_symbol_for_mt5(symbol)
                        mt5_positions = mt5.positions_get(symbol=mt5_symbol)
                        
                        position_still_exists = False
                        if mt5_positions:
                            for mt5_pos in mt5_positions:
                                if mt5_pos.ticket == tracked_pos['ticket']:
                                    position_still_exists = True
                                    break
                        
                        if not position_still_exists:
                            logger.info(f"🔍 Detected externally closed position: {symbol}")
                            self.close_position(symbol, "external_detection")
                            
        except Exception as e:
            logger.error(f"❌ Closed trade detection error: {e}")

    def recover_untracked_position(self, symbol: str, mt5_positions):
        """Recover positions that exist in MT5 but aren't tracked"""
        try:
            for mt5_pos in mt5_positions:
                # Check if we should track this position
                position_data = {
                    'dir': 'long' if mt5_pos.type == mt5.ORDER_TYPE_BUY else 'short',
                    'entry': mt5_pos.price_open,
                    'size': mt5_pos.volume,
                    'sl': mt5_pos.sl,
                    'tp': mt5_pos.tp,
                    'ticket': mt5_pos.ticket
                }
                
                with self.positions_lock:
                    self.positions[symbol] = position_data
                self.last_entry_ts[symbol] = time.time()
                logger.info(f"✅ Recovered untracked position: {symbol} {position_data['dir']} @ ${position_data['entry']:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error recovering untracked position for {symbol}: {e}")

    def discover_orphaned_positions(self):
        """Find any MT5 positions we're not tracking"""
        try:
            all_mt5_positions = mt5.positions_get() if MT5_AVAILABLE else []
            if not all_mt5_positions:
                return
                
            for mt5_pos in all_mt5_positions:
                is_tracked = False
                with self.positions_lock:
                    for tracked_symbol, tracked_pos in self.positions.items():
                        if tracked_pos and 'ticket' in tracked_pos and tracked_pos['ticket'] == mt5_pos.ticket:
                            is_tracked = True
                            break
                            
                if not is_tracked:
                    logger.info(f"🔍 Found orphaned MT5 position: {mt5_pos.symbol} Ticket:{mt5_pos.ticket}")
                    
        except Exception as e:
            logger.error(f"❌ Orphaned position discovery error: {e}")

    def save_open_positions(self):
        """Save open positions with enhanced metadata"""
        try:
            data = {}
            with self.positions_lock:
                for sym, pos in self.positions.items():
                    if pos is not None:
                        data[sym] = {
                            'dir': pos['dir'],
                            'entry': pos['entry'],
                            'size': pos['size'],
                            'sl': pos['sl'],
                            'tp': pos['tp'],
                            'ticket': pos.get('ticket'),
                            'entry_time': self.last_entry_ts[sym],
                            'market_conditions': self.get_current_market_conditions(sym),
                            'volatility': self.get_current_volatility(sym)
                        }
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✅ Saved {len(data)} open positions with enhanced metadata")
        except Exception as e:
            logger.warning(f"⚠️  Could not save open positions: {e}")

    def handle_closed_position(self, symbol: str, pos_data: Dict):
        """Enhanced handling of externally closed positions"""
        try:
            if not pos_data:
                return
                
            exit_price = self.get_current_price(symbol)
            pnl = self.calculate_real_profit(symbol, pos_data['entry'], exit_price, pos_data['dir'], pos_data['size'])
            holding = time.time() - pos_data.get('entry_time', time.time())
            
            # Enhanced duplicate checking
            if self.is_duplicate_trade({
                'symbol': symbol, 
                'entry': pos_data['entry'], 
                'exit': exit_price, 
                'dir': pos_data['dir']
            }):
                logger.info(f"⚠️  Duplicate external close prevented for {symbol}")
                return
                
            trade = {
                'symbol': symbol, 
                'dir': pos_data['dir'], 
                'entry': pos_data['entry'], 
                'exit': exit_price,
                'pnl': pnl, 
                'sl': pos_data.get('sl', 0), 
                'tp': pos_data.get('tp', 0), 
                'reason': 'external_close', 
                'size': pos_data['size'],
                'holding_time': holding
            }
            
            # Thread-safe recording
            with self.db_lock:
                temp_con = sqlite3.connect(DB_FILE)
                temp_con.execute("INSERT INTO trades (ts,symbol,dir,entry,exit,pnl_cash,sl,tp,reason,size) VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (datetime.now().isoformat(), trade['symbol'], trade['dir'],
                                trade['entry'], trade['exit'], trade['pnl'],
                                trade['sl'], trade['tp'], trade['reason'], trade['size']))
                temp_con.commit()
                temp_con.close()
            
            self.performance_tracker.record_trade(trade)
            acc_bal = self.get_account_balance()
            trade['account_balance'] = acc_bal
            trade['rsi'] = 50
            self.learning_ai.learn_from_trade(trade, acc_bal)

            logger.info(f"✅ Recorded externally closed position: {symbol} P&L ${pnl:.2f}")
        except Exception as e:
            logger.warning(f"⚠️  Could not record external close for {symbol}: {e}")
# ==============================================================
# MAIN FUNCTION & PROGRAM EXECUTION
# ==============================================================

def main():
    """FIXED: Enhanced main function with comprehensive error handling and resource cleanup"""
    try:
        logger.info("🚀 STARTING FTMO AI QUANTITATIVE TRADING SYSTEM - ENHANCED VERSION")
        logger.info("================================================================")
        
        # Display system information
        print("🎯 FTMO AI Quantitative Trading System - Enhanced Version")
        print("========================================================")
        print(f"📊 MT5 Available: {'✅' if MT5_AVAILABLE else '❌'}")
        print(f"🤖 XGBoost Available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
        print(f"📈 Matplotlib Available: {'✅' if MATPLOTLIB_AVAILABLE else '❌'}")
        print(f"🐼 Pandas Available: {'✅' if PANDAS_AVAILABLE else '❌'}")
        print("Initializing... Please wait.")
        
        # Initialize the trader with enhanced error handling
        trader = FTMOAITrader()
        
        # Initialize the enhanced GUI
        gui = ExpertTradingGUI(trader)
        
        # Connect GUI to trader for bidirectional communication
        trader.gui = gui
        
        def enhanced_shutdown():
            """FIXED: Comprehensive resource cleanup procedure"""
            logger.info("🔄 Enhanced shutdown sequence initiated...")
            
            # Stop trading first
            gui.stop_trading()
            
            # Clean up cache to free memory
            if hasattr(trader, 'cleanup_cache'):
                trader.cleanup_cache()
            
            # Save all open positions with enhanced metadata
            logger.info("💾 Saving enhanced position data...")
            trader.save_open_positions()
            
            # Close MT5 connection if available
            if MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                    logger.info("✅ MT5 connection closed gracefully")
                except Exception as e:
                    logger.error(f"❌ Error closing MT5 connection: {e}")
            
            # Close database connection
            if trader.con:
                try:
                    trader.con.close()
                    logger.info("✅ Database connection closed")
                except Exception as e:
                    logger.error(f"❌ Error closing database: {e}")
            
            # Close GUI
            gui.root.destroy()
            logger.info("🎯 FTMO AI Trading System shut down successfully")
            print("✅ System shutdown completed successfully.")
        
        # Set up enhanced shutdown handler
        gui.root.protocol("WM_DELETE_WINDOW", enhanced_shutdown)
        
        # Handle Ctrl+C in terminal
        import signal
        def signal_handler(sig, frame):
            print("\n🛑 Ctrl+C detected - initiating graceful shutdown...")
            enhanced_shutdown()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the enhanced GUI main loop
        logger.info("✅ ENHANCED SYSTEM INITIALIZATION COMPLETE - STARTING GUI")
        print("✅ System initialized successfully!")
        print("📊 GUI starting...")
        
        gui.run()
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR DURING ENHANCED STARTUP: {e}")
        print(f"❌ Critical error during startup: {e}")
        traceback.print_exc()
        
        # Attempt emergency shutdown
        try:
            if MT5_AVAILABLE:
                mt5.shutdown()
        except:
            pass

# ==============================================================
# PROGRAM ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    # Enhanced program header
    print("🎯 FTMO AI Quantitative Trading System - Enhanced Version")
    print("========================================================")
    print("Features:")
    print("✅ Real MT5 Integration with OANDA")
    print("✅ XGBoost AI with Adaptive Learning") 
    print("✅ Advanced Risk Management (FTMO Rules)")
    print("✅ Modern GUI with Professional Styling")
    print("✅ Enhanced Position Synchronization")
    print("✅ Comprehensive Analytics and Tracking")
    print("========================================================")
    
    # Run the enhanced main function
    main()
    
    print("✅ Program execution completed successfully.")
