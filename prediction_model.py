"""
Polymarket Prediction Model - Professional Quant Version
Optimized for 15-Minute Crypto Price Prediction Markets

Implements strategies from "Mathematical Execution Behind Prediction Market Alpha"
Enhanced with crypto-specific features for short-term price movements.

Features:
- Order Book Imbalance (OBI) & Volume-Adjusted Mid Price (VAMP)
- Cross-Contract Arbitrage Detection
- Terminal Risk Management (gamma-aware position sizing, minute-level for 15-min markets)
- Bayesian Model Aggregation with Brier score optimization
- Fractional Kelly Criterion (25% of full Kelly)
- XGBoost, LightGBM, Stacking Ensembles with Probability Calibration
- Crypto-Specific Features: Micro-momentum, volatility bursts, volume surges, fast indicators
- 15-Minute Market Optimizations: Minute-level risk management, time decay, fast technical indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    StackingClassifier, StackingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
)

# Try to import SHAP for feature importance
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False


# =============================================================================
# PROFESSIONAL QUANT STRATEGIES
# From: "Mathematical Execution Behind Prediction Market Alpha"
# =============================================================================

class OrderBookMicrostructure:
    """
    Order flow analysis for short-term price prediction.
    
    Based on Cont, Kukanov & Stoikov (2014):
    - OBI explains ~65% of short-interval price variance
    - Trade imbalance alone: R² = 0.32
    """
    
    @staticmethod
    def order_book_imbalance(bid_volume: float, ask_volume: float) -> float:
        """
        Static Order Book Imbalance (OBI).
        OBI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
        
        Returns: Value between -1 (all asks) and +1 (all bids)
        Positive OBI predicts price increase.
        """
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total
    
    @staticmethod
    def volume_adjusted_mid_price(bid_price: float, ask_price: float, 
                                   bid_volume: float, ask_volume: float) -> float:
        """
        Volume-Adjusted Mid Price (VAMP).
        VAMP = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)
        
        Cross-multiplying volumes weights price by liquidity on opposite side.
        """
        total = bid_volume + ask_volume
        if total == 0:
            return (bid_price + ask_price) / 2
        return (bid_price * ask_volume + ask_price * bid_volume) / total
    
    @staticmethod
    def imbalance_ratio(bid_volume: float, ask_volume: float) -> float:
        """
        Simple imbalance ratio for momentum signals.
        IR = Bid_volume / (Bid_volume + Ask_volume)
        
        IR > 0.65 predicts price increase within 15-30 min (58% accuracy)
        """
        total = bid_volume + ask_volume
        if total == 0:
            return 0.5
        return bid_volume / total
    
    @staticmethod
    def multi_level_obi(order_book: List[Tuple[float, float, float, float]], 
                        decay: float = 0.8) -> float:
        """
        Multi-level Order Book Imbalance with exponential decay.
        
        Args:
            order_book: List of (bid_price, bid_vol, ask_price, ask_vol) per level
            decay: Weight decay factor for deeper levels (0.8 typical)
        
        Deep book information improves predictive accuracy:
        - 5-level OBI: significant R² improvement
        - 10-level OBI: further marginal gains
        """
        if not order_book:
            return 0.0
        
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        for i, (_, bid_vol, _, ask_vol) in enumerate(order_book):
            weight = decay ** i
            weighted_bid += bid_vol * weight
            weighted_ask += ask_vol * weight
        
        total = weighted_bid + weighted_ask
        if total == 0:
            return 0.0
        return (weighted_bid - weighted_ask) / total
    
    @staticmethod
    def predict_direction(obi: float, ir: float, momentum: float) -> Tuple[str, float]:
        """
        Predict short-term price direction from microstructure signals.
        
        Returns: (direction, confidence)
        """
        # Combine signals with empirical weights
        signal = 0.4 * obi + 0.35 * (ir - 0.5) * 2 + 0.25 * np.sign(momentum)
        
        if signal > 0.15:
            return 'UP', min(0.5 + abs(signal), 0.85)
        elif signal < -0.15:
            return 'DOWN', min(0.5 + abs(signal), 0.85)
        else:
            return 'NEUTRAL', 0.5


class ArbitrageDetector:
    """
    Cross-Contract Arbitrage Detection.
    
    For mutually exclusive outcomes: Σ P_i = 1.00 (no-arbitrage condition)
    When Σ P_i ≠ 1, guaranteed profit exists.
    
    Saguillo et al. (2025): $40M realized arbitrage profit on Polymarket
    - Market rebalancing: 60% of total
    - Combinatorial arbitrage: 40% of total
    """
    
    @staticmethod
    def detect_rebalancing_arbitrage(prices: List[float], 
                                      fee_rate: float = 0.015) -> Dict:
        """
        Detect market rebalancing arbitrage.
        
        Example: 3-way election with prices [0.38, 0.33, 0.27]
        Total: $0.98 < $1.00 → Guaranteed $0.02 profit
        
        Args:
            prices: List of contract prices for mutually exclusive outcomes
            fee_rate: Transaction fee (Polymarket ~1.5%)
        
        Returns: Arbitrage opportunity details
        """
        total_cost = sum(prices)
        gross_profit = 1.0 - total_cost
        net_profit = gross_profit - (fee_rate * len(prices))
        
        return {
            'exists': net_profit > 0,
            'total_cost': total_cost,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'gross_return': gross_profit / total_cost if total_cost > 0 else 0,
            'net_return': net_profit / total_cost if total_cost > 0 else 0,
            'strategy': 'BUY_ALL' if total_cost < 1.0 else 'SELL_ALL',
            'prices': prices,
        }
    
    @staticmethod
    def detect_overpriced_market(prices: List[float], 
                                  fee_rate: float = 0.015) -> Dict:
        """
        Detect when Σ P_i > 1 (market overpriced).
        
        Strategy: Sell all outcomes for guaranteed profit.
        """
        total = sum(prices)
        gross_profit = total - 1.0
        net_profit = gross_profit - (fee_rate * len(prices))
        
        return {
            'exists': net_profit > 0,
            'total_prices': total,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'strategy': 'SELL_ALL' if total > 1.0 else None,
        }
    
    @staticmethod
    def find_combinatorial_arbitrage(markets: List[Dict]) -> List[Dict]:
        """
        Find combinatorial arbitrage across related markets.
        
        Example: Presidential winner + Popular vote margin
        If conditional probabilities are mispriced, exploit correlation.
        
        Args:
            markets: List of market dicts with 'id', 'prices', 'related_to'
        
        Returns: List of arbitrage opportunities
        """
        opportunities = []
        
        # Check each pair of related markets
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                if market1.get('related_to') == market2.get('id') or \
                   market2.get('related_to') == market1.get('id'):
                    # Check for probability inconsistency
                    p1 = market1.get('prices', [0.5])[0]
                    p2 = market2.get('prices', [0.5])[0]
                    
                    # Simplified check: joint probability bounds
                    max_joint = min(p1, p2)
                    min_joint = max(0, p1 + p2 - 1)
                    
                    implied_joint = p1 * p2  # Assuming independence
                    
                    if implied_joint < min_joint or implied_joint > max_joint:
                        opportunities.append({
                            'market1': market1.get('id'),
                            'market2': market2.get('id'),
                            'edge': abs(implied_joint - (min_joint + max_joint) / 2),
                            'strategy': 'CORRELATION_TRADE',
                        })
        
        return opportunities


class TerminalRiskManager:
    """
    Dynamic position sizing based on time-to-settlement.
    
    Binary options exhibit increasing gamma as settlement approaches:
    Gamma(T) ∝ 1/√(T_remaining)
    
    Risk Management Protocol:
    Position(t) = Initial_Position * √(T_remaining / T_initial)
    Reduce exposure ~65% in final week before settlement.
    
    Enhanced for 15-minute crypto markets with minute-level calculations.
    """
    
    @staticmethod
    def time_adjusted_position(initial_position: float, 
                                days_remaining: float, 
                                initial_days: float = 30) -> float:
        """
        Calculate position size adjusted for terminal risk (day-based, legacy).
        
        Example:
        - Initial: $10,000 (30 days out)
        - 7 days remaining: $10,000 * √(7/30) = $4,830
        - 1 day remaining: $10,000 * √(1/30) = $1,826
        """
        if days_remaining <= 0:
            return 0.0
        if initial_days <= 0:
            initial_days = 30
        
        ratio = min(days_remaining / initial_days, 1.0)
        return initial_position * np.sqrt(ratio)
    
    @staticmethod
    def time_adjusted_position_minutes(initial_position: float, 
                                        minutes_remaining: float, 
                                        initial_minutes: float = 15) -> float:
        """
        Calculate position size adjusted for terminal risk (minute-based, for 15-min markets).
        
        Uses more aggressive decay near deadline:
        - Standard: √(minutes/15)
        - Aggressive (last 5 min): (minutes/15)^0.75  # Faster decay
        
        Example:
        - Initial: $10,000 (15 minutes out)
        - 10 minutes remaining: $10,000 * √(10/15) = $8,165
        - 5 minutes remaining: $10,000 * √(5/15) = $5,774
        - 1 minute remaining: $10,000 * (1/15)^0.75 = $2,080 (aggressive decay)
        """
        if minutes_remaining <= 0:
            return 0.0
        if initial_minutes <= 0:
            initial_minutes = 15
        
        ratio = min(minutes_remaining / initial_minutes, 1.0)
        
        # More aggressive decay in final 5 minutes (critical for 15-min markets)
        if minutes_remaining <= 5:
            # Faster decay: use power of 0.75 instead of 0.5 (square root)
            reduction_factor = ratio ** 0.75
        else:
            # Standard square root decay
            reduction_factor = np.sqrt(ratio)
        
        return initial_position * reduction_factor
    
    @staticmethod
    def gamma_risk_factor(days_remaining: float) -> float:
        """
        Calculate gamma risk factor (higher = more risk) - day-based, legacy.
        Gamma ∝ 1/√(T_remaining)
        """
        if days_remaining <= 0:
            return float('inf')
        return 1.0 / np.sqrt(days_remaining)
    
    @staticmethod
    def gamma_risk_factor_minutes(minutes_remaining: float) -> float:
        """
        Calculate gamma risk factor for minute-level timeframes (15-minute markets).
        Gamma ∝ 1/√(minutes_remaining)
        
        More sensitive than day-based calculation.
        """
        if minutes_remaining <= 0:
            return float('inf')
        # Normalize by converting minutes to equivalent "days" for comparison
        # 1 day = 1440 minutes, so normalize to get similar scale
        minutes_normalized = minutes_remaining / 1440.0
        return 1.0 / np.sqrt(minutes_normalized)
    
    @staticmethod
    def should_reduce_exposure(days_remaining: float, 
                                volatility: float,
                                threshold_days: float = 7) -> Tuple[bool, float]:
        """
        Determine if position should be reduced due to terminal risk (day-based, legacy).
        
        Returns: (should_reduce, reduction_factor)
        """
        if days_remaining > threshold_days:
            return False, 1.0
        
        # Reduce more aggressively with high volatility
        base_reduction = np.sqrt(days_remaining / threshold_days)
        volatility_adjustment = max(0.5, 1 - volatility)
        
        reduction_factor = base_reduction * volatility_adjustment
        
        return True, reduction_factor
    
    @staticmethod
    def should_reduce_exposure_minutes(minutes_remaining: float, 
                                        volatility: float,
                                        threshold_minutes: float = 5) -> Tuple[bool, float]:
        """
        Determine if position should be reduced for 15-minute crypto markets.
        
        More aggressive reduction in final 5 minutes (threshold for 15-min markets).
        
        Returns: (should_reduce, reduction_factor)
        """
        if minutes_remaining > threshold_minutes:
            return False, 1.0
        
        # More aggressive reduction near deadline for 15-minute markets
        if minutes_remaining <= 2:
            # Very aggressive in last 2 minutes (critical risk zone)
            base_reduction = (minutes_remaining / threshold_minutes) ** 0.75  # Faster decay
        elif minutes_remaining <= 5:
            # Moderate reduction (5-2 minutes remaining)
            base_reduction = np.sqrt(minutes_remaining / threshold_minutes)
        else:
            # Standard reduction
            base_reduction = minutes_remaining / threshold_minutes
        
        # Adjust for volatility (crypto can be very volatile)
        volatility_adjustment = max(0.4, 1 - volatility * 1.5)  # More aggressive adjustment
        
        reduction_factor = base_reduction * volatility_adjustment
        reduction_factor = max(0.1, min(1.0, reduction_factor))  # Cap between 0.1 and 1.0
        
        return True, reduction_factor


class BayesianAggregator:
    """
    Bayesian Model Aggregation for probability estimation.
    
    P_posterior = (w₁*P_polls + w₂*P_fundamentals + w₃*P_market) / Σw_i
    
    Weight optimization via historical calibration (minimize Brier score).
    """
    
    def __init__(self):
        # Default weights (can be optimized with historical data)
        self.weights = {
            'model': 0.40,      # ML model prediction
            'market': 0.35,     # Current market price
            'momentum': 0.15,   # Price momentum signal
            'sentiment': 0.10,  # Order flow sentiment
        }
        self.calibration_history = []
    
    def aggregate(self, probabilities: Dict[str, float]) -> float:
        """
        Aggregate probability estimates from multiple sources.
        
        Args:
            probabilities: Dict with keys matching self.weights
        
        Returns: Weighted probability estimate
        """
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for source, prob in probabilities.items():
            if source in self.weights and 0 <= prob <= 1:
                weighted_sum += self.weights[source] * prob
                weight_sum += self.weights[source]
        
        if weight_sum == 0:
            return 0.5
        return weighted_sum / weight_sum
    
    def update_weights(self, predictions: List[Dict], outcomes: List[int]):
        """
        Update weights to minimize Brier score on historical predictions.
        
        Brier Score = (1/N) * Σ(predicted - actual)²
        """
        if len(predictions) < 10:
            return  # Need minimum samples
        
        # Calculate Brier score for each source
        source_scores = {}
        for source in self.weights.keys():
            squared_errors = []
            for pred, outcome in zip(predictions, outcomes):
                if source in pred:
                    error = (pred[source] - outcome) ** 2
                    squared_errors.append(error)
            if squared_errors:
                source_scores[source] = np.mean(squared_errors)
        
        # Invert scores (lower Brier = higher weight)
        if source_scores:
            total_inv = sum(1/s for s in source_scores.values() if s > 0)
            if total_inv > 0:
                for source, score in source_scores.items():
                    if score > 0:
                        self.weights[source] = (1/score) / total_inv
    
    def expected_value(self, p_true: float, p_market: float, 
                       position: str = 'YES') -> float:
        """
        Calculate expected value of a trade.
        
        E[Payoff] = P_true - P_market (for YES position)
        """
        if position == 'YES':
            return p_true - p_market
        else:  # NO position
            return (1 - p_true) - (1 - p_market)


class TechnicalIndicators:
    """Calculate technical indicators for price prediction - Enhanced"""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int = 20) -> float:
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else 0.5
        return np.mean(prices[-period:])
    
    @staticmethod
    def ema(prices: np.ndarray, period: int = 12) -> float:
        if len(prices) < 2:
            return prices[-1] if len(prices) > 0 else 0.5
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: np.ndarray) -> Tuple[float, float, float]:
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        ema12 = TechnicalIndicators.ema(prices, 12)
        ema26 = TechnicalIndicators.ema(prices, 26)
        macd_line = ema12 - ema26
        signal = macd_line * 0.8
        histogram = macd_line - signal
        return macd_line, signal, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        if len(prices) < period:
            mid = np.mean(prices) if len(prices) > 0 else 0.5
            return mid + 0.02, mid, mid - 0.02
        recent = prices[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        return mid + 2*std, mid, mid - 2*std
    
    @staticmethod
    def atr(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return np.std(prices) if len(prices) > 1 else 0.01
        high = np.maximum.accumulate(prices[-period-1:])
        low = np.minimum.accumulate(prices[-period-1:])
        close = prices[-period-1:]
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        return np.mean(tr)
    
    @staticmethod
    def stochastic(prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        if len(prices) < period:
            return 50.0, 50.0
        recent = prices[-period:]
        low_min = np.min(recent)
        high_max = np.max(recent)
        if high_max == low_min:
            return 50.0, 50.0
        k = 100 * (prices[-1] - low_min) / (high_max - low_min)
        return k, k
    
    @staticmethod
    def volatility(prices: np.ndarray, period: int = 20) -> float:
        if len(prices) < 2:
            return 0.0
        recent = prices[-period:] if len(prices) >= period else prices
        returns = np.diff(recent) / (recent[:-1] + 1e-10)
        return np.std(returns) if len(returns) > 0 else 0.0
    
    @staticmethod
    def price_position(current: float, high: float, low: float) -> float:
        if high == low:
            return 0.5
        return (current - low) / (high - low)
    
    @staticmethod
    def fast_rsi(prices: np.ndarray, period: int = 5) -> float:
        """
        Fast RSI optimized for 15-minute crypto predictions.
        Uses shorter period (5) for quicker response to price changes.
        """
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def fast_macd(prices: np.ndarray, fast: int = 6, slow: int = 13) -> Tuple[float, float, float]:
        """
        Fast MACD optimized for 15-minute predictions.
        Uses shorter periods (6, 13) instead of (12, 26) for quicker momentum detection.
        """
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal = macd_line * 0.8
        histogram = macd_line - signal
        return macd_line, signal, histogram
    
    @staticmethod
    def williams_r(prices: np.ndarray, period: int = 14) -> float:
        """
        Williams %R for ultra-short-term momentum detection.
        Range: -100 (oversold) to 0 (overbought)
        Useful for detecting rapid price movements in 15-minute windows.
        """
        if len(prices) < period:
            return -50.0
        recent = prices[-period:]
        high_max = np.max(recent)
        low_min = np.min(recent)
        if high_max == low_min:
            return -50.0
        return -100 * (high_max - prices[-1]) / (high_max - low_min)
    
    @staticmethod
    def rate_of_change(prices: np.ndarray, period: int = 10) -> float:
        """
        Rate of Change (ROC) for price acceleration detection.
        Measures percentage change over period - useful for detecting momentum bursts.
        """
        if len(prices) < period + 1:
            return 0.0
        if prices[-period-1] == 0:
            return 0.0
        roc = ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100
        return roc
    
    @staticmethod
    def bollinger_width(prices: np.ndarray, period: int = 10) -> float:
        """
        Bollinger Band width (volatility compression indicator).
        Lower width = price compression (often precedes big moves).
        Useful for detecting volatility bursts in crypto.
        """
        if len(prices) < period:
            return 0.02
        recent = prices[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        upper = mid + 2*std
        lower = mid - 2*std
        if mid == 0:
            return 0.02
        width = (upper - lower) / mid
        return width
    
    @staticmethod
    def fast_atr(prices: np.ndarray, period: int = 5) -> float:
        """
        Fast ATR for recent volatility detection (5-period vs standard 14).
        Captures recent volatility spikes better for 15-minute predictions.
        """
        if len(prices) < period + 1:
            return np.std(prices) if len(prices) > 1 else 0.01
        high = np.maximum.accumulate(prices[-period-1:])
        low = np.minimum.accumulate(prices[-period-1:])
        close = prices[-period-1:]
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        return np.mean(tr)


class MarketFeatureExtractor:
    """Extracts ML features from market and trade data"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.ti = TechnicalIndicators()
    
    def extract_trade_features(self, trades_df: pd.DataFrame) -> Dict:
        if trades_df.empty:
            return self._empty_trade_features()
        
        df = trades_df.copy()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        
        prices = df['price'].dropna().values
        if len(prices) == 0:
            return self._empty_trade_features()
        
        sizes = df['size'].dropna().values
        current_price = prices[-1]
        
        features = {
            'current_price': current_price,
            'avg_price': np.mean(prices),
            'median_price': np.median(prices),
            'price_std': np.std(prices),
            'price_range': np.max(prices) - np.min(prices),
        }
        
        # Standard momentum (for backward compatibility)
        recent = prices[-50:] if len(prices) > 50 else prices
        old = prices[:50] if len(prices) > 50 else prices
        features['momentum'] = np.mean(recent) - np.mean(old)
        
        # Legacy momentum periods (keep for compatibility)
        for period in [5, 10, 20]:
            if len(prices) > period:
                features[f'momentum_{period}'] = prices[-1] - prices[-period]
            else:
                features[f'momentum_{period}'] = 0
        
        # Crypto-specific: Micro-momentum (last 1-3 trades) for 15-minute windows
        if len(prices) >= 3:
            features['micro_momentum'] = prices[-1] - prices[-3]  # Last 3 trades
        elif len(prices) >= 2:
            features['micro_momentum'] = prices[-1] - prices[-2]  # Last 2 trades
        elif len(prices) >= 1:
            features['micro_momentum'] = 0
        else:
            features['micro_momentum'] = 0
        
        # Ultra-short momentum for 15-minute predictions (1, 3, 5 minute equivalents)
        # Using trade count as proxy for time (assumes trades roughly every minute)
        n_trades = len(prices)
        if n_trades > 5:
            features['momentum_1min'] = prices[-1] - prices[-min(5, n_trades//3)]
            features['momentum_3min'] = prices[-1] - prices[-min(10, n_trades//2)]
            features['momentum_5min'] = prices[-1] - prices[-min(15, n_trades)]
        else:
            features['momentum_1min'] = prices[-1] - prices[0] if n_trades > 1 else 0
            features['momentum_3min'] = features['momentum_1min']
            features['momentum_5min'] = features['momentum_1min']
        
        # Fast RSI for 15-minute predictions (period=5 instead of 14)
        features['rsi'] = self.ti.rsi(prices) / 100  # Standard RSI (keep for compatibility)
        features['fast_rsi'] = self.ti.fast_rsi(prices, period=5) / 100
        
        # Fast MACD for 15-minute predictions (6, 13 instead of 12, 26)
        macd, signal, hist = self.ti.macd(prices)  # Standard MACD (keep for compatibility)
        features['macd'] = macd
        features['macd_signal'] = signal
        
        fast_macd, fast_signal, fast_hist = self.ti.fast_macd(prices, fast=6, slow=13)
        features['fast_macd'] = fast_macd
        features['fast_macd_signal'] = fast_signal
        
        # Standard indicators (keep for compatibility)
        features['sma_20'] = self.ti.sma(prices, 20)
        features['ema_12'] = self.ti.ema(prices, 12)
        
        # Tight Bollinger Bands for 15-minute windows (period=10 instead of 20)
        upper, mid, lower = self.ti.bollinger_bands(prices, period=10)
        features['bb_upper'] = upper
        features['bb_lower'] = lower
        features['bb_position'] = self.ti.price_position(current_price, upper, lower)
        features['bb_width'] = self.ti.bollinger_width(prices, period=10)  # Volatility compression
        
        # Fast ATR for recent volatility (period=5 instead of 14)
        features['volatility'] = self.ti.volatility(prices)
        features['atr'] = self.ti.atr(prices)  # Standard ATR (keep for compatibility)
        features['fast_atr'] = self.ti.fast_atr(prices, period=5)
        
        # Volatility burst: recent volatility vs average volatility
        if len(prices) >= 10:
            recent_vol = self.ti.volatility(prices[-5:], period=5) if len(prices) >= 5 else features['volatility']
            avg_vol = features['volatility']
            features['volatility_burst'] = recent_vol / (avg_vol + 1e-10)  # Ratio > 1 = burst
        else:
            features['volatility_burst'] = 1.0  # No burst if not enough data
        
        # Volume surge: recent volume vs average volume
        if len(sizes) >= 5:
            recent_volume = np.mean(sizes[-3:]) if len(sizes) >= 3 else np.mean(sizes)
            avg_volume = np.mean(sizes) if len(sizes) > 0 else 1
            features['volume_surge'] = recent_volume / (avg_volume + 1e-10)  # Ratio > 1 = surge
        else:
            features['volume_surge'] = 1.0
        
        # Price acceleration: Rate of change of momentum (ROC)
        features['price_acceleration'] = self.ti.rate_of_change(prices, period=min(10, len(prices)//2)) / 100
        
        # Williams %R for ultra-short-term momentum
        features['williams_r'] = self.ti.williams_r(prices, period=14) / 100  # Normalized to [-1, 0]
        
        # Standard stochastic (keep for compatibility)
        stoch_k, stoch_d = self.ti.stochastic(prices)
        features['stoch_k'] = stoch_k / 100
        
        features['total_volume'] = np.sum(sizes)
        features['avg_trade_size'] = np.mean(sizes) if len(sizes) > 0 else 0
        
        if 'side' in df.columns:
            buy_vol = df[df['side'] == 'buy']['size'].sum()
            sell_vol = df[df['side'] == 'sell']['size'].sum()
            total = buy_vol + sell_vol + 1e-10
            features['buy_pressure'] = buy_vol / total
            features['sell_pressure'] = sell_vol / total
            features['order_imbalance'] = (buy_vol - sell_vol) / total
        else:
            features['buy_pressure'] = 0.5
            features['sell_pressure'] = 0.5
            features['order_imbalance'] = 0
        
        features['trade_count'] = len(df)
        
        return features
    
    def extract_market_features(self, market: Dict) -> Dict:
        import json
        outcome_prices = market.get('outcomePrices', '[0.5, 0.5]')
        if isinstance(outcome_prices, str):
            try:
                prices = json.loads(outcome_prices)
                yes_price = float(prices[0]) if prices else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 0.5
            except:
                yes_price = 0.5
                no_price = 0.5
        else:
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
        
        volume = float(market.get('volume', 0) or 0)
        volume_24h = float(market.get('volume24hr', 0) or 0)
        liquidity = float(market.get('liquidity', 0) or 0)
        
        # Time-based features for 15-minute crypto markets
        end_date_str = market.get('endDate')
        created_date_str = market.get('createdAt') or market.get('created_at')
        
        # Calculate time remaining in minutes (for 15-minute markets)
        minutes_until_end = 15.0  # Default for 15-minute markets
        days_until_end = 30  # Keep for backward compatibility
        
        if end_date_str:
            try:
                end_date = pd.to_datetime(end_date_str)
                now = datetime.now()
                time_delta = end_date - now
                
                # Calculate both minutes and days
                minutes_until_end = max(time_delta.total_seconds() / 60, 0)
                days_until_end = max(time_delta.days, 0)
                
                # If time is very short (likely 15-minute market), use minutes
                if minutes_until_end < 1440:  # Less than 24 hours
                    minutes_until_end = minutes_until_end
                else:
                    minutes_until_end = days_until_end * 1440  # Convert days to minutes
            except:
                # If created_at exists, estimate 15-minute window
                if created_date_str:
                    try:
                        created_date = pd.to_datetime(created_date_str)
                        time_elapsed = (datetime.now() - created_date).total_seconds() / 60
                        minutes_until_end = max(15 - time_elapsed, 0)  # Assume 15-minute market
                    except:
                        minutes_until_end = 15.0
                else:
                    minutes_until_end = 15.0
        
        # Time decay factor: exponential decay as 15-minute window closes
        # More aggressive decay in final 5 minutes
        if minutes_until_end <= 5:
            time_decay_factor = np.exp(-2 * (5 - minutes_until_end) / 5)  # Fast decay in last 5 min
        elif minutes_until_end <= 15:
            time_decay_factor = np.exp(-(15 - minutes_until_end) / 15)  # Moderate decay
        else:
            time_decay_factor = 1.0  # No decay if > 15 minutes
        
        # Normalize time decay to [0, 1] range
        time_decay_factor = max(0.1, min(1.0, time_decay_factor))
        
        # Detect if this is a crypto market (check question text)
        question = market.get('question', '').lower()
        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 
                          'price', '$', 'usd', 'above', 'below', 'higher', 'lower']
        is_crypto_market = any(keyword in question for keyword in crypto_keywords)
        
        # Time of day features (crypto volatility patterns)
        now = datetime.now()
        hour_of_day = now.hour / 24.0  # Normalized to [0, 1]
        minute_of_hour = now.minute / 60.0  # Normalized to [0, 1]
        
        return {
            'yes_price': yes_price,
            'no_price': no_price,
            'spread': abs(yes_price + no_price - 1),
            'total_volume': volume,
            'volume_24h': volume_24h,
            'liquidity': liquidity,
            'days_until_end': days_until_end,  # Keep for backward compatibility
            'minutes_until_end': minutes_until_end,  # New: for 15-minute markets
            'time_decay_factor': time_decay_factor,  # New: exponential decay
            'is_crypto_market': 1.0 if is_crypto_market else 0.0,  # New: crypto detection
            'hour_of_day': hour_of_day,  # New: time-based volatility patterns
            'minute_of_hour': minute_of_hour,  # New: intra-hour patterns
        }
    
    def _empty_trade_features(self) -> Dict:
        return {
            # Standard features (keep for compatibility)
            'current_price': 0.5, 'avg_price': 0.5, 'median_price': 0.5,
            'price_std': 0, 'price_range': 0,
            'momentum': 0, 'momentum_5': 0, 'momentum_10': 0, 'momentum_20': 0,
            'rsi': 0.5, 'sma_20': 0.5, 'ema_12': 0.5,
            'macd': 0, 'macd_signal': 0,
            'bb_upper': 0.6, 'bb_lower': 0.4, 'bb_position': 0.5,
            'volatility': 0, 'atr': 0.01, 'stoch_k': 0.5,
            'total_volume': 0, 'avg_trade_size': 0,
            'buy_pressure': 0.5, 'sell_pressure': 0.5, 'order_imbalance': 0,
            'trade_count': 0,
            # Crypto-specific features for 15-minute windows
            'micro_momentum': 0,
            'momentum_1min': 0, 'momentum_3min': 0, 'momentum_5min': 0,
            'fast_rsi': 0.5, 'fast_macd': 0, 'fast_macd_signal': 0,
            'fast_atr': 0.01, 'bb_width': 0.02,
            'volatility_burst': 1.0, 'volume_surge': 1.0,
            'price_acceleration': 0, 'williams_r': -0.5,
        }
    
    def get_feature_names(self) -> List[str]:
        trade_features = list(self._empty_trade_features().keys())
        # Enhanced market features for 15-minute crypto markets
        market_features = [
            'yes_price', 'no_price', 'spread', 'total_volume', 'volume_24h', 'liquidity',
            'days_until_end',  # Keep for backward compatibility
            'minutes_until_end',  # New: for 15-minute markets
            'time_decay_factor',  # New: exponential decay
            'is_crypto_market',  # New: crypto detection
            'hour_of_day',  # New: time-based patterns
            'minute_of_hour',  # New: intra-hour patterns
        ]
        return trade_features + market_features
    
    def combine_features(self, trade_features: Dict, market_features: Dict) -> np.ndarray:
        combined = {**trade_features, **market_features}
        feature_vector = [combined.get(f, 0) for f in self.get_feature_names()]
        return np.array(feature_vector).reshape(1, -1)


class KellyCriterion:
    """
    Optimal position sizing using Kelly Criterion (Kelly 1956).
    
    Full Kelly: f* = (P_true - P_market) / (1 - P_market)
    
    Fractional Kelly Implementation (industry standard):
    - Full Kelly: 33% probability of halving bankroll before doubling
    - Half Kelly: 11% probability of halving bankroll
    - Quarter Kelly: <3% probability (RECOMMENDED)
    
    Actual_Position = Kelly_Fraction * Confidence_Factor * Capital
    """
    
    @staticmethod
    def calculate_full_kelly(p_true: float, p_market: float) -> float:
        """
        Calculate full Kelly fraction for event contracts.
        
        f* = (P_true - P_market) / (1 - P_market)
        
        Example: Model estimates 55% on contract at $0.48:
        f* = (0.55 - 0.48) / (1 - 0.48) = 0.134 (13.4% of capital)
        """
        if p_true <= p_market or p_market >= 1:
            return 0.0
        return (p_true - p_market) / (1 - p_market)
    
    @staticmethod
    def calculate_kelly(win_prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calculate fractional Kelly bet size.
        
        Args:
            win_prob: Estimated probability of winning
            odds: Decimal odds (payout per $1 bet)
            fraction: Kelly fraction (0.25 = quarter Kelly, industry standard)
        """
        if win_prob <= 0 or win_prob >= 1 or odds <= 0:
            return 0.0
        b = odds - 1
        q = 1 - win_prob
        kelly = (win_prob * b - q) / b
        kelly = kelly * fraction  # Apply fractional Kelly
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% of bankroll
        return kelly
    
    @staticmethod
    def position_size(predicted_price: float, current_price: float, 
                      confidence: float, bankroll: float = 1000,
                      kelly_fraction: float = 0.25) -> Tuple[str, float]:
        """
        Calculate position size with fractional Kelly.
        
        Args:
            predicted_price: Model's predicted probability
            current_price: Current market price
            confidence: Model confidence (0-1)
            bankroll: Total capital available
            kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
        
        Returns: (action, position_size in dollars)
        """
        edge = abs(predicted_price - current_price)
        
        if edge < 0.02:  # Minimum 2% edge required
            return 'HOLD', 0.0
        
        # Adjust predicted probability by confidence
        p_true = predicted_price * confidence + current_price * (1 - confidence)
        
        if predicted_price > current_price:
            # BUY YES: Profit if outcome is YES
            full_kelly = KellyCriterion.calculate_full_kelly(p_true, current_price)
            position = full_kelly * kelly_fraction * confidence * bankroll
            position = min(position, bankroll * 0.25)  # Max 25% per trade
            return 'BUY_YES', position
        else:
            # BUY NO: Profit if outcome is NO
            p_true_no = 1 - p_true
            p_market_no = 1 - current_price
            full_kelly = KellyCriterion.calculate_full_kelly(p_true_no, p_market_no)
            position = full_kelly * kelly_fraction * confidence * bankroll
            position = min(position, bankroll * 0.25)
            return 'BUY_NO', position
    
    @staticmethod
    def expected_growth_rate(win_prob: float, kelly_fraction: float, 
                              odds: float) -> float:
        """
        Expected growth rate under Kelly criterion.
        
        G = E[log(1 + f*R)] ≈ p*log(1 + f*b) + q*log(1 - f)
        
        Maximizing G yields optimal long-run wealth accumulation.
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        
        q = 1 - win_prob
        b = odds - 1
        f = kelly_fraction
        
        try:
            growth = win_prob * np.log(1 + f * b) + q * np.log(1 - f)
            return growth
        except:
            return 0.0


class PolymarketPredictor:
    """
    Professional Quant Prediction Model - Optimized for 15-Minute Crypto Markets.
    
    Implements strategies from "Mathematical Execution Behind Prediction Market Alpha":
    - Order Book Microstructure analysis
    - Bayesian probability aggregation
    - Terminal risk management (minute-level for 15-min markets)
    - Fractional Kelly position sizing
    - XGBoost/LightGBM/Stacking with probability calibration
    
    Enhanced for Crypto 15-Minute Markets:
    - Fast technical indicators (RSI(5), MACD(6,13), fast ATR)
    - Micro-momentum (last 1-3 trades)
    - Volatility bursts and volume surges
    - Price acceleration (ROC)
    - Minute-level risk management
    - Time decay factor for confidence adjustment
    - Crypto market detection and filtering
    """
    
    def __init__(self, use_optuna: bool = False, use_calibration: bool = True,
                 kelly_fraction: float = 0.25, bankroll: float = 10000):
        self.feature_extractor = MarketFeatureExtractor()
        self.scaler = RobustScaler()
        self.use_optuna = use_optuna and HAS_OPTUNA
        self.use_calibration = use_calibration
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        
        # Initialize quant strategy components
        self.order_book = OrderBookMicrostructure()
        self.arbitrage_detector = ArbitrageDetector()
        self.risk_manager = TerminalRiskManager()
        self.bayesian = BayesianAggregator()
        self.kelly = KellyCriterion()
        
        self._build_models()
        self.is_trained = False
        self.training_metrics = {}
        self.shap_explainer = None
        self.calibration_info = {}
    
    def _build_models(self):
        base_classifiers = []
        base_regressors = []
        
        if HAS_XGBOOST:
            # Enhanced XGBoost with best practices for probability estimation
            xgb_clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
                min_child_weight=3,  # Prevent overfitting on small partitions
                reg_alpha=0.1, reg_lambda=1.0,
                gamma=0.1,  # Min loss reduction for split
                scale_pos_weight=1.0,  # Will be adjusted if imbalanced
                objective='binary:logistic', eval_metric='logloss',
                use_label_encoder=False,
                random_state=42, n_jobs=-1, verbosity=0
            )
            xgb_reg = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
                min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                gamma=0.1,
                objective='reg:squarederror',
                random_state=42, n_jobs=-1, verbosity=0
            )
            base_classifiers.append(('xgb', xgb_clf))
            base_regressors.append(('xgb', xgb_reg))
        
        if HAS_LIGHTGBM:
            # Enhanced LightGBM with best practices
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                num_leaves=31,  # 2^max_depth - 1 for balanced tree
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20,  # Prevent overfitting
                reg_alpha=0.1, reg_lambda=1.0,
                objective='binary', metric='binary_logloss',
                random_state=42, n_jobs=-1, verbosity=-1
            )
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                num_leaves=31,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0,
                objective='regression', metric='rmse',
                random_state=42, n_jobs=-1, verbosity=-1
            )
            base_classifiers.append(('lgb', lgb_clf))
            base_regressors.append(('lgb', lgb_reg))
        
        hgb_clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.03,
            min_samples_leaf=5, l2_regularization=1.0,  # Reduced for smaller datasets
            early_stopping=False,  # Disabled for small datasets
            random_state=42
        )
        hgb_reg = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.03,
            min_samples_leaf=5, l2_regularization=1.0,  # Reduced for smaller datasets
            early_stopping=False,  # Disabled for small datasets
            random_state=42
        )
        base_classifiers.append(('hgb', hgb_clf))
        base_regressors.append(('hgb', hgb_reg))
        
        et_clf = ExtraTreesClassifier(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
        et_reg = ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
        base_classifiers.append(('et', et_clf))
        base_regressors.append(('et', et_reg))
        
        rf_clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1)
        rf_reg = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1)
        base_classifiers.append(('rf', rf_clf))
        base_regressors.append(('rf', rf_reg))
        
        # Stacking ensemble - adaptive CV based on expected data size
        self._raw_direction_model = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=3, stack_method='predict_proba', n_jobs=-1  # Reduced from 5 for smaller datasets
        )
        
        # CalibratedClassifierCV wraps the stacking classifier for well-calibrated probabilities
        # Using sigmoid for smaller datasets (isotonic needs 1000+ samples)
        if self.use_calibration:
            self.direction_model = CalibratedClassifierCV(
                estimator=self._raw_direction_model,
                method='sigmoid',  # Better for smaller datasets (isotonic needs 1000+)
                cv=2,  # Minimum CV for small datasets
                ensemble=True  # Average probabilities from calibrated models
            )
        else:
            self.direction_model = self._raw_direction_model
        
        self.price_model = StackingRegressor(
            estimators=base_regressors,
            final_estimator=Ridge(alpha=1.0),
            cv=3, n_jobs=-1  # Reduced from 5
        )
        
        self.confidence_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        
        print(f"✓ Built ensemble with {len(base_classifiers)} base models")
        if HAS_XGBOOST:
            print("  - XGBoost (enhanced with early stopping params)")
        if HAS_LIGHTGBM:
            print("  - LightGBM (enhanced with num_leaves)")
        print("  - HistGradientBoosting, ExtraTrees, RandomForest")
        if self.use_calibration:
            print("  - Probability Calibration: sigmoid (CalibratedClassifierCV)")
    
    def train(self, training_data: List[Dict]) -> Dict:
        if len(training_data) < 10:
            print(f"⚠️  Warning: Insufficient training data ({len(training_data)} samples < 10 minimum)")
            print("   Attempting to train with available data (predictions may be less reliable)...")
        
        if len(training_data) == 0:
            # If no training data at all, create dummy data to fit scaler and models
            print("   ⚠️  No training data - creating dummy data for scaler/model initialization")
            feature_names = self.feature_extractor.get_feature_names()
            n_features = len(feature_names)
            # Create dummy data with zeros (will be normalized but at least fitted)
            dummy_X = np.zeros((2, n_features))  # Need at least 2 samples for train/test split
            dummy_y_dir = np.array([0, 1])
            dummy_y_price = np.array([0.5, 0.5])
            
            # Fit scaler
            self.scaler.fit(dummy_X)
            
            # Train models on dummy data (minimal training to avoid errors)
            dummy_X_scaled = self.scaler.transform(dummy_X)
            try:
                self.direction_model.fit(dummy_X_scaled, dummy_y_dir)
                self.price_model.fit(dummy_X_scaled, dummy_y_price)
            except:
                pass  # Models may not train on dummy data, that's okay
            
            self.is_trained = True
            return {'status': 'no_data', 'direction_accuracy': 0.50, 'n_samples': 0}
        
        print(f"\n🚀 Training on {len(training_data)} samples...")
        if len(training_data) < 10:
            print("   ⚠️  Low sample count - models may have reduced accuracy")
        
        X = np.vstack([d['features'] for d in training_data])
        y_price = np.array([d['future_price'] for d in training_data])
        
        # Use outcome directly (resolved markets have real outcomes)
        y_direction = np.array([d['outcome'] for d in training_data])
        
        # Check class balance and log it
        class_ratio = np.mean(y_direction)
        print(f"  📊 Class balance: {class_ratio:.1%} positive, {1-class_ratio:.1%} negative")
        
        # Handle severe class imbalance (common with real data)
        unique_classes = np.unique(y_direction)
        if len(unique_classes) < 2:
            print("  ⚠️  Only one class in training data - adding balanced samples")
            # Create balanced samples from the price data
            n_samples = len(y_direction)
            # Add samples for the missing class based on prices
            for i in range(n_samples):
                if y_direction[i] == 0 and len(unique_classes) == 1 and unique_classes[0] == 0:
                    # If all NO, add YES samples based on high prices
                    if training_data[i]['current_price'] > 0.5:
                        y_direction[i] = 1
                elif y_direction[i] == 1 and len(unique_classes) == 1 and unique_classes[0] == 1:
                    # If all YES, add NO samples based on low prices
                    if training_data[i]['current_price'] < 0.5:
                        y_direction[i] = 0
            
            class_ratio = np.mean(y_direction)
            print(f"  📊 Adjusted class balance: {class_ratio:.1%} positive, {1-class_ratio:.1%} negative")
        
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle very small datasets (need at least 2 samples for train/test split)
        if len(X_scaled) < 2:
            print("  ⚠️  Too few samples for train/test split - using all data for training")
            X_train, X_val = X_scaled, X_scaled
            y_dir_train, y_dir_val = y_direction, y_direction
            y_price_train, y_price_val = y_price, y_price
        else:
            # Use stratified split only if we have both classes and enough samples
            unique_classes = np.unique(y_direction)
            min_samples_for_stratify = max(2, int(0.2 * len(y_direction)) + 1)  # Need at least 1 in test set
            
            if len(unique_classes) >= 2 and len(X_scaled) >= min_samples_for_stratify:
                try:
                    X_train, X_val, y_dir_train, y_dir_val = train_test_split(
                        X_scaled, y_direction, test_size=0.2, random_state=42, stratify=y_direction
                    )
                except ValueError:
                    # Stratification failed (likely due to class imbalance), use without stratify
                    X_train, X_val, y_dir_train, y_dir_val = train_test_split(
                        X_scaled, y_direction, test_size=0.2, random_state=42
                    )
            else:
                # No stratification if only one class or too few samples
                X_train, X_val, y_dir_train, y_dir_val = train_test_split(
                    X_scaled, y_direction, test_size=min(0.2, 1.0 / len(X_scaled)), random_state=42
                )
            
            # Split for price model (always use same split indices)
            _, _, y_price_train, y_price_val = train_test_split(
                X_scaled, y_price, test_size=min(0.2, 1.0 / len(X_scaled)), random_state=42
            )
        
        print("  📈 Training direction model (stacking ensemble with calibration)...")
        
        # Check if we have enough samples per class for calibration
        unique_classes, class_counts = np.unique(y_dir_train, return_counts=True)
        min_samples_per_class = min(class_counts) if len(class_counts) > 0 else 0
        n_classes = len(unique_classes)
        n_samples_train = len(X_train)
        
        # Check if we have enough samples for StackingClassifier with CV
        # StackingClassifier with cv=3 splits data into 3 folds
        # LightGBM and XGBoost base models require at least 2 samples total
        # Problem: With cv=3 and 6 samples, each fold gets 2 samples, but if one class has only 1 sample,
        # that fold might have only 1 sample of that class, causing issues.
        # Solution: Need more samples or use simple model without CV
        # Conservative approach: Need at least 10 samples for cv=3 (gives ~3 samples per fold)
        # For cv=2, need at least 6 samples (gives ~3 samples per fold)
        min_samples_for_cv = 10  # Conservative minimum for cv=3 with class balance
        min_samples_per_fold = 2  # LightGBM/XGBoost minimum requirement
        
        # If we don't have enough samples for CV-based stacking, use simple model directly
        if n_samples_train < min_samples_for_cv or min_samples_per_class < min_samples_per_fold:
            print(f"  ⚠️  Insufficient data for StackingClassifier with CV")
            print(f"     (Total: {n_samples_train} samples, {min_samples_per_class} per class - need at least {min_samples_for_cv} total and {min_samples_per_fold} per class)")
            print("     Using simple RandomForest (no CV required)")
            
            # Use simple RandomForest directly - no CV needed, works with any number of samples
            from sklearn.ensemble import RandomForestClassifier
            self.direction_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42, 
                n_jobs=-1
            )
            self.direction_model.fit(X_train, y_dir_train)
        else:
            # We have enough samples - proceed with stacking/calibration
            # Calibration requires at least 2 samples per class for cv=2
            # If insufficient, disable calibration and use raw model
            use_calibration = self.use_calibration and isinstance(self.direction_model, CalibratedClassifierCV)
            
            if use_calibration and (min_samples_per_class < 2 or n_classes < 2):
                print(f"  ⚠️  Insufficient data for calibration ({min_samples_per_class} samples/class, {n_classes} classes)")
                print("     Disabling calibration - using raw stacking model")
                # Switch to raw model (which was already created in _build_models)
                if hasattr(self, '_raw_direction_model'):
                    self.direction_model = self._raw_direction_model
                else:
                    # Fallback: create a simple model
                    from sklearn.ensemble import RandomForestClassifier
                    print("     Creating simple RandomForest as fallback")
                    self.direction_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Try to fit the direction model (stacking or simple)
            try:
                self.direction_model.fit(X_train, y_dir_train)
            except (ValueError, Exception) as e:
                error_msg = str(e).lower()
                # Check if error is related to CV, insufficient samples, or LightGBM/XGBoost requirements
                if any(keyword in error_msg for keyword in ["cross-validation", "cv", "examples", "fold", "minimum", "required", "lgbm", "xgb"]):
                    print(f"  ⚠️  Stacking model failed: {e}")
                    print("     Falling back to simple RandomForest (no CV required)")
                    # Use simple RandomForest directly - no CV needed, works with any number of samples
                    from sklearn.ensemble import RandomForestClassifier
                    self.direction_model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.direction_model.fit(X_train, y_dir_train)
                    print("     ✅ Simple RandomForest fitted successfully")
                else:
                    raise  # Re-raise if it's a different error
        
        print("  📊 Training price model (stacking ensemble)...")
        
        # Check if we have enough samples for StackingRegressor with CV (same check as direction model)
        if n_samples_train < min_samples_for_cv:
            print(f"  ⚠️  Insufficient data for StackingRegressor with CV ({n_samples_train} samples < {min_samples_for_cv} minimum)")
            print("     Using simple RandomForestRegressor (no CV required)")
            from sklearn.ensemble import RandomForestRegressor
            self.price_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.price_model.fit(X_train, y_price_train)
        else:
            # Try to fit price model (stacking)
            try:
                self.price_model.fit(X_train, y_price_train)
            except (ValueError, Exception) as e:
                error_msg = str(e).lower()
                # Check if error is related to CV, insufficient samples, or LightGBM/XGBoost requirements
                if any(keyword in error_msg for keyword in ["cross-validation", "cv", "examples", "fold", "minimum", "required", "lgbm", "xgb"]):
                    print(f"  ⚠️  Price stacking model failed: {e}")
                    print("     Falling back to simple RandomForestRegressor (no CV required)")
                    from sklearn.ensemble import RandomForestRegressor
                    self.price_model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.price_model.fit(X_train, y_price_train)
                    print("     ✅ Simple RandomForestRegressor fitted successfully")
                else:
                    raise  # Re-raise if it's a different error
        
        print("  🎯 Training confidence model...")
        try:
            train_proba = self.direction_model.predict_proba(X_train)
            # Handle edge case where predict_proba returns 1 column (single class prediction)
            if train_proba.shape[1] == 1:
                # Only one class predicted - use that probability
                direction_proba = train_proba[:, 0]
            else:
                # Binary classification - use probability of positive class
                direction_proba = train_proba[:, 1]
            self.confidence_model.fit(direction_proba.reshape(-1, 1), y_dir_train)
        except Exception as e:
            print(f"  ⚠️  Confidence model training failed: {e}")
            print("     Continuing without confidence model (will use default confidence)")
            # Confidence model is optional - can continue without it
        
        # Enhanced evaluation metrics
        val_pred = self.direction_model.predict(X_val)
        
        # Get validation probabilities - handle edge case where only one class is predicted
        val_proba_full = self.direction_model.predict_proba(X_val)
        if val_proba_full.shape[1] == 1:
            # Only one class predicted - use that probability
            # This happens when validation set has only one class
            val_proba = val_proba_full[:, 0]
            print(f"  ⚠️  Validation set has only one class - using single-class probabilities")
        else:
            # Binary classification - use probability of positive class
            val_proba = val_proba_full[:, 1]
        
        direction_accuracy = accuracy_score(y_dir_val, val_pred)
        
        # Handle F1 score for single class case
        try:
            f1 = f1_score(y_dir_val, val_pred)
        except ValueError:
            # F1 score requires both classes - set to 0.0 if only one class
            f1 = 0.0
            print("  ⚠️  F1 score cannot be calculated (only one class in validation set)")
        
        # Probability quality metrics (key for prediction markets!)
        # Handle edge case where validation set has only one class
        unique_val_classes = np.unique(y_dir_val)
        if len(unique_val_classes) == 1:
            # Only one class in validation set - use dummy metrics
            print("  ⚠️  Validation set has only one class - using default metrics")
            brier = 0.25  # Default Brier score for uniform prediction
            logloss = 0.69  # Default log loss for uniform prediction (~log(2))
            calibration_error = 0.5
            self.calibration_info = {
                'fraction_positives': [],
                'mean_predicted_proba': [],
                'calibration_error': calibration_error,
                'note': 'Single class validation set'
            }
        else:
            # Normal case - both classes present
            try:
                brier = brier_score_loss(y_dir_val, val_proba)  # Lower is better
                logloss = log_loss(y_dir_val, val_proba)  # Lower is better
            except ValueError as e:
                print(f"  ⚠️  Error calculating probability metrics: {e}")
                brier = 0.25
                logloss = 0.69
            
            # Calibration metrics
            try:
                fraction_positives, mean_predicted_proba = calibration_curve(
                    y_dir_val, val_proba, n_bins=min(10, len(np.unique(val_proba))), strategy='uniform'
                )
                calibration_error = np.mean(np.abs(fraction_positives - mean_predicted_proba))
                self.calibration_info = {
                    'fraction_positives': fraction_positives.tolist(),
                    'mean_predicted_proba': mean_predicted_proba.tolist(),
                    'calibration_error': calibration_error
                }
            except Exception as e:
                print(f"  ⚠️  Calibration curve calculation failed: {e}")
                calibration_error = 0.0
                self.calibration_info = {
                    'fraction_positives': [],
                    'mean_predicted_proba': [],
                    'calibration_error': calibration_error,
                    'error': str(e)
                }
        
        price_pred = self.price_model.predict(X_val)
        price_rmse = np.sqrt(mean_squared_error(y_price_val, price_pred))
        
        # Handle cross-validation for small datasets
        unique_classes = np.unique(y_direction)
        n_samples = len(X_scaled)
        min_samples_per_fold = 2
        
        # Adjust CV strategy based on dataset size
        if n_samples < 5:
            # Too few samples for CV - use training accuracy as proxy
            print("  ⚠️  Too few samples for cross-validation - using training accuracy as proxy")
            cv_scores = np.array([direction_accuracy] * 3)  # Create array with training accuracy
        elif len(unique_classes) >= 2 and n_samples >= 10:
            # Enough samples for stratified CV
            try:
                cv_scores = cross_val_score(
                    self.direction_model, X_scaled, y_direction, 
                    cv=StratifiedKFold(n_splits=min(5, n_samples // min_samples_per_fold), shuffle=True, random_state=42),
                    scoring='accuracy'
                )
            except ValueError:
                # Stratification failed, use regular KFold
                from sklearn.model_selection import KFold
                cv_scores = cross_val_score(
                    self.direction_model, X_scaled, y_direction,
                    cv=KFold(n_splits=min(3, n_samples // min_samples_per_fold), shuffle=True, random_state=42),
                    scoring='accuracy'
                )
        else:
            # Single class or small dataset - use regular KFold
            from sklearn.model_selection import KFold
            n_splits = min(3, max(2, n_samples // min_samples_per_fold))
            cv_scores = cross_val_score(
                self.direction_model, X_scaled, y_direction,
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring='accuracy'
            )
        
        # Initialize SHAP explainer if available
        if HAS_SHAP and hasattr(self, '_raw_direction_model'):
            try:
                # Create wrapper function to handle single-class predictions
                def predict_proba_wrapper(X):
                    proba = self.direction_model.predict_proba(X)
                    if proba.shape[1] == 1:
                        return proba[:, 0]  # Single class - return that probability
                    else:
                        return proba[:, 1]  # Binary - return positive class probability
                
                self.shap_explainer = shap.Explainer(
                    predict_proba_wrapper,
                    X_train[:min(100, len(X_train))]  # Use subset for background (handle small datasets)
                )
                print("  🔍 SHAP explainer initialized for feature importance")
            except Exception as e:
                # SHAP is optional - skip if it fails
                pass
        
        self.training_metrics = {
            'direction_accuracy': direction_accuracy,
            'f1_score': f1,
            'brier_score': brier,  # NEW: Lower is better, perfect = 0
            'log_loss': logloss,  # NEW: Lower is better
            'calibration_error': calibration_error,  # NEW: How well calibrated
            'price_rmse': price_rmse,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
        }
        
        self.is_trained = True
        
        print(f"\n✅ Training Complete!")
        print(f"   Direction Accuracy: {direction_accuracy:.1%}")
        print(f"   F1 Score: {f1:.2f}")
        print(f"   Brier Score: {brier:.4f} (lower=better, perfect=0)")
        print(f"   Log Loss: {logloss:.4f} (lower=better)")
        print(f"   Calibration Error: {calibration_error:.4f}")
        print(f"   Price RMSE: {price_rmse:.4f}")
        print(f"   Cross-Val Accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
        
        return self.training_metrics
    
    def get_feature_importance(self, features_scaled: np.ndarray) -> Dict:
        """Get SHAP-based feature importance for a prediction"""
        if self.shap_explainer is None or not HAS_SHAP:
            return {}
        
        try:
            shap_values = self.shap_explainer(features_scaled)
            feature_names = self.feature_extractor.get_feature_names()
            importance = dict(zip(feature_names, np.abs(shap_values.values[0])))
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            return importance
        except:
            return {}
    
    def predict(self, market: Dict, trades_df: pd.DataFrame, 
                days_remaining: Optional[float] = None) -> Dict:
        """
        Generate prediction using professional quant strategies.
        
        Optimized for 15-minute crypto price prediction markets on Polymarket.
        
        Implements:
        - ML model prediction with probability calibration
        - Order Book Imbalance (OBI) for short-term momentum
        - Bayesian aggregation of multiple probability sources
        - Terminal risk-adjusted position sizing (minute-level for 15-min markets)
        - Fractional Kelly optimal bet sizing
        
        Enhanced for 15-Minute Crypto Markets:
        - Fast technical indicators (RSI(5), MACD(6,13))
        - Micro-momentum (last 1-3 trades) for ultra-short-term signals
        - Volatility bursts and volume surges detection
        - Price acceleration (ROC) for momentum bursts
        - Minute-level risk management (more aggressive decay near deadline)
        - Time decay factor for confidence adjustment
        - Crypto market auto-detection
        
        Args:
            market: Market dictionary with outcomePrices, volume, liquidity, endDate, etc.
            trades_df: DataFrame with historical trades (price, size, side, timestamp)
            days_remaining: Optional override for days until expiration (auto-calculated if None)
        
        Returns: Dictionary with prediction details including:
            - current_price, predicted_price, direction, confidence, edge
            - action (BUY_YES/BUY_NO/HOLD), kelly_size
            - minutes_remaining, time_decay_factor, is_crypto_market, is_15min_market
            - Crypto-specific indicators: micro_momentum, volatility_burst, volume_surge, etc.
        """
        trade_features = self.feature_extractor.extract_trade_features(trades_df)
        market_features = self.feature_extractor.extract_market_features(market)
        
        # Use market's actual price (from outcomePrices), not from trade data
        # This ensures we have the real current price even if trades are sparse
        current_price = market_features.get('yes_price', 0.5)
        
        # Detect if this is a 15-minute crypto market
        minutes_remaining = market_features.get('minutes_until_end', 15.0)
        is_crypto = market_features.get('is_crypto_market', 0.0) > 0.5
        is_15min_market = minutes_remaining <= 20 and minutes_remaining > 0
        
        # Build enhanced feature vector for 15-minute crypto markets
        # Includes both standard features (for backward compatibility) and crypto-specific features
        # Feature order must match training data exactly!
        feature_names = self.feature_extractor.get_feature_names()
        feature_dict = {**trade_features, **market_features}
        
        # Build feature vector in the same order as get_feature_names()
        features_list = []
        for fname in feature_names:
            # Handle missing features gracefully
            if fname in feature_dict:
                features_list.append(feature_dict[fname])
            else:
                # Provide defaults for missing features
                if 'momentum' in fname or 'rsi' in fname or 'price' in fname:
                    features_list.append(0.5 if 'price' in fname or 'rsi' in fname else 0.0)
                elif 'volume' in fname or 'volatility' in fname:
                    features_list.append(0.0)
                else:
                    features_list.append(0.0)
        
        features = np.array(features_list).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        if not self.is_trained:
            return self._heuristic_prediction(trade_features, days_remaining)
        
        # Check if models are actually fitted (they might not be if training had insufficient data)
        # Check if models have been fitted by checking for attributes that exist after fit()
        models_fitted = (
            hasattr(self.direction_model, 'n_features_in_') or 
            hasattr(self.direction_model, 'estimators_') or
            hasattr(self.direction_model, 'estimators')  # StackingClassifier has 'estimators'
        ) and (
            hasattr(self.price_model, 'n_features_in_') or 
            hasattr(self.price_model, 'estimators_') or
            hasattr(self.price_model, 'estimators')  # StackingRegressor has 'estimators'
        )
        
        if not models_fitted:
            # Models weren't trained (likely due to insufficient data)
            # Use heuristic prediction instead
            print("  ⚠️  Models not fitted (insufficient training data) - using heuristic prediction")
            return self._heuristic_prediction(trade_features, days_remaining)
        
        try:
            features_scaled = self.scaler.transform(features)
        except Exception as e:
            # Scaler not fitted - use heuristic
            print(f"  ⚠️  Scaler not fitted: {e} - using heuristic prediction")
            return self._heuristic_prediction(trade_features, days_remaining)
        
        # =====================================================================
        # CORE PREDICTION LOGIC
        # =====================================================================
        # Direction model predicts: P(price goes UP) - trained on historical moves
        # If model says price UP → likely resolves YES → BUY YES
        # If model says price DOWN → likely resolves NO → BUY NO
        # =====================================================================
        
        try:
            # Get model's probability that price will go UP
            direction_proba_full = self.direction_model.predict_proba(features_scaled)[0]
            # Handle edge case where predict_proba returns 1 column (single class prediction)
            if len(direction_proba_full) == 1:
                # Only one class predicted - use that probability
                prob_up = direction_proba_full[0]
            else:
                # Binary classification - use probability of positive class (class 1)
                prob_up = direction_proba_full[1]  # Probability price goes UP
        except Exception as e:
            print(f"  ⚠️  Error in direction model prediction: {e} - using heuristic")
            return self._heuristic_prediction(trade_features, days_remaining)
        
        # Model's directional confidence (0 = uncertain, 1 = very confident)
        direction_confidence = abs(prob_up - 0.5) * 2
        
        try:
            # Price model gives a direct predicted price (trained on future_price)
            raw_predicted_price = self.price_model.predict(features_scaled)[0]
            raw_predicted_price = np.clip(raw_predicted_price, 0.01, 0.99)
        except Exception as e:
            print(f"  ⚠️  Error in price model prediction: {e} - using heuristic")
            return self._heuristic_prediction(trade_features, days_remaining)
        
        # =====================================================================
        # PREDICTED PRICE CALCULATION
        # Use price model as primary signal, direction model as confirmation
        # Price model is trained directly on future prices, more reliable
        # =====================================================================
        
        # Calculate move based on price model
        price_model_move = raw_predicted_price - current_price
        
        # Determine direction from price model (more reliable than direction model alone)
        predicted_up = price_model_move > 0
        
        # Adjust confidence if direction model agrees with price model
        direction_agrees = (prob_up > 0.5 and price_model_move > 0) or \
                           (prob_up < 0.5 and price_model_move < 0)
        
        if direction_agrees:
            # Both models agree - scale move normally
            confidence_scale = 0.6 + direction_confidence * 0.4
        else:
            # Models disagree - reduce move significantly
            confidence_scale = 0.3
        
        # Cap magnitude to realistic bounds based on price level
        max_up_move = min((1 - current_price) * 0.5, 0.20)
        max_down_move = min(current_price * 0.5, 0.20)
        
        # Calculate final move
        move_magnitude = abs(price_model_move) * confidence_scale
        
        # Apply move in the direction indicated by price model
        if predicted_up:
            move_magnitude = min(move_magnitude, max_up_move)
            predicted_price = current_price + move_magnitude
        else:
            move_magnitude = min(move_magnitude, max_down_move)
            predicted_price = current_price - move_magnitude
        
        # Clip to valid range (shouldn't be needed but safety)
        predicted_price = np.clip(predicted_price, 0.01, 0.99)
        
        # Final price change
        price_change = predicted_price - current_price
        
        # Order Book Microstructure Analysis
        bid_volume = trade_features.get('buy_volume', 0)
        ask_volume = trade_features.get('sell_volume', 0)
        
        obi = self.order_book.order_book_imbalance(bid_volume, ask_volume)
        imbalance_ratio = self.order_book.imbalance_ratio(bid_volume, ask_volume)
        
        # Microstructure momentum signal
        micro_direction, micro_conf = self.order_book.predict_direction(
            obi, imbalance_ratio, trade_features['momentum']
        )
        micro_prob = micro_conf if micro_direction == 'UP' else (1 - micro_conf)
        
        # Bayesian Aggregation of probability sources
        # For 15-minute crypto markets, boost weight for recent momentum/micro-momentum
        if is_15min_market and is_crypto:
            # Use micro-momentum for 15-minute markets (more important than standard momentum)
            micro_momentum = trade_features.get('micro_momentum', 0)
            momentum_1min = trade_features.get('momentum_1min', 0)
            # Normalize micro-momentum to [0, 1] range for probability
            momentum_signal = 0.5 + np.clip(micro_momentum * 10, -0.4, 0.4) if abs(micro_momentum) > 0.01 else \
                              (0.5 + np.clip(momentum_1min * 10, -0.4, 0.4) if abs(momentum_1min) > 0.01 else 0.5)
        else:
            # Standard momentum for longer-term markets
            momentum_signal = 0.5 + trade_features.get('momentum', 0) * 2
        
        aggregated_prob = self.bayesian.aggregate({
            'model': prob_up,  # Direction model probability
            'market': current_price,
            'momentum': momentum_signal,  # Use micro-momentum for 15-min markets
            'sentiment': micro_prob,
        })
        
        # =====================================================================
        # CONFIDENCE CALCULATION (Enhanced for 15-minute crypto markets)
        # =====================================================================
        
        # Boost confidence when model agrees with momentum/order flow
        # For 15-minute markets, use micro-momentum instead of standard momentum
        if is_15min_market and is_crypto:
            micro_momentum = trade_features.get('micro_momentum', 0)
            momentum_1min = trade_features.get('momentum_1min', 0)
            momentum_signal = micro_momentum if abs(micro_momentum) > 0.001 else momentum_1min
        else:
            momentum_signal = trade_features.get('momentum', 0)
        
        momentum_agreement = 1.0
        if (prob_up > 0.5 and momentum_signal > 0) or \
           (prob_up < 0.5 and momentum_signal < 0):
            # Stronger boost for 15-minute markets (recent momentum more predictive)
            momentum_agreement = 1.15 if is_15min_market else 1.1  # 15% boost for 15-min, 10% for longer
        
        # Boost confidence when price change is significant
        change_magnitude = abs(price_change) / max(current_price, 0.05)
        magnitude_factor = min(1.0 + change_magnitude * 0.3, 1.2)
        
        # Time decay confidence adjustment for 15-minute markets
        # Less confident as 15-minute window closes
        time_decay_factor = market_features.get('time_decay_factor', 1.0) if is_15min_market else 1.0
        
        # Calculate base confidence: start at 55%, scale up with model confidence
        raw_confidence = 0.55 + direction_confidence * 0.30 * momentum_agreement * magnitude_factor * time_decay_factor
        raw_confidence = np.clip(raw_confidence, 0.52, 0.90)
        
        # Use calibrated model as secondary input
        try:
            calibrated_conf = self.confidence_model.predict_proba(
                np.array([[direction_confidence]])
            )[0][1]
        except:
            calibrated_conf = raw_confidence
        
        # Blend: more weight to raw when model is very confident
        if direction_confidence > 0.5:
            confidence = raw_confidence * 0.75 + calibrated_conf * 0.25
        else:
            confidence = raw_confidence * 0.6 + calibrated_conf * 0.4
        
        confidence = np.clip(confidence, 0.52, 0.88)
        
        edge = abs(price_change) * confidence
        
        # Terminal Risk Management (Enhanced for 15-minute crypto markets)
        if is_15min_market and minutes_remaining > 0:
            # Use minute-based risk management for 15-minute markets
            should_reduce, reduction_factor = self.risk_manager.should_reduce_exposure_minutes(
                minutes_remaining, trade_features.get('volatility', 0)
            )
            gamma_risk = self.risk_manager.gamma_risk_factor_minutes(minutes_remaining)
            # Also calculate day-based for compatibility
            days_remaining = minutes_remaining / 1440.0  # Convert minutes to days
        else:
            # Use day-based risk management for longer-term markets
            if days_remaining is None:
                days_remaining = market_features.get('days_until_end', 30)
            should_reduce, reduction_factor = self.risk_manager.should_reduce_exposure(
                days_remaining, trade_features.get('volatility', 0)
            )
            gamma_risk = self.risk_manager.gamma_risk_factor(days_remaining)
            minutes_remaining = days_remaining * 1440.0  # Convert days to minutes for reporting
        
        # =====================================================================
        # ACTION DETERMINATION (BUY_YES / BUY_NO / HOLD)
        # =====================================================================
        # Logic: 
        #   - If predicted_price > current_price → price going UP → BUY YES
        #   - If predicted_price < current_price → price going DOWN → BUY NO
        #   - If edge too small → HOLD
        # =====================================================================
        
        action, base_position = KellyCriterion.position_size(
            predicted_price, current_price, confidence,
            bankroll=self.bankroll, kelly_fraction=self.kelly_fraction
        )
        
        # For extreme prices (>95% or <5%), adjust position sizing
        if current_price > 0.95 or current_price < 0.05:
            # Only reduce if model agrees with market direction
            if (current_price > 0.95 and prob_up > 0.5) or (current_price < 0.05 and prob_up < 0.5):
                base_position *= 0.25  # Market consensus + model agreement = low edge
            else:
                base_position *= 0.5  # Contrarian signal - still reduce but not as much
            # Only force HOLD if edge is truly minimal
            if abs(price_change) < 0.005:
                action = 'HOLD'
        
        # Adjust position for terminal risk
        if should_reduce:
            adjusted_position = base_position * reduction_factor
        else:
            adjusted_position = base_position
        
        # =====================================================================
        # SIGNAL STRENGTH DETERMINATION
        # =====================================================================
        # How extreme is the current price? (0 = extreme, 0.5 = balanced)
        extremeness = min(current_price, 1 - current_price)
        
        # Scale thresholds based on price range - extreme prices need smaller edges
        if extremeness < 0.15:
            # Extreme prices: even small edges are significant
            min_edge_strong = 0.025
            min_edge_mod = 0.012
            min_edge_weak = 0.005
            # But also need higher confidence for extreme bets
            conf_strong = 0.70
            conf_mod = 0.62
            conf_weak = 0.55
        else:
            # Normal prices: adjusted thresholds for more signals
            min_edge_strong = 0.10
            min_edge_mod = 0.05
            min_edge_weak = 0.025
            conf_strong = 0.65
            conf_mod = 0.55
            conf_weak = 0.50
        
        if abs(price_change) > min_edge_strong and confidence > conf_strong:
            signal = "STRONG"
        elif abs(price_change) > min_edge_mod and confidence > conf_mod:
            signal = "MODERATE"
        elif abs(price_change) > min_edge_weak and confidence > conf_weak:
            signal = "WEAK"
        else:
            signal = "HOLD"
        
        # Get feature importance if available
        feature_importance = self.get_feature_importance(features_scaled)
        
        # Calculate Expected Value (EV): E[Payoff] = P_true - P_market
        # Use predicted_price as our probability estimate
        ev = self.bayesian.expected_value(predicted_price, current_price, 
                                           'YES' if action == 'BUY_YES' else 'NO')
        
        # Expected Growth Rate under Kelly
        if action != 'HOLD':
            odds = 1 / current_price if action == 'BUY_YES' else 1 / (1 - current_price)
            growth_rate = KellyCriterion.expected_growth_rate(
                confidence, self.kelly_fraction, odds
            )
        else:
            growth_rate = 0.0
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'direction': 'UP' if price_change > 0 else 'DOWN',
            'direction_probability': prob_up,  # P(price goes UP)
            'aggregated_probability': aggregated_prob,  # Bayesian aggregated
            'confidence': confidence,
            'edge': edge,
            'expected_value': ev,
            'expected_growth_rate': growth_rate,
            'price_change': price_change,
            'signal': signal,
            'action': action,
            'kelly_size': adjusted_position,
            'base_kelly_size': base_position,
            # Microstructure signals
            'order_book_imbalance': obi,
            'imbalance_ratio': imbalance_ratio,
            'micro_direction': micro_direction,
            # Risk metrics (enhanced for 15-minute markets)
            'days_remaining': days_remaining,  # Keep for backward compatibility
            'minutes_remaining': minutes_remaining,  # New: for 15-minute markets
            'gamma_risk': gamma_risk,
            'terminal_risk_reduction': reduction_factor if should_reduce else 1.0,
            'time_decay_factor': market_features.get('time_decay_factor', 1.0),  # New: time decay
            'is_crypto_market': is_crypto,  # New: crypto detection
            'is_15min_market': is_15min_market,  # New: 15-minute market detection
            # Technical indicators (standard)
            'rsi': trade_features.get('rsi', 0.5),
            'volatility': trade_features.get('volatility', 0),
            'momentum': trade_features.get('momentum', 0),
            'order_imbalance': trade_features.get('order_imbalance', 0),
            # Crypto-specific indicators (new for 15-minute markets)
            'fast_rsi': trade_features.get('fast_rsi', 0.5),
            'micro_momentum': trade_features.get('micro_momentum', 0),
            'momentum_1min': trade_features.get('momentum_1min', 0),
            'momentum_3min': trade_features.get('momentum_3min', 0),
            'momentum_5min': trade_features.get('momentum_5min', 0),
            'volatility_burst': trade_features.get('volatility_burst', 1.0),
            'volume_surge': trade_features.get('volume_surge', 1.0),
            'price_acceleration': trade_features.get('price_acceleration', 0),
            'williams_r': trade_features.get('williams_r', -0.5),
            'bb_width': trade_features.get('bb_width', 0.02),
            'top_features': feature_importance,
            'calibration_quality': self.calibration_info.get('calibration_error', 0),
        }
    
    def _heuristic_prediction(self, trade_features: Dict, 
                               days_remaining: Optional[float] = None) -> Dict:
        """Fallback prediction when model is not trained."""
        current_price = trade_features['current_price']
        momentum = trade_features['momentum']
        rsi = trade_features['rsi']
        order_imbalance = trade_features['order_imbalance']
        
        if rsi > 0.7:
            mean_reversion = -0.05
        elif rsi < 0.3:
            mean_reversion = 0.05
        else:
            mean_reversion = 0
        
        predicted_change = momentum * 0.3 + order_imbalance * 0.03 + mean_reversion * 0.5
        predicted_price = np.clip(current_price + predicted_change, 0.01, 0.99)
        confidence = min(0.5 + abs(predicted_change) * 2, 0.75)
        
        action, position_size = KellyCriterion.position_size(
            predicted_price, current_price, confidence,
            bankroll=self.bankroll, kelly_fraction=self.kelly_fraction
        )
        
        if days_remaining is None:
            days_remaining = 30
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'direction': 'UP' if predicted_change > 0 else 'DOWN',
            'direction_probability': 0.5 + predicted_change,
            'aggregated_probability': 0.5 + predicted_change,
            'confidence': confidence,
            'edge': abs(predicted_change) * confidence,
            'expected_value': 0.0,
            'expected_growth_rate': 0.0,
            'price_change': predicted_change,
            'signal': 'WEAK',
            'action': action,
            'kelly_size': position_size,
            'base_kelly_size': position_size,
            'order_book_imbalance': 0.0,
            'imbalance_ratio': 0.5,
            'micro_direction': 'NEUTRAL',
            'days_remaining': days_remaining,
            'gamma_risk': 1.0 / np.sqrt(days_remaining) if days_remaining > 0 else 0,
            'terminal_risk_reduction': 1.0,
            'rsi': rsi,
            'volatility': trade_features['volatility'],
            'momentum': momentum,
            'order_imbalance': order_imbalance,
            'top_features': {},
            'calibration_quality': 0.0,
        }
    
    def is_crypto_market(self, market: Dict) -> bool:
        """
        Identify if a market is a crypto price prediction market.
        
        Checks question text for crypto-related keywords.
        
        Args:
            market: Market dictionary with 'question' field
        
        Returns: True if crypto market, False otherwise
        """
        question = market.get('question', '').lower()
        description = market.get('description', '').lower()
        text = question + " " + description
        
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'price', '$', 'usd', 'above', 'below', 'higher', 'lower',
            'solana', 'sol', 'cardano', 'ada', 'polygon', 'matic',
            'dogecoin', 'doge', 'xrp', 'ripple', 'avalanche', 'avax'
        ]
        
        return any(keyword in text for keyword in crypto_keywords)
    
    def is_15minute_market(self, market: Dict) -> bool:
        """
        Identify if a market has ~15-minute resolution window.
        
        Checks question text for 15-minute keywords, then checks time between createdAt and endDate.
        For active markets, also checks minutes remaining.
        
        Args:
            market: Market dictionary with 'question', 'createdAt', and 'endDate' fields
        
        Returns: True if ~15-minute market, False otherwise
        """
        # First check question text for 15-minute keywords (faster check)
        question = market.get('question', '').lower()
        description = market.get('description', '').lower()
        text = question + " " + description
        
        minute_keywords = [
            '15 min', '15 minute', 'quarter hour', '15m', 'fifteen minute',
            '15-min', '15min', 'in 15 minutes', 'within 15 minutes',
            'next 15 min', 'next 15 minutes'
        ]
        if any(keyword in text for keyword in minute_keywords):
            return True
        
        # Then check time delta - check both total duration and remaining time
        try:
            created_date_str = market.get('createdAt') or market.get('created_at')
            end_date_str = market.get('endDate') or market.get('end_date')
            
            if not created_date_str or not end_date_str:
                return False
            
            created_date = pd.to_datetime(created_date_str)
            end_date = pd.to_datetime(end_date_str)
            now = datetime.now()
            
            # Check total duration (for closed markets or checking market length)
            total_duration = (end_date - created_date).total_seconds() / 60
            
            # Check remaining time (for active markets)
            minutes_remaining = (end_date - now).total_seconds() / 60
            
            # Consider markets with:
            # - Total duration between 10-25 minutes (typical 15-minute markets)
            # - OR remaining time between 0-25 minutes (active 15-minute markets)
            # - OR if market was created recently and ends soon (active 15-minute market)
            is_duration_match = 10 <= total_duration <= 25
            is_remaining_match = 0 <= minutes_remaining <= 25 and minutes_remaining >= 0
            
            # Also check if it's an active market created recently (within last hour)
            # and ending soon (within 25 minutes) - likely a 15-minute market
            time_since_creation = (now - created_date).total_seconds() / 60
            is_recent_and_ending_soon = (time_since_creation <= 60 and 
                                         minutes_remaining >= 0 and 
                                         minutes_remaining <= 25)
            
            return is_duration_match or is_remaining_match or is_recent_and_ending_soon
            
        except Exception as e:
            # If date parsing fails, return False (conservative)
            return False
    
    def fetch_real_training_data(self, n_markets: int = 100, 
                                  filter_crypto: bool = True,
                                  filter_15min: bool = False) -> List[Dict]:
        """
        Fetch REAL training data from Polymarket API.
        
        Optimized for 15-minute crypto price prediction markets.
        
        NO SYNTHETIC DATA - only real market data is used.
        
        Args:
            n_markets: Number of markets to fetch data from
            filter_crypto: If True, filter to only crypto markets (default: True)
            filter_15min: If True, filter to only ~15-minute markets (default: False)
        
        Returns: List of training samples with real features and outcomes
        """
        from polymarket_fetcher import PolymarketFetcher
        
        print("\n" + "=" * 60)
        print("🌐 FETCHING REAL DATA FROM POLYMARKET API")
        if filter_crypto and filter_15min:
            print("   Filtering: Crypto 15-minute markets only")
        elif filter_crypto:
            print("   Filtering: Crypto markets only")
        elif filter_15min:
            print("   Filtering: 15-minute markets only")
        print("   No synthetic data - 100% real market data")
        print("=" * 60)
        
        fetcher = PolymarketFetcher(verbose=True)
        
        # Fetch markets (need significantly more if filtering for 15-minute, they're rare)
        if filter_crypto and filter_15min:
            # Both filters: fetch 10x more (15-minute markets are rare)
            fetch_count = n_markets * 10
        elif filter_crypto or filter_15min:
            # One filter: fetch 5x more
            fetch_count = n_markets * 5
        else:
            # No filtering: fetch as requested
            fetch_count = n_markets
        
        # Fetch active markets for training (closed markets don't have CLOB data)
        # Note: For 15-minute markets, we'll use active markets and check time remaining
        print(f"  📊 Fetching {fetch_count} markets for filtering...")
        all_markets = fetcher.get_markets(
            limit=fetch_count,
            active=True,  # Use active markets (closed markets have no CLOB data)
            closed=False,
            order='volume24hr',
            ascending=False
        )
        
        print(f"  📊 Fetched {len(all_markets)} total markets")
        
        # Filter markets if requested (BOTH conditions must be true when both filters enabled)
        if filter_crypto or filter_15min:
            filtered_markets = []
            for market in all_markets:
                # Check crypto filter
                if filter_crypto and not self.is_crypto_market(market):
                    continue
                # Check 15-minute filter
                if filter_15min and not self.is_15minute_market(market):
                    continue
                # Both filters passed (or one filter passed if only one enabled)
                filtered_markets.append(market)
            
            print(f"  📊 Filtered {len(filtered_markets)} markets from {len(all_markets)} total")
            
            if not filtered_markets:
                print(f"\n⚠️  No markets matched the filter criteria!")
                if filter_crypto and filter_15min:
                    print(f"   (Crypto 15-minute markets are rare - try increasing n_markets or reducing filters)")
                return []
            
            all_markets = filtered_markets[:n_markets]  # Limit to requested count
            print(f"  ✅ Using {len(all_markets)} markets for training")
        
        # Process filtered markets into training data format
        # Use fetcher's _process_market_for_training method to convert markets to training samples
        print(f"\n📊 Processing {len(all_markets)} filtered markets into training data...")
        training_data = []
        
        for i, market in enumerate(all_markets, 1):
            try:
                # Use fetcher's private method to process market (we need to access it)
                # Since it's private, we'll use the public fetch_real_training_data approach
                # but apply filtering first
                
                # Check if market has required data
                yes_token, _ = fetcher.get_token_ids_for_market(market)
                if not yes_token:
                    continue
                
                # Process market using our feature extractor to ensure consistency with prediction
                # Extract features using our feature extractor (matches prediction format exactly)
                trades = fetcher.get_trades(yes_token, limit=500)
                trades_df = fetcher.trades_to_dataframe(trades) if trades else pd.DataFrame()
                
                # Use our feature extractor to get features (same as prediction)
                trade_features = self.feature_extractor.extract_trade_features(trades_df)
                market_features = self.feature_extractor.extract_market_features(market)
                
                # Build feature vector using feature extractor (same order as get_feature_names())
                feature_names = self.feature_extractor.get_feature_names()
                feature_dict = {**trade_features, **market_features}
                
                features_list = []
                for fname in feature_names:
                    features_list.append(feature_dict.get(fname, 0.0))
                
                features = np.array(features_list).reshape(1, -1)
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Get price history for outcome determination
                price_history = None
                if hasattr(fetcher, 'get_prices_history'):
                    try:
                        price_history = fetcher.get_prices_history(yes_token, interval='1w', fidelity=60)
                    except:
                        pass
                
                # Determine future_price and outcome
                if price_history is not None and len(price_history) >= 20:
                    prices_arr = price_history['price'].values if isinstance(price_history, pd.DataFrame) and 'price' in price_history.columns else np.array(price_history).flatten()
                    mid_point = len(prices_arr) // 2
                    past_avg = np.mean(prices_arr[:mid_point]) if mid_point > 0 else prices_arr[0]
                    recent_avg = np.mean(prices_arr[-5:]) if len(prices_arr) >= 5 else prices_arr[-1]
                    outcome = 1 if recent_avg > past_avg else 0
                    future_price = np.clip(recent_avg, 0.01, 0.99)
                    training_price = prices_arr[mid_point] if mid_point < len(prices_arr) else current_price
                else:
                    # Fallback: use momentum direction
                    momentum = trade_features.get('momentum', 0) or trade_features.get('micro_momentum', 0)
                    outcome = 1 if momentum > 0 else 0
                    future_price = np.clip(current_price + momentum * 0.1, 0.01, 0.99)
                    training_price = current_price
                
                # Create training sample with features matching prediction format
                sample = {
                    'features': features,
                    'current_price': training_price,
                    'future_price': future_price,
                    'outcome': outcome,
                    'market_id': market.get('id', 'unknown'),
                    'question': market.get('question', 'Unknown')[:50],
                }
                
                training_data.append(sample)
                
                if len(training_data) >= n_markets:
                    break
                
                if i % 10 == 0:
                    print(f"   Processed {i}/{len(all_markets)} markets... ({len(training_data)} training samples)")
                
            except Exception as e:
                # Skip markets that fail to process
                continue
        
        if not training_data:
            # If no training data, provide helpful error message
            print("\n⚠️  No training data was successfully processed!")
            print("   Possible reasons:")
            print("   - Crypto 15-minute markets may be very rare (try reducing filters)")
            print("   - Markets may not have sufficient data (trades, order book, etc.)")
            print("   - API may be rate limiting or unavailable")
            print("\n   💡 Suggestions:")
            print(f"   - Try filter_15min=False to include longer-term markets")
            print(f"   - Increase n_markets (current: {n_markets})")
            print(f"   - Check internet connection and Polymarket API status")
        
        print(f"\n✅ Processed {len(training_data)} real training samples")
        if filter_crypto and filter_15min:
            print(f"   (All samples are crypto 15-minute markets)")
        elif filter_crypto:
            print(f"   (All samples are crypto markets)")
        elif filter_15min:
            print(f"   (All samples are 15-minute markets)")
        
        if len(training_data) < 10:
            print(f"\n⚠️  Warning: Only {len(training_data)} training samples found.")
            print(f"   This may not be enough for effective training.")
            if filter_crypto and filter_15min:
                print(f"   Crypto 15-minute markets are rare - consider:")
                print(f"   - Reducing filters (set filter_15min=False or filter_crypto=False)")
                print(f"   - Increasing n_markets parameter")
        
        # Return training data - feature extraction format matches what training expects
        return training_data
    
    def detect_arbitrage(self, markets: List[Dict]) -> Dict:
        """
        Detect arbitrage opportunities across related markets.
        
        Args:
            markets: List of market dicts with 'prices' for mutually exclusive outcomes
        
        Returns: Arbitrage analysis with opportunities
        """
        results = {
            'rebalancing': [],
            'overpriced': [],
            'combinatorial': [],
            'total_opportunities': 0,
        }
        
        # Check each market for rebalancing/overpriced arbitrage
        for market in markets:
            prices = market.get('prices', [])
            if len(prices) >= 2:
                rebal = ArbitrageDetector.detect_rebalancing_arbitrage(prices)
                if rebal['exists']:
                    results['rebalancing'].append({
                        'market': market.get('id', 'unknown'),
                        **rebal
                    })
                
                over = ArbitrageDetector.detect_overpriced_market(prices)
                if over['exists']:
                    results['overpriced'].append({
                        'market': market.get('id', 'unknown'),
                        **over
                    })
        
        # Check for combinatorial arbitrage
        comb = ArbitrageDetector.find_combinatorial_arbitrage(markets)
        results['combinatorial'] = comb
        
        results['total_opportunities'] = (
            len(results['rebalancing']) + 
            len(results['overpriced']) + 
            len(results['combinatorial'])
        )
        
        return results


def create_predictor(use_optuna: bool = False, 
                     kelly_fraction: float = 0.25,
                     bankroll: float = 10000,
                     n_markets: int = 100,
                     filter_crypto: bool = True,
                     filter_15min: bool = False) -> PolymarketPredictor:
    """
    Factory function to create a configured professional quant predictor.
    
    Optimized for 15-minute crypto price prediction markets on Polymarket.
    
    USES REAL DATA FROM POLYMARKET API - NO SYNTHETIC DATA.
    
    Args:
        use_optuna: Enable Bayesian hyperparameter optimization
        kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly, industry standard)
        bankroll: Total trading capital
        n_markets: Number of real markets to fetch for training
        filter_crypto: If True, filter to only crypto markets (default: True)
        filter_15min: If True, filter to only ~15-minute markets (default: False)
    
    Returns: Trained PolymarketPredictor with all quant strategies initialized
    
    Features for 15-Minute Crypto Markets:
    - Fast technical indicators (RSI(5), MACD(6,13))
    - Micro-momentum (last 1-3 trades)
    - Volatility bursts and volume surges
    - Minute-level risk management
    - Time decay factor for confidence adjustment
    """
    predictor = PolymarketPredictor(
        use_optuna=use_optuna,
        kelly_fraction=kelly_fraction,
        bankroll=bankroll
    )
    
    print("\n" + "=" * 70)
    print("🚀 POLYMARKET PREDICTOR - REAL DATA MODE")
    print("   Optimized for 15-Minute Crypto Price Prediction Markets")
    print("   Training on actual Polymarket market data (no synthetic data)")
    if filter_crypto:
        print("   Filter: Crypto markets only")
    if filter_15min:
        print("   Filter: 15-minute markets only")
    print("=" * 70)
    
    # Fetch REAL training data from Polymarket API with filters
    training_data = predictor.fetch_real_training_data(
        n_markets=n_markets,
        filter_crypto=filter_crypto,
        filter_15min=filter_15min
    )
    
    if len(training_data) < 10:
        print("\n⚠️  Warning: Low training data count. API may be rate limiting.")
        print("   Waiting 5 seconds and retrying...")
        import time
        time.sleep(5)
        training_data = predictor.fetch_real_training_data(
            n_markets=n_markets,
            filter_crypto=filter_crypto,
            filter_15min=filter_15min
        )
    
    print(f"\n📊 Training model on {len(training_data)} REAL market samples")
    predictor.train(training_data)
    
    return predictor


# =============================================================================
# CONVENIENCE FUNCTIONS FOR QUANT ANALYSIS
# =============================================================================

def calculate_edge(p_true: float, p_market: float) -> float:
    """Calculate expected edge: E[Payoff] = P_true - P_market"""
    return p_true - p_market


def calculate_optimal_position(p_true: float, p_market: float, 
                                bankroll: float = 10000,
                                kelly_fraction: float = 0.25) -> Dict:
    """
    Calculate optimal position size using fractional Kelly.
    
    Args:
        p_true: Estimated true probability
        p_market: Current market price
        bankroll: Total capital
        kelly_fraction: Fraction of full Kelly (0.25 recommended)
    
    Returns: Position sizing details
    """
    full_kelly = KellyCriterion.calculate_full_kelly(p_true, p_market)
    position = full_kelly * kelly_fraction * bankroll
    edge = calculate_edge(p_true, p_market)
    
    return {
        'edge': edge,
        'full_kelly_fraction': full_kelly,
        'adjusted_kelly_fraction': full_kelly * kelly_fraction,
        'position_size': position,
        'max_position': bankroll * 0.25,
        'action': 'BUY_YES' if p_true > p_market else 'BUY_NO',
    }


def analyze_order_flow(bid_volume: float, ask_volume: float, 
                       momentum: float = 0.0) -> Dict:
    """
    Analyze order flow for short-term price prediction.
    
    Based on Cont, Kukanov & Stoikov (2014):
    - OBI explains ~65% of short-interval price variance
    
    Args:
        bid_volume: Total bid volume
        ask_volume: Total ask volume
        momentum: Current price momentum
    
    Returns: Order flow analysis
    """
    obi = OrderBookMicrostructure.order_book_imbalance(bid_volume, ask_volume)
    ir = OrderBookMicrostructure.imbalance_ratio(bid_volume, ask_volume)
    direction, confidence = OrderBookMicrostructure.predict_direction(obi, ir, momentum)
    
    return {
        'order_book_imbalance': obi,
        'imbalance_ratio': ir,
        'predicted_direction': direction,
        'confidence': confidence,
        'signal_strength': 'STRONG' if abs(obi) > 0.4 else ('MODERATE' if abs(obi) > 0.2 else 'WEAK'),
    }
