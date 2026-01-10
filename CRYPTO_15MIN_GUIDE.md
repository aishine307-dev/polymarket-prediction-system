# Guide: Running Crypto 15-Minute Market Predictions on Polymarket

This guide explains how to use the enhanced prediction system for 15-minute crypto price prediction markets on Polymarket.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Options](#usage-options)
5. [Running Predictions](#running-predictions)
6. [Understanding Output](#understanding-output)
7. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

- Python 3.8 or higher
- Internet connection (for Polymarket API access)
- Basic understanding of prediction markets

---

## Installation

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /path/to/polymarket-prediction-system

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import numpy, pandas, sklearn, xgboost, lightgbm; print('✓ All dependencies installed')"
```

---

## Quick Start

### Option 1: Use the Enhanced Main Script (Recommended)

Create a new script `crypto_15min_main.py`:

```python
#!/usr/bin/env python3
"""
Crypto 15-Minute Market Predictor
Optimized for 15-minute crypto price prediction markets on Polymarket
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
from datetime import datetime
import pandas as pd

from polymarket_fetcher import PolymarketFetcher
from prediction_model import create_predictor

def main():
    print("\n" + "="*70)
    print("🚀 CRYPTO 15-MINUTE MARKET PREDICTOR")
    print("   Optimized for Polymarket crypto price prediction markets")
    print("="*70)
    
    # Create predictor optimized for crypto 15-minute markets
    predictor = create_predictor(
        use_optuna=False,          # Set True for hyperparameter optimization (slower)
        kelly_fraction=0.25,       # Quarter Kelly (industry standard)
        bankroll=10000,            # Starting capital
        n_markets=100,             # Fetch 100 markets for training
        filter_crypto=True,        # Filter to only crypto markets
        filter_15min=False         # Set True to filter to only 15-minute markets
    )
    
    # Initialize fetcher
    fetcher = PolymarketFetcher(verbose=True)
    
    # Fetch crypto markets (you may need to adjust filtering)
    print("\n📡 Fetching crypto markets...")
    markets = fetcher.get_markets(
        limit=200,                 # Fetch more to account for filtering
        order='volume24hr',        # Sort by 24h volume
        active=True
    )
    
    # Filter for crypto markets
    crypto_markets = []
    for market in markets:
        question = market.get('question', '').lower()
        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 
                          'price', '$', 'above', 'below', 'higher', 'lower']
        if any(keyword in question for keyword in crypto_keywords):
            crypto_markets.append(market)
        if len(crypto_markets) >= 10:  # Analyze top 10 crypto markets
            break
    
    print(f"✅ Found {len(crypto_markets)} crypto markets to analyze\n")
    
    # Analyze each market
    for i, market in enumerate(crypto_markets, 1):
        try:
            print(f"\n{'='*70}")
            print(f"Market #{i}: {market.get('question', 'Unknown')[:60]}")
            print('='*70)
            
            # Get token IDs for fetching trades
            yes_token, _ = fetcher.get_token_ids_for_market(market)
            
            # Get current price
            prices_str = market.get('outcomePrices', '[0.5, 0.5]')
            try:
                prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                current_price = float(prices[0]) if prices else 0.5
            except:
                current_price = 0.5
            
            # Fetch trade data
            trades_df = pd.DataFrame()
            if yes_token:
                trades = fetcher.get_trades(yes_token, limit=500)
                if trades:
                    trades_df = fetcher.trades_to_dataframe(trades)
            
            # If no trades, create minimal DataFrame
            if trades_df.empty:
                trades_df = pd.DataFrame({
                    'price': [current_price],
                    'size': [0],
                    'timestamp': [datetime.now()]
                })
            
            # Make prediction
            prediction = predictor.predict(market, trades_df)
            
            # Display results
            print(f"\n📊 Current Price: {prediction['current_price']:.2%}")
            print(f"📈 Predicted Price: {prediction['predicted_price']:.2%}")
            print(f"📉 Price Change: {prediction['price_change']*100:+.2f}%")
            print(f"🎯 Direction: {prediction['direction']}")
            print(f"💪 Confidence: {prediction['confidence']:.1%}")
            print(f"💰 Edge: {prediction['edge']:.4f}")
            print(f"📋 Signal: {prediction['signal']}")
            print(f"🎲 Action: {prediction['action']}")
            print(f"💵 Kelly Size: ${prediction['kelly_size']:.2f}")
            
            # Crypto-specific features
            if prediction.get('is_crypto_market'):
                print(f"\n🔍 Crypto-Specific Indicators:")
                print(f"   Micro-Momentum: {prediction.get('micro_momentum', 0):.4f}")
                print(f"   Momentum (1min): {prediction.get('momentum_1min', 0):.4f}")
                print(f"   Fast RSI: {prediction.get('fast_rsi', 0.5):.2%}")
                print(f"   Volatility Burst: {prediction.get('volatility_burst', 1.0):.2f}x")
                print(f"   Volume Surge: {prediction.get('volume_surge', 1.0):.2f}x")
                print(f"   Price Acceleration: {prediction.get('price_acceleration', 0):.4f}")
            
            # 15-minute market features
            if prediction.get('is_15min_market'):
                print(f"\n⏱️  15-Minute Market Features:")
                print(f"   Minutes Remaining: {prediction.get('minutes_remaining', 15):.1f}")
                print(f"   Time Decay Factor: {prediction.get('time_decay_factor', 1.0):.2f}")
                print(f"   Terminal Risk Reduction: {prediction.get('terminal_risk_reduction', 1.0):.2f}")
            
            # Risk metrics
            print(f"\n⚠️  Risk Metrics:")
            print(f"   Gamma Risk: {prediction.get('gamma_risk', 0):.4f}")
            print(f"   Expected Value: {prediction.get('expected_value', 0):+.4f}")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error analyzing market: {e}")
            continue
    
    print("\n" + "="*70)
    print("✅ Analysis Complete")
    print("="*70)

if __name__ == "__main__":
    main()
```

### Run the Script

```bash
python crypto_15min_main.py
```

---

## Usage Options

### Option 2: Using Python Interactive Mode

```python
from polymarket_fetcher import PolymarketFetcher
from prediction_model import create_predictor
import pandas as pd

# Create predictor for crypto 15-minute markets
predictor = create_predictor(
    filter_crypto=True,    # Filter to crypto markets
    filter_15min=False,    # Set True for only 15-min markets
    n_markets=100
)

# Initialize fetcher
fetcher = PolymarketFetcher(verbose=True)

# Fetch a specific crypto market
markets = fetcher.get_markets(limit=50, order='volume24hr')
crypto_market = [m for m in markets if 'bitcoin' in m.get('question', '').lower() or 
                 'btc' in m.get('question', '').lower()][0]

# Get trades
yes_token, _ = fetcher.get_token_ids_for_market(crypto_market)
trades = fetcher.get_trades(yes_token, limit=500)
trades_df = fetcher.trades_to_dataframe(trades)

# Make prediction
prediction = predictor.predict(crypto_market, trades_df)

# View results
print(f"Action: {prediction['action']}")
print(f"Confidence: {prediction['confidence']:.1%}")
print(f"Kelly Size: ${prediction['kelly_size']:.2f}")
```

### Option 3: Modify Existing main.py

Update `main.py` to filter for crypto markets:

```python
# In main.py, modify run_single_analysis():

def run_single_analysis(num_markets=10):
    print_header()
    
    fetcher = PolymarketFetcher(verbose=False)
    
    # Create predictor with crypto filtering
    predictor = create_predictor(
        use_optuna=False,
        filter_crypto=True,      # ADD THIS: Filter to crypto markets
        filter_15min=False,      # ADD THIS: Set True for only 15-min markets
        n_markets=100
    )
    
    # Rest of the function remains the same...
    # The existing filtering logic will work, but you may want to add crypto-specific filtering
```

---

## Running Predictions

### Basic Prediction

```python
from prediction_model import create_predictor
from polymarket_fetcher import PolymarketFetcher
import pandas as pd

# Step 1: Create predictor (trains on crypto markets)
predictor = create_predictor(
    filter_crypto=True,
    n_markets=100
)

# Step 2: Fetch market data
fetcher = PolymarketFetcher()
markets = fetcher.get_markets(limit=10, order='volume24hr')

# Step 3: Get market and trades
market = markets[0]  # Or filter for specific crypto market
yes_token, _ = fetcher.get_token_ids_for_market(market)
trades = fetcher.get_trades(yes_token, limit=500)
trades_df = fetcher.trades_to_dataframe(trades)

# Step 4: Make prediction
prediction = predictor.predict(market, trades_df)

# Step 5: Interpret results
print(f"Action: {prediction['action']}")  # BUY_YES, BUY_NO, or HOLD
print(f"Confidence: {prediction['confidence']:.1%}")
print(f"Position Size: ${prediction['kelly_size']:.2f}")
```

### Filtering for 15-Minute Markets Only

```python
# Train on only 15-minute crypto markets
predictor = create_predictor(
    filter_crypto=True,
    filter_15min=True,  # Only 15-minute markets
    n_markets=50        # May need fewer due to filtering
)
```

---

## Understanding Output

### Prediction Dictionary Keys

The `predict()` method returns a dictionary with these keys:

**Basic Prediction:**
- `current_price`: Current market price (0.0-1.0)
- `predicted_price`: Model's predicted price (0.0-1.0)
- `price_change`: Predicted change (predicted - current)
- `direction`: 'UP' or 'DOWN'
- `confidence`: Model confidence (0.52-0.88)
- `edge`: Expected advantage (abs(price_change) * confidence)
- `signal`: 'STRONG', 'MODERATE', 'WEAK', or 'HOLD'
- `action`: 'BUY_YES', 'BUY_NO', or 'HOLD'
- `kelly_size`: Recommended position size in dollars

**Crypto-Specific Indicators:**
- `micro_momentum`: Last 1-3 trades momentum
- `momentum_1min`: 1-minute momentum
- `momentum_3min`: 3-minute momentum
- `momentum_5min`: 5-minute momentum
- `fast_rsi`: Fast RSI(5) indicator
- `volatility_burst`: Recent volatility vs average
- `volume_surge`: Recent volume vs average
- `price_acceleration`: Rate of change of momentum
- `williams_r`: Williams %R indicator

**15-Minute Market Features:**
- `minutes_remaining`: Time until market resolution
- `time_decay_factor`: Exponential decay factor (0.1-1.0)
- `is_crypto_market`: Boolean (1.0 if crypto, 0.0 otherwise)
- `is_15min_market`: Boolean (1.0 if 15-minute market)
- `terminal_risk_reduction`: Position reduction factor (0.1-1.0)

**Risk Metrics:**
- `gamma_risk`: Gamma risk factor (higher = more risk)
- `expected_value`: Expected value of trade
- `expected_growth_rate`: Kelly growth rate

**Technical Indicators:**
- `rsi`: Standard RSI(14)
- `volatility`: Price volatility
- `momentum`: Price momentum
- `order_imbalance`: Buy/sell imbalance
- `order_book_imbalance`: OBI from microstructure analysis

---

## Advanced Configuration

### Custom Configuration Example

```python
from prediction_model import PolymarketPredictor

# Create predictor with custom settings
predictor = PolymarketPredictor(
    use_optuna=True,          # Enable hyperparameter optimization
    use_calibration=True,     # Enable probability calibration
    kelly_fraction=0.20,      # More conservative (20% of Kelly)
    bankroll=50000            # Larger bankroll
)

# Manually fetch and filter training data
training_data = predictor.fetch_real_training_data(
    n_markets=200,
    filter_crypto=True,       # Only crypto markets
    filter_15min=True         # Only 15-minute markets
)

# Train the model
metrics = predictor.train(training_data)

# Check training quality
print(f"Brier Score: {metrics['brier_score']:.4f}")  # Lower is better
print(f"Calibration Error: {metrics['calibration_error']:.4f}")  # Lower is better
```

### Real-Time Prediction Loop

```python
import time
from datetime import datetime

def monitor_crypto_15min_markets(predictor, fetcher, interval=60):
    """
    Monitor crypto 15-minute markets every N seconds
    
    Args:
        predictor: Trained PolymarketPredictor instance
        fetcher: PolymarketFetcher instance
        interval: Seconds between checks
    """
    print("🔄 Starting real-time monitoring...")
    print(f"   Checking every {interval} seconds\n")
    
    while True:
        try:
            # Fetch crypto markets
            markets = fetcher.get_markets(limit=50, order='volume24hr', active=True)
            
            # Filter for crypto markets
            crypto_markets = [m for m in markets if predictor.is_crypto_market(m)]
            
            # Filter for 15-minute markets
            crypto_15min = [m for m in crypto_markets if predictor.is_15minute_market(m)]
            
            if not crypto_15min:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No active 15-minute crypto markets")
                time.sleep(interval)
                continue
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(crypto_15min)} active markets")
            
            # Analyze each market
            for market in crypto_15min:
                try:
                    yes_token, _ = fetcher.get_token_ids_for_market(market)
                    trades = fetcher.get_trades(yes_token, limit=500) if yes_token else []
                    trades_df = fetcher.trades_to_dataframe(trades) if trades else pd.DataFrame()
                    
                    prediction = predictor.predict(market, trades_df)
                    
                    # Only show strong signals
                    if prediction['signal'] in ['STRONG', 'MODERATE']:
                        print(f"\n📊 {market.get('question', 'Unknown')[:50]}")
                        print(f"   Action: {prediction['action']} | "
                              f"Confidence: {prediction['confidence']:.0%} | "
                              f"Edge: {prediction['edge']:.2%}")
                        print(f"   Minutes Remaining: {prediction.get('minutes_remaining', 0):.1f}")
                    
                except Exception as e:
                    print(f"Error analyzing {market.get('question', 'Unknown')}: {e}")
                    continue
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(interval)

# Usage
if __name__ == "__main__":
    predictor = create_predictor(filter_crypto=True, n_markets=100)
    fetcher = PolymarketFetcher(verbose=False)
    monitor_crypto_15min_markets(predictor, fetcher, interval=60)
```

---

## Troubleshooting

### Issue: "No crypto markets found"

**Solution:** The market detection uses keyword matching. If markets don't contain expected keywords, they won't be detected. You can customize the `is_crypto_market()` method in `prediction_model.py` to add more keywords.

### Issue: "Insufficient training data"

**Solution:** Increase `n_markets` parameter or reduce filtering:
```python
predictor = create_predictor(
    n_markets=200,      # Increase from 100
    filter_crypto=True,
    filter_15min=False  # Don't filter 15-min if not enough data
)
```

### Issue: "API rate limiting"

**Solution:** Add delays between API calls:
```python
import time
time.sleep(0.5)  # Wait 0.5 seconds between requests
```

### Issue: "Feature mismatch errors"

**Solution:** Ensure training and prediction use the same feature extractor. The system automatically handles this, but if you modify features, retrain the model.

---

## Best Practices

1. **Start Conservative**: Use `kelly_fraction=0.20` (20% of Kelly) until confident
2. **Monitor Calibration**: Check Brier score - should be <0.20 for good calibration
3. **Filter Appropriately**: Use `filter_15min=True` only if you have enough 15-minute markets
4. **Check Data Quality**: Verify markets have sufficient volume and liquidity
5. **Monitor Time Decay**: For 15-minute markets, watch `time_decay_factor` - very low means deadline approaching
6. **Use Crypto Features**: Pay attention to `micro_momentum`, `volatility_burst`, and `volume_surge` for crypto markets
7. **Position Sizing**: Never exceed 25% of bankroll per trade (already enforced)

---

## Example Output

```
🚀 CRYPTO 15-MINUTE MARKET PREDICTOR
   Optimized for Polymarket crypto price prediction markets
======================================================================

🚀 POLYMARKET PREDICTOR - REAL DATA MODE
   Optimized for 15-Minute Crypto Price Prediction Markets
   Training on actual Polymarket market data (no synthetic data)
   Filter: Crypto markets only
======================================================================

✅ Found 10 crypto markets to analyze

======================================================================
Market #1: Will Bitcoin (BTC) be above $45,000 at 3:00 PM EST today?
======================================================================

📊 Current Price: 65.00%
📈 Predicted Price: 68.50%
📉 Price Change: +3.50%
🎯 Direction: UP
💪 Confidence: 72%
💰 Edge: 0.0252
📋 Signal: MODERATE
🎲 Action: BUY_YES
💵 Kelly Size: $180.25

🔍 Crypto-Specific Indicators:
   Micro-Momentum: 0.0234
   Momentum (1min): 0.0156
   Fast RSI: 58.32%
   Volatility Burst: 1.45x
   Volume Surge: 1.32x
   Price Acceleration: 0.0012

⏱️  15-Minute Market Features:
   Minutes Remaining: 12.5
   Time Decay Factor: 0.85
   Terminal Risk Reduction: 0.92

⚠️  Risk Metrics:
   Gamma Risk: 0.2828
   Expected Value: +0.0350
```

---

## Next Steps

1. **Backtest**: Test predictions on historical data
2. **Optimize**: Adjust `kelly_fraction` and `bankroll` based on risk tolerance
3. **Monitor**: Track prediction accuracy over time
4. **Refine**: Add more crypto-specific features based on performance

---

## Support

For issues or questions:
- Check the `TUTORIAL_prediction_model.md` for detailed concepts
- Review `CONCEPTS_SUMMARY.md` for quick reference
- Examine `prediction_model.py` docstrings for API details

---

**Disclaimer:** This software is for educational purposes only. Not financial advice. Prediction markets involve substantial risk. Always do your own research and never invest more than you can afford to lose.
