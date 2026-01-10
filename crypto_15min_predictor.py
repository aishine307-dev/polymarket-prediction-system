#!/usr/bin/env python3
"""
Crypto 15-Minute Market Predictor - Quick Start Script
Simple script to run predictions on crypto 15-minute markets on Polymarket
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import time
from datetime import datetime
import pandas as pd

from polymarket_fetcher import PolymarketFetcher
from prediction_model import create_predictor


def print_separator():
    print("=" * 70)


def print_header():
    print_separator()
    print("🚀 CRYPTO 15-MINUTE MARKET PREDICTOR")
    print("   Optimized for Polymarket crypto price prediction markets")
    print_separator()


def format_price(price):
    """Format price as percentage"""
    return f"{price:.2%}"


def format_money(amount):
    """Format amount as dollars"""
    return f"${amount:,.2f}"


def analyze_market(predictor, fetcher, market, rank=None):
    """Analyze a single crypto market and return formatted results"""
    
    try:
        question = market.get('question', 'Unknown Market')
        
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
        
        # Format results
        rank_str = f"#{rank} " if rank else ""
        question_short = question[:55] + "..." if len(question) > 55 else question
        
        print(f"\n{rank_str}{question_short}")
        print("-" * 70)
        
        # Basic prediction info
        current = prediction['current_price']
        predicted = prediction['predicted_price']
        change = prediction['price_change']
        direction = prediction['direction']
        confidence = prediction['confidence']
        signal = prediction['signal']
        action = prediction['action']
        edge = prediction['edge']
        kelly_size = prediction['kelly_size']
        
        print(f"📊 Price: {format_price(current)} → {format_price(predicted)} ({change*100:+.2f}%)")
        print(f"📈 Direction: {direction} | Confidence: {format_price(confidence)}")
        print(f"🎯 Signal: {signal} | Action: {action}")
        print(f"💰 Edge: {format_price(edge)} | Position Size: {format_money(kelly_size)}")
        
        # Crypto-specific indicators
        is_crypto = prediction.get('is_crypto_market', False)
        is_15min = prediction.get('is_15min_market', False)
        
        if is_crypto or is_15min:
            print(f"\n🔍 Advanced Indicators:")
            
            if is_crypto:
                micro_mom = prediction.get('micro_momentum', 0)
                mom_1min = prediction.get('momentum_1min', 0)
                fast_rsi = prediction.get('fast_rsi', 0.5)
                vol_burst = prediction.get('volatility_burst', 1.0)
                vol_surge = prediction.get('volume_surge', 1.0)
                price_accel = prediction.get('price_acceleration', 0)
                
                print(f"   Micro-Momentum: {micro_mom:+.4f}")
                print(f"   1-Min Momentum: {mom_1min:+.4f}")
                print(f"   Fast RSI: {format_price(fast_rsi)}")
                print(f"   Volatility Burst: {vol_burst:.2f}x")
                print(f"   Volume Surge: {vol_surge:.2f}x")
                print(f"   Price Acceleration: {price_accel:+.4f}")
            
            if is_15min:
                minutes_remaining = prediction.get('minutes_remaining', 15)
                time_decay = prediction.get('time_decay_factor', 1.0)
                risk_reduction = prediction.get('terminal_risk_reduction', 1.0)
                
                print(f"\n⏱️  15-Minute Market Info:")
                print(f"   Minutes Remaining: {minutes_remaining:.1f}")
                print(f"   Time Decay Factor: {time_decay:.2f}")
                print(f"   Risk Reduction: {risk_reduction:.2f}")
        
        # Risk metrics
        gamma_risk = prediction.get('gamma_risk', 0)
        ev = prediction.get('expected_value', 0)
        
        if gamma_risk > 0 or abs(ev) > 0.01:
            print(f"\n⚠️  Risk Metrics:")
            if gamma_risk > 0:
                print(f"   Gamma Risk: {gamma_risk:.4f}")
            if abs(ev) > 0.01:
                print(f"   Expected Value: {ev:+.4f}")
        
        # Recommendation
        if action == 'BUY_YES' and signal in ['STRONG', 'MODERATE']:
            print(f"\n✅ RECOMMENDATION: BUY YES (High confidence signal)")
        elif action == 'BUY_NO' and signal in ['STRONG', 'MODERATE']:
            print(f"\n✅ RECOMMENDATION: BUY NO (High confidence signal)")
        elif action == 'HOLD':
            print(f"\n⚪ RECOMMENDATION: HOLD (Insufficient edge or confidence)")
        
        return {
            'question': question,
            'action': action,
            'signal': signal,
            'confidence': confidence,
            'edge': edge,
            'kelly_size': kelly_size,
            'is_crypto': is_crypto,
            'is_15min': is_15min
        }
        
    except Exception as e:
        print(f"❌ Error analyzing market: {e}")
        return None


def main():
    """Main execution function"""
    
    print_header()
    
    # Configuration
    num_markets = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    filter_crypto = True   # Filter to crypto markets
    filter_15min = True    # Filter to only 15-minute markets (crypto 15-min markets)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Markets to analyze: {num_markets}")
    print(f"   Filter crypto: {filter_crypto}")
    print(f"   Filter 15-min: {filter_15min}")
    print(f"   Target: Crypto 15-minute markets only")
    print_separator()
    
    try:
        # Step 1: Create predictor (this will train the model)
        print("\n📚 Step 1: Creating and training predictor...")
        print("   This may take a few minutes as it fetches and trains on real data...")
        
        predictor = create_predictor(
            use_optuna=False,          # Set True for hyperparameter optimization (slower)
            kelly_fraction=0.25,       # Quarter Kelly (industry standard)
            bankroll=10000,            # Starting capital
            n_markets=100,             # Fetch 100 markets for training
            filter_crypto=filter_crypto,   # Filter to crypto markets
            filter_15min=filter_15min      # Filter to 15-minute markets
        )
        
        print("✅ Predictor trained successfully!")
        
        # Step 2: Initialize fetcher
        print("\n📡 Step 2: Initializing market fetcher...")
        fetcher = PolymarketFetcher(verbose=False)
        print("✅ Fetcher initialized")
        
        # Step 3: Fetch crypto 15-minute markets
        print(f"\n🔍 Step 3: Fetching crypto 15-minute markets...")
        # Fetch more markets when filtering for 15-minute (they're rare)
        fetch_limit = num_markets * 10 if filter_15min else num_markets * 5
        markets = fetcher.get_markets(
            limit=fetch_limit,        # Fetch 10x for 15-minute markets (they're rare)
            order='volume24hr',       # Sort by 24h volume
            active=True
        )
        
        print(f"   Fetched {len(markets)} total markets, filtering for crypto 15-minute markets...")
        
        # Filter for crypto 15-minute markets (BOTH conditions must be true)
        crypto_15min_markets = []
        for market in markets:
            is_crypto = predictor.is_crypto_market(market)
            is_15min = predictor.is_15minute_market(market) if filter_15min else True
            
            # Only add if BOTH crypto AND 15-minute (when filter_15min is True)
            if is_crypto and is_15min:
                crypto_15min_markets.append(market)
                if len(crypto_15min_markets) >= num_markets:
                    break
        
        print(f"✅ Found {len(crypto_15min_markets)} crypto 15-minute markets from {len(markets)} total markets")
        
        if not crypto_15min_markets:
            print("\n⚠️  No crypto 15-minute markets found. Please check:")
            print("   - Internet connection")
            print("   - Polymarket API availability")
            print("   - There may be no active 15-minute crypto markets at this time")
            print("   - Try reducing filtering (set filter_15min=False) or increasing fetch_limit")
            return
        
        # Step 4: Analyze markets
        print(f"\n📊 Step 4: Analyzing {len(crypto_15min_markets)} crypto 15-minute markets...")
        print_separator()
        
        results = []
        for i, market in enumerate(crypto_15min_markets, 1):
            result = analyze_market(predictor, fetcher, market, rank=i)
            if result:
                results.append(result)
            time.sleep(0.3)  # Rate limiting
        
        # Step 5: Summary
        print_separator()
        print("\n📊 SUMMARY")
        print_separator()
        
        if not results:
            print("⚠️  No markets were successfully analyzed")
            return
        
        strong_signals = [r for r in results if r['signal'] in ['STRONG', 'MODERATE'] and r['action'] != 'HOLD']
        buy_yes = [r for r in results if r['action'] == 'BUY_YES']
        buy_no = [r for r in results if r['action'] == 'BUY_NO']
        hold = [r for r in results if r['action'] == 'HOLD']
        
        crypto_count = sum(1 for r in results if r['is_crypto'])
        min15_count = sum(1 for r in results if r['is_15min'])
        
        print(f"\n📈 Total Markets Analyzed: {len(results)}")
        print(f"   Crypto Markets: {crypto_count}")
        print(f"   15-Minute Markets: {min15_count}")
        
        print(f"\n🎯 Signals:")
        print(f"   BUY YES: {len(buy_yes)}")
        print(f"   BUY NO: {len(buy_no)}")
        print(f"   HOLD: {len(hold)}")
        print(f"   Strong/Moderate Signals: {len(strong_signals)}")
        
        if strong_signals:
            print(f"\n💎 TOP OPPORTUNITIES:")
            # Sort by confidence * edge
            strong_signals.sort(key=lambda x: x['confidence'] * x['edge'], reverse=True)
            
            for i, result in enumerate(strong_signals[:5], 1):
                question = result['question'][:50]
                print(f"\n   {i}. {question}...")
                print(f"      Action: {result['action']} | "
                      f"Signal: {result['signal']} | "
                      f"Confidence: {format_price(result['confidence'])}")
                print(f"      Edge: {format_price(result['edge'])} | "
                      f"Position: {format_money(result['kelly_size'])}")
        
        print_separator()
        print("\n⚠️  DISCLAIMER:")
        print("   This software is for educational purposes only.")
        print("   Not financial advice. Prediction markets involve substantial risk.")
        print("   Always do your own research and never invest more than you can afford to lose.")
        print_separator()
        print()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
