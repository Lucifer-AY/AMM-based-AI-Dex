"""Technical Analysis Tools for Crypto Trading Agent.

This module provides all technical indicators and analysis functions:
- Price action (trend detection)
- RSI (Relative Strength Index)
- Volume analysis
- EMA (Exponential Moving Average)
- Support/Resistance levels
- Bollinger Bands
- Liquidity areas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ta
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice


class TechnicalAnalyzer:
    """Comprehensive technical analysis for cryptocurrency price data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with price data.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
    def analyze_price_action(self) -> Dict:
        """
        Analyze price action to determine trend direction.
        
        Returns:
            Dict with trend, strength, and recent price change
        """
        df = self.df
        
        # Calculate percentage change over different periods
        current_price = df['close'].iloc[-1]
        price_1d_ago = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_7d_ago = df['close'].iloc[-7] if len(df) > 7 else df['close'].iloc[0]
        price_30d_ago = df['close'].iloc[-30] if len(df) > 30 else df['close'].iloc[0]
        
        change_1d = ((current_price - price_1d_ago) / price_1d_ago) * 100
        change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
        change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
        
        # Determine overall trend
        if change_7d > 5 and change_30d > 10:
            trend = "strong_uptrend"
            strength = "strong"
        elif change_7d > 2:
            trend = "uptrend"
            strength = "moderate"
        elif change_7d < -5 and change_30d < -10:
            trend = "strong_downtrend"
            strength = "strong"
        elif change_7d < -2:
            trend = "downtrend"
            strength = "moderate"
        else:
            trend = "sideways"
            strength = "weak"
            
        return {
            "trend": trend,
            "strength": strength,
            "change_1d": round(change_1d, 2),
            "change_7d": round(change_7d, 2),
            "change_30d": round(change_30d, 2),
            "current_price": round(current_price, 2)
        }
    
    def calculate_rsi(self, period: int = 14) -> Dict:
        """
        Calculate RSI and determine market zone.
        
        Args:
            period: RSI period (default 14)
            
        Returns:
            Dict with RSI value and zone interpretation
        """
        rsi_indicator = RSIIndicator(close=self.df['close'], window=period)
        rsi_values = rsi_indicator.rsi()
        current_rsi = rsi_values.iloc[-1]
        
        # Determine RSI zone
        if current_rsi >= 70:
            zone = "overbought"
            signal = "sell_signal"
        elif current_rsi >= 60:
            zone = "approaching_overbought"
            signal = "caution"
        elif current_rsi <= 30:
            zone = "oversold"
            signal = "buy_signal"
        elif current_rsi <= 40:
            zone = "approaching_oversold"
            signal = "accumulation"
        else:
            zone = "neutral"
            signal = "hold"
            
        return {
            "rsi": round(current_rsi, 2),
            "zone": zone,
            "signal": signal,
            "period": period
        }
    
    def analyze_volume(self) -> Dict:
        """
        Analyze volume patterns and trends.
        
        Returns:
            Dict with volume analysis
        """
        df = self.df
        
        current_volume = df['volume'].iloc[-1]
        avg_volume_7d = df['volume'].iloc[-7:].mean()
        avg_volume_30d = df['volume'].iloc[-30:].mean() if len(df) >= 30 else df['volume'].mean()
        
        volume_trend = "increasing" if current_volume > avg_volume_7d else "decreasing"
        
        # Volume spike detection
        volume_spike = current_volume > (avg_volume_7d * 1.5)
        
        # Calculate volume ratio safely to avoid division by zero
        volume_ratio = round(current_volume / avg_volume_7d, 2) if avg_volume_7d > 0 else 0
        
        return {
            "current_volume": round(current_volume, 2),
            "avg_volume_7d": round(avg_volume_7d, 2),
            "avg_volume_30d": round(avg_volume_30d, 2),
            "volume_trend": volume_trend,
            "volume_spike": volume_spike,
            "volume_ratio": volume_ratio
        }
    
    def calculate_ema(self, periods: List[int] = [9, 21, 50, 200]) -> Dict:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            periods: List of EMA periods
            
        Returns:
            Dict with EMA values and crossover signals
        """
        current_price = self.df['close'].iloc[-1]
        emas = {}
        
        for period in periods:
            if len(self.df) >= period:
                ema_indicator = EMAIndicator(close=self.df['close'], window=period)
                ema_value = ema_indicator.ema_indicator().iloc[-1]
                emas[f'ema_{period}'] = round(ema_value, 2)
            else:
                emas[f'ema_{period}'] = None
        
        # Check for golden cross / death cross (50/200 EMA)
        signal = "neutral"
        if emas.get('ema_50') and emas.get('ema_200'):
            if emas['ema_50'] > emas['ema_200'] and current_price > emas['ema_50']:
                signal = "bullish_golden_cross"
            elif emas['ema_50'] < emas['ema_200'] and current_price < emas['ema_50']:
                signal = "bearish_death_cross"
        
        # Price position relative to short-term EMA
        if emas.get('ema_9'):
            if current_price > emas['ema_9']:
                short_term_signal = "above_short_ema"
            else:
                short_term_signal = "below_short_ema"
        else:
            short_term_signal = "insufficient_data"
        
        return {
            "emas": emas,
            "signal": signal,
            "short_term_signal": short_term_signal,
            "current_price": round(current_price, 2)
        }
    
    def find_support_resistance(self, lookback: int = 50) -> Dict:
        """
        Identify support and resistance levels using pivot points.
        
        Args:
            lookback: Number of periods to analyze
            
        Returns:
            Dict with support and resistance levels
        """
        df = self.df.tail(lookback)
        
        # Find local maxima (resistance) and minima (support)
        highs = df['high'].values
        lows = df['low'].values
        current_price = df['close'].iloc[-1]
        
        # Simple pivot point calculation
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(df) - 2):
            # Resistance: local maximum
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
            
            # Support: local minimum
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Get nearest levels
        resistance_levels = sorted(set([r for r in resistance_levels if r > current_price]))[:3]
        support_levels = sorted(set([s for s in support_levels if s < current_price]), reverse=True)[:3]
        
        # Calculate distance to nearest levels
        nearest_resistance = resistance_levels[0] if resistance_levels else None
        nearest_support = support_levels[0] if support_levels else None
        
        return {
            "resistance_levels": [round(r, 2) for r in resistance_levels],
            "support_levels": [round(s, 2) for s in support_levels],
            "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None,
            "nearest_support": round(nearest_support, 2) if nearest_support else None,
            "current_price": round(current_price, 2),
            "distance_to_resistance": round(((nearest_resistance - current_price) / current_price * 100), 2) if nearest_resistance else None,
            "distance_to_support": round(((current_price - nearest_support) / current_price * 100), 2) if nearest_support else None
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> Dict:
        """
        Calculate Bollinger Bands.
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dict with Bollinger Bands values and signals
        """
        bb_indicator = BollingerBands(
            close=self.df['close'],
            window=period,
            window_dev=std_dev
        )
        
        current_price = self.df['close'].iloc[-1]
        bb_high = bb_indicator.bollinger_hband().iloc[-1]
        bb_mid = bb_indicator.bollinger_mavg().iloc[-1]
        bb_low = bb_indicator.bollinger_lband().iloc[-1]
        bb_width = bb_indicator.bollinger_wband().iloc[-1]
        
        # Determine position and signal
        if current_price >= bb_high:
            position = "above_upper_band"
            signal = "overbought_sell_signal"
        elif current_price <= bb_low:
            position = "below_lower_band"
            signal = "oversold_buy_signal"
        elif current_price > bb_mid:
            position = "upper_half"
            signal = "bullish"
        else:
            position = "lower_half"
            signal = "bearish"
        
        # Squeeze detection (narrow bands indicate low volatility, potential breakout)
        avg_width = bb_indicator.bollinger_wband().tail(20).mean()
        squeeze = bb_width < (avg_width * 0.8)
        
        return {
            "upper_band": round(bb_high, 2),
            "middle_band": round(bb_mid, 2),
            "lower_band": round(bb_low, 2),
            "band_width": round(bb_width, 4),
            "current_price": round(current_price, 2),
            "position": position,
            "signal": signal,
            "squeeze_detected": squeeze
        }
    
    def identify_liquidity_areas(self, volume_threshold: float = 1.5) -> Dict:
        """
        Identify high liquidity areas based on volume clustering.
        
        Args:
            volume_threshold: Multiplier for average volume to identify high liquidity
            
        Returns:
            Dict with liquidity zones
        """
        df = self.df.tail(100)  # Last 100 periods
        avg_volume = df['volume'].mean()
        
        # Find high volume areas
        high_volume_periods = df[df['volume'] > (avg_volume * volume_threshold)]
        
        if len(high_volume_periods) > 0:
            # Cluster prices where high volume occurred
            liquidity_prices = high_volume_periods['close'].values
            
            # Simple clustering: find price levels with multiple high-volume occurrences
            liquidity_zones = []
            current_price = self.df['close'].iloc[-1]
            
            for price in liquidity_prices:
                # Group prices within 2% range
                similar_found = False
                for zone in liquidity_zones:
                    if abs(price - zone['price']) / zone['price'] < 0.02:
                        zone['strength'] += 1
                        similar_found = True
                        break
                
                if not similar_found:
                    liquidity_zones.append({'price': price, 'strength': 1})
            
            # Sort by strength
            liquidity_zones = sorted(liquidity_zones, key=lambda x: x['strength'], reverse=True)[:5]
            
            # Categorize as support or resistance
            support_liquidity = [z for z in liquidity_zones if z['price'] < current_price]
            resistance_liquidity = [z for z in liquidity_zones if z['price'] > current_price]
            
        else:
            support_liquidity = []
            resistance_liquidity = []
        
        return {
            "high_liquidity_support": [{"price": round(z['price'], 2), "strength": z['strength']} for z in support_liquidity],
            "high_liquidity_resistance": [{"price": round(z['price'], 2), "strength": z['strength']} for z in resistance_liquidity],
            "current_price": round(self.df['close'].iloc[-1], 2),
            "avg_volume": round(avg_volume, 2)
        }
    
    def get_complete_analysis(self) -> Dict:
        """
        Run all technical analysis and return comprehensive results.
        
        Returns:
            Dict with all technical indicators
        """
        return {
            "price_action": self.analyze_price_action(),
            "rsi": self.calculate_rsi(),
            "volume": self.analyze_volume(),
            "ema": self.calculate_ema(),
            "support_resistance": self.find_support_resistance(),
            "bollinger_bands": self.calculate_bollinger_bands(),
            "liquidity_areas": self.identify_liquidity_areas()
        }


def format_analysis_for_llm(analysis: Dict) -> str:
    """
    Format technical analysis results into a readable string for LLM.
    
    Args:
        analysis: Complete technical analysis dict
        
    Returns:
        Formatted string with all analysis results
    """
    output = []
    
    # Price Action
    pa = analysis['price_action']
    output.append(f"📊 PRICE ACTION:")
    output.append(f"  Current Price: ${pa['current_price']}")
    output.append(f"  Trend: {pa['trend'].upper()} ({pa['strength']} strength)")
    output.append(f"  Changes: 1D: {pa['change_1d']}% | 7D: {pa['change_7d']}% | 30D: {pa['change_30d']}%")
    
    # RSI
    rsi = analysis['rsi']
    output.append(f"\n📈 RSI ANALYSIS:")
    output.append(f"  RSI({rsi['period']}): {rsi['rsi']}")
    output.append(f"  Zone: {rsi['zone'].upper()}")
    output.append(f"  Signal: {rsi['signal'].upper()}")
    
    # Volume
    vol = analysis['volume']
    output.append(f"\n📊 VOLUME ANALYSIS:")
    output.append(f"  Current: {vol['current_volume']:,.0f}")
    output.append(f"  7D Average: {vol['avg_volume_7d']:,.0f}")
    output.append(f"  Trend: {vol['volume_trend'].upper()}")
    output.append(f"  Volume Spike: {'YES' if vol['volume_spike'] else 'NO'}")
    
    # EMA
    ema = analysis['ema']
    output.append(f"\n📉 EMA ANALYSIS:")
    for key, value in ema['emas'].items():
        if value:
            output.append(f"  {key.upper()}: ${value}")
    output.append(f"  Signal: {ema['signal'].upper()}")
    output.append(f"  Short-term: {ema['short_term_signal'].upper()}")
    
    # Support/Resistance
    sr = analysis['support_resistance']
    output.append(f"\n🎯 SUPPORT & RESISTANCE:")
    if sr['resistance_levels']:
        output.append(f"  Resistance: {', '.join([f'${r}' for r in sr['resistance_levels']])}")
    if sr['support_levels']:
        output.append(f"  Support: {', '.join([f'${s}' for s in sr['support_levels']])}")
    
    # Bollinger Bands
    bb = analysis['bollinger_bands']
    output.append(f"\n📊 BOLLINGER BANDS:")
    output.append(f"  Upper: ${bb['upper_band']} | Mid: ${bb['middle_band']} | Lower: ${bb['lower_band']}")
    output.append(f"  Position: {bb['position'].upper()}")
    output.append(f"  Signal: {bb['signal'].upper()}")
    output.append(f"  Squeeze: {'YES' if bb['squeeze_detected'] else 'NO'}")
    
    # Liquidity
    liq = analysis['liquidity_areas']
    output.append(f"\n💧 LIQUIDITY AREAS:")
    if liq['high_liquidity_support']:
        support_zones = ', '.join([f"${z['price']} (strength: {z['strength']})" for z in liq['high_liquidity_support']])
        output.append(f"  Support Zones: {support_zones}")
    if liq['high_liquidity_resistance']:
        resistance_zones = ', '.join([f"${z['price']} (strength: {z['strength']})" for z in liq['high_liquidity_resistance']])
        output.append(f"  Resistance Zones: {resistance_zones}")
    
    return "\n".join(output)
