"""Visualization Utilities for Crypto Analysis.

Generate charts and visualizations for technical analysis and predictions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta


def plot_price_with_indicators(
    df: pd.DataFrame,
    analysis: Dict,
    predictions: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """
    Create comprehensive price chart with technical indicators.
    
    Args:
        df: Historical price data
        analysis: Technical analysis results
        predictions: Price predictions (optional)
        save_path: Path to save the chart (optional)
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Crypto Technical Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Price with EMAs and Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['close'], label='Price', color='black', linewidth=2)
    
    # Plot EMAs if available
    if 'ema' in analysis and analysis['ema']['emas']:
        colors = ['blue', 'orange', 'green', 'red']
        for i, (ema_key, ema_value) in enumerate(analysis['ema']['emas'].items()):
            if ema_value and ema_key in ['ema_9', 'ema_21', 'ema_50', 'ema_200']:
                # Calculate EMA for plotting (simplified)
                period = int(ema_key.split('_')[1])
                ema_line = df['close'].ewm(span=period, adjust=False).mean()
                ax1.plot(df['timestamp'], ema_line, label=ema_key.upper(), 
                        color=colors[i % len(colors)], alpha=0.7, linestyle='--')
    
    # Plot Bollinger Bands
    if 'bollinger_bands' in analysis:
        # Calculate Bollinger Bands for full dataset
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        ax1.fill_between(df['timestamp'], upper_band, lower_band, alpha=0.2, color='gray', label='Bollinger Bands')
    
    # Plot predictions if available
    if predictions and 'predictions' in predictions:
        last_date = df['timestamp'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=len(predictions['predictions']))
        
        ax1.plot(future_dates, predictions['predictions'], 
                label='Predictions', color='purple', linewidth=2, linestyle='--', marker='o')
        
        # Confidence intervals
        if 'confidence_lower' in predictions and 'confidence_upper' in predictions:
            ax1.fill_between(future_dates, 
                           predictions['confidence_lower'], 
                           predictions['confidence_upper'],
                           alpha=0.2, color='purple', label='Confidence Interval')
    
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Action with EMAs and Bollinger Bands')
    
    # 2. RSI
    ax2 = axes[1]
    # Calculate RSI for full dataset
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ax2.plot(df['timestamp'], rsi, label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.fill_between(df['timestamp'], 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Relative Strength Index (RSI)')
    
    # 3. Volume
    ax3 = axes[2]
    colors_vol = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
    ax3.bar(df['timestamp'], df['volume'], color=colors_vol, alpha=0.6, label='Volume')
    ax3.plot(df['timestamp'], df['volume'].rolling(window=7).mean(), 
            color='blue', linewidth=2, label='7-day MA')
    ax3.set_ylabel('Volume', fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Trading Volume')
    
    # 4. Support and Resistance
    ax4 = axes[3]
    ax4.plot(df['timestamp'], df['close'], label='Price', color='black', linewidth=2)
    
    if 'support_resistance' in analysis:
        # Plot support levels
        for support in analysis['support_resistance']['support_levels']:
            ax4.axhline(y=support, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        # Plot resistance levels
        for resistance in analysis['support_resistance']['resistance_levels']:
            ax4.axhline(y=resistance, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend entries
        if analysis['support_resistance']['support_levels']:
            ax4.plot([], [], color='green', linestyle='--', label='Support Levels')
        if analysis['support_resistance']['resistance_levels']:
            ax4.plot([], [], color='red', linestyle='--', label='Resistance Levels')
    
    ax4.set_ylabel('Price ($)', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Support and Resistance Levels')
    
    # Rotate x-axis labels
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    return fig


def create_prediction_chart(
    current_price: float,
    predictions: Dict,
    symbol: str,
    save_path: Optional[str] = None
):
    """
    Create a focused chart showing price predictions.
    
    Args:
        current_price: Current market price
        predictions: Prediction results
        symbol: Crypto symbol
        save_path: Path to save chart (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = list(range(len(predictions['predictions']) + 1))
    prices = [current_price] + predictions['predictions']
    
    # Main prediction line
    ax.plot(days, prices, marker='o', linewidth=2, markersize=8, 
           color='blue', label='Predicted Price')
    
    # Confidence interval
    if 'confidence_lower' in predictions and 'confidence_upper' in predictions:
        lower = [current_price] + predictions['confidence_lower']
        upper = [current_price] + predictions['confidence_upper']
        ax.fill_between(days[1:], lower[1:], upper[1:], alpha=0.2, color='blue', 
                       label='Confidence Interval')
    
    # Current price line
    ax.axhline(y=current_price, color='red', linestyle='--', alpha=0.7, 
              label=f'Current Price: ${current_price:.2f}')
    
    ax.set_xlabel('Days from Now', fontweight='bold')
    ax.set_ylabel('Price ($)', fontweight='bold')
    ax.set_title(f'{symbol} Price Prediction - Next {len(predictions["predictions"])} Days', 
                fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Annotate final prediction
    final_price = predictions['predictions'][-1]
    change_pct = ((final_price - current_price) / current_price) * 100
    ax.annotate(f'${final_price:.2f}\n({change_pct:+.1f}%)',
               xy=(len(days)-1, final_price),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction chart saved to: {save_path}")
    
    return fig


# Example usage function
def generate_analysis_charts(symbol: str, save_dir: str = './charts'):
    """
    Generate and save all analysis charts for a cryptocurrency.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC')
        save_dir: Directory to save charts
    """
    import os
    from data_fetcher import get_data_fetcher
    from technical_analysis import TechnicalAnalyzer
    from gru_model import GRUPricePredictor
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Fetch data
    print(f"Fetching data for {symbol}...")
    fetcher = get_data_fetcher()
    df = fetcher.get_historical_data(symbol, days=90, interval='1d')
    current = fetcher.get_current_price(symbol)
    
    # Technical analysis
    print("Running technical analysis...")
    analyzer = TechnicalAnalyzer(df)
    analysis = analyzer.get_complete_analysis()
    
    # Predictions
    print("Generating predictions...")
    predictor = GRUPricePredictor()
    predictor.train(df, epochs=30, verbose=0)
    predictions = predictor.predict_next_prices(df, periods=7)
    
    # Generate charts
    print("Creating charts...")
    plot_price_with_indicators(
        df, analysis, predictions,
        save_path=f"{save_dir}/{symbol}_full_analysis.png"
    )
    
    create_prediction_chart(
        current['price'], predictions, symbol,
        save_path=f"{save_dir}/{symbol}_predictions.png"
    )
    
    print(f"✅ Charts saved to {save_dir}/")


if __name__ == "__main__":
    # Example: Generate charts for Bitcoin
    generate_analysis_charts('BTC')
