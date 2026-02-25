"""
Simple Price Prediction Module - No Training Required!

Uses momentum analysis and CoinGecko data for instant predictions.
This replaces the heavy GRU model with a lightweight, instant approach.
"""

import numpy as np
from typing import Dict, List


def predict_with_momentum(
    current_price: float,
    change_7d: float,
    change_30d: float,
    change_1y: float,
    horizon: int = 7,
    volatility: float = 0.1
) -> Dict:
    """
    Predict future prices using momentum analysis.
    
    Args:
        current_price: Current market price
        change_7d: 7-day percentage change
        change_30d: 30-day percentage change
        change_1y: 1-year percentage change
        horizon: Number of days to predict
        volatility: Market volatility (0-1)
        
    Returns:
        Dict with predictions and confidence intervals
    """
    # Calculate weighted momentum (more weight to recent changes)
    momentum_7d = change_7d * 0.5
    momentum_30d = change_30d * 0.3
    momentum_1y = (change_1y / 52) * 0.2  # Weekly from yearly
    
    combined_momentum = momentum_7d + momentum_30d + momentum_1y
    
    # Generate predictions
    predictions = []
    for i in range(1, horizon + 1):
        # Apply momentum decay for longer predictions
        decay_factor = 0.9 ** (i - 1)
        predicted_change = (combined_momentum / 7) * i * decay_factor
        predicted_price = current_price * (1 + predicted_change / 100)
        predictions.append(predicted_price)
    
    # Calculate confidence intervals based on volatility
    confidence_margin = max(0.05, min(0.15, volatility * 2))
    confidence_lower = [p * (1 - confidence_margin) for p in predictions]
    confidence_upper = [p * (1 + confidence_margin) for p in predictions]
    
    return {
        'predictions': predictions,
        'confidence_lower': confidence_lower,
        'confidence_upper': confidence_upper,
        'current_price': current_price,
        'momentum': combined_momentum
    }


def generate_trading_signals(
    current_price: float,
    predictions: List[float]
) -> Dict:
    """
    Generate trading signals from predictions.
    
    Args:
        current_price: Current market price
        predictions: List of predicted prices
        
    Returns:
        Dict with trading signals
    """
    avg_prediction = sum(predictions) / len(predictions)
    expected_return = ((avg_prediction - current_price) / current_price) * 100
    
    if expected_return > 10:
        signal = "strong_buy"
        confidence = "high"
    elif expected_return > 5:
        signal = "buy"
        confidence = "medium"
    elif expected_return < -10:
        signal = "strong_sell"
        confidence = "high"
    elif expected_return < -5:
        signal = "sell"
        confidence = "medium"
    else:
        signal = "hold"
        confidence = "low"
    
    return {
        'signal': signal,
        'confidence': confidence,
        'expected_return': round(expected_return, 2),
        'short_term_return': round(((predictions[0] - current_price) / current_price) * 100, 2),
        'medium_term_return': round(((predictions[min(3, len(predictions)-1)] - current_price) / current_price) * 100, 2),
        'long_term_return': round(((predictions[-1] - current_price) / current_price) * 100, 2)
    }


def format_predictions_for_analysis(prediction_data: Dict) -> str:
    """
    Format predictions for LLM analysis.
    Supports both single and dual predictor modes.
    
    Args:
        prediction_data: Prediction results
        
    Returns:
        Formatted string
    """
    output = []
    
    # Check if dual predictor mode
    is_dual = prediction_data.get('method') == 'dual_predictor'
    
    if is_dual:
        output.append("🔮 PRICE PREDICTIONS (DUAL PREDICTOR - Combined Analysis):")
        output.append(f"  Current Price: ${prediction_data['current_price']:.2f}")
        output.append(f"\n  📊 COMBINED FINAL PREDICTIONS:")
    else:
        output.append("🔮 PRICE PREDICTIONS (Momentum-Based):")
        output.append(f"  Current Price: ${prediction_data['current_price']:.2f}")
        output.append(f"\n  Predicted Prices:")
    
    for i, price in enumerate(prediction_data['predictions'], 1):
        lower = prediction_data['confidence_lower'][i-1]
        upper = prediction_data['confidence_upper'][i-1]
        output.append(f"    Day {i}: ${price:.2f} (Range: ${lower:.2f} - ${upper:.2f})")
    
    if 'signal' in prediction_data:
        output.append(f"\n  Final Trading Signal: {prediction_data['signal'].upper()}")
        output.append(f"  Confidence: {prediction_data['confidence'].upper()}")
        
        if 'long_term_return' in prediction_data:
            output.append(f"  Expected Return (7-day): {prediction_data['long_term_return']:.2f}%")
    
    # Show individual predictor results in dual mode
    if is_dual and 'momentum_predictor' in prediction_data:
        output.append(f"\n  📊 Momentum Predictor:")
        mom = prediction_data['momentum_predictor']
        output.append(f"    Signal: {mom['signal'].upper()}")
        output.append(f"    Return: {mom['return']:.2f}%")
        
        output.append(f"\n  🤖 RL Predictor (Q-Learning):")
        rl = prediction_data['rl_predictor']
        output.append(f"    Signal: {rl['signal'].upper()}")
        output.append(f"    Confidence: {rl.get('confidence', 'medium').upper()}")
        output.append(f"    Return: {rl['return']:.2f}%")
        
        if 'q_values' in rl and rl['q_values']:
            output.append(f"    Q-Values:")
            for action, q_val in rl['q_values'].items():
                indicator = "👉" if action == rl['signal'] else "  "
                output.append(f"      {indicator} {action:12s}: {q_val:6.3f}")
        
        output.append(f"\n  🎯 Combined Score: {prediction_data.get('combined_score', 0):.2f}")
        output.append(f"     (Range: -2 to +2, where -2=Strong Sell, +2=Strong Buy)")
    
    return "\n".join(output)
