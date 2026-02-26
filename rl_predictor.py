"""Reinforcement Learning-Based Price Predictor.

Simple Q-Learning approach for cryptocurrency trading signals.
No extensive training needed - uses pattern recognition and adaptive learning.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import json
import os
from config import settings


class RLPredictor:
    """
    Q-Learning based predictor for crypto trading.
    
    State space: Price trends, momentum, volatility
    Action space: Strong Buy, Buy, Hold, Sell, Strong Sell
    Reward: Based on predicted vs actual price movements
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize RL predictor with default parameters.
        
        Args:
            learning_rate: How quickly to update Q-values
            discount_factor: Future reward importance
            epsilon: Exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> action -> Q-value
        # We'll use a simplified state representation
        self.q_table = {}
        
        # Actions: 0=strong_sell, 1=sell, 2=hold, 3=buy, 4=strong_buy
        self.actions = ['strong_sell', 'sell', 'hold', 'buy', 'strong_buy']
        self.action_values = [-2, -1, 0, 1, 2]  # Numeric representation
        
    def _get_state_key(self, features: Dict) -> str:
        """
        Convert continuous features to discrete state key.
        
        Args:
            features: Dictionary of market features
            
        Returns:
            State key string
        """
        # Discretize continuous values
        momentum = features.get('momentum_7d', 0)
        volatility = features.get('volatility', 0.05)
        rsi = features.get('rsi', 50)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # Create discrete buckets
        momentum_state = 'up' if momentum > 5 else 'down' if momentum < -5 else 'neutral'
        volatility_state = 'high' if volatility > 0.1 else 'low'
        rsi_state = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'normal'
        volume_state = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.7 else 'normal'
        
        return f"{momentum_state}_{volatility_state}_{rsi_state}_{volume_state}"
    
    def _get_q_value(self, state_key: str, action_idx: int) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state_key: State identifier
            action_idx: Action index (0-4)
            
        Returns:
            Q-value (initialized to 0 if not seen before)
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        return self.q_table[state_key][action_idx]
    
    def predict_signal(self, market_data: Dict, technical_analysis: Dict = None) -> Dict:
        """
        Predict trading signal using Q-learning.
        
        Args:
            market_data: Current market data (price, volume, changes)
            technical_analysis: Technical indicators (RSI, EMA, etc.)
            
        Returns:
            Dictionary with signal, confidence, and Q-values
        """
        # Extract features for state representation
        features = self._extract_features(market_data, technical_analysis)
        state_key = self._get_state_key(features)
        
        # Get Q-values for all actions
        q_values = []
        for i in range(len(self.actions)):
            q_values.append(self._get_q_value(state_key, i))
        
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, len(self.actions))
        else:
            # Exploit: best action based on Q-values
            action_idx = np.argmax(q_values)
        
        # Calculate confidence based on Q-value spread
        q_array = np.array(q_values)
        if np.std(q_array) > 0:
            confidence = min(1.0, abs(q_array[action_idx] - np.mean(q_array)) / np.std(q_array))
        else:
            confidence = 0.5
        
        # Map confidence to categorical
        if confidence > 0.7:
            confidence_level = 'high'
        elif confidence > 0.4:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return {
            'signal': self.actions[action_idx],
            'confidence': confidence_level,
            'confidence_score': round(confidence, 2),
            'q_values': {action: round(q, 3) for action, q in zip(self.actions, q_values)},
            'state': state_key,
            'action_idx': action_idx,
            'action_strength': self.action_values[action_idx],
            'method': 'q_learning'
        }
    
    def predict_prices(self, current_price: float, market_data: Dict, 
                      technical_analysis: Dict, horizon: int = 7) -> Dict:
        """
        Predict future prices using RL signal and momentum.
        
        Args:
            current_price: Current cryptocurrency price
            market_data: Market data dictionary
            technical_analysis: Technical indicators
            horizon: Number of days to predict
            
        Returns:
            Price predictions with confidence intervals
        """
        features = self._extract_features(market_data, technical_analysis)
        signal_data = self.predict_signal(market_data, technical_analysis)
        
        # Use action strength and momentum to predict
        action_strength = signal_data['action_strength']
        momentum = features.get('momentum_7d', 0)
        
        # Combine RL signal with momentum
        combined_trend = (action_strength * 0.6 + (momentum / 10) * 0.4)
        
        # Generate predictions
        predictions = []
        for day in range(1, horizon + 1):
            # Decay factor: predictions become less certain over time
            decay = 0.95 ** (day - 1)
            daily_change = (combined_trend / 7) * day * decay
            predicted_price = current_price * (1 + daily_change / 100)
            predictions.append(predicted_price)
        
        # Calculate confidence intervals based on volatility
        volatility = features.get('volatility', 0.05)
        confidence_margin = max(0.05, min(0.20, volatility * 2))
        
        confidence_lower = [p * (1 - confidence_margin) for p in predictions]
        confidence_upper = [p * (1 + confidence_margin) for p in predictions]
        
        return {
            'predictions': predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'current_price': current_price,
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'confidence_score': signal_data.get('confidence_score', 0.5),
            'q_values': signal_data['q_values'],
            'state': signal_data.get('state'),
            'action_idx': signal_data.get('action_idx'),
            'action_strength': signal_data.get('action_strength', 0),
            'expected_return': ((predictions[-1] - current_price) / current_price) * 100,
            'method': 'rl_q_learning'
        }
    
    def _extract_features(self, market_data: Dict, technical_analysis: Dict = None) -> Dict:
        """
        Extract features from market data and technical analysis.
        
        Args:
            market_data: Market data (price, volume, etc.)
            technical_analysis: Technical indicators
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Price features
        features['current_price'] = market_data.get('price', 0)
        features['change_24h'] = market_data.get('change_24h', 0)
        features['momentum_7d'] = market_data.get('price_change_percentage_7d', 
                                                   market_data.get('change_24h', 0) * 3)
        
        # Volume features
        features['volume_24h'] = market_data.get('volume_24h', 0)
        features['volume_ratio'] = market_data.get('volume_ratio', 1.0)
        
        # Volatility (from 24h high/low)
        price = features['current_price']
        if price > 0:
            high = market_data.get('high_24h', price)
            low = market_data.get('low_24h', price)
            features['volatility'] = abs(high - low) / price
        else:
            features['volatility'] = 0.05
        
        # Technical indicators (if available)
        if technical_analysis:
            rsi_data = technical_analysis.get('rsi', {})
            features['rsi'] = rsi_data.get('current_rsi', rsi_data.get('rsi', 50))
            
            # EMA trend
            ema = technical_analysis.get('ema', {})
            if ema:
                ema_vals = ema.get('ema_values', ema.get('emas', {}))
                if 'ema_9' in ema_vals and 'ema_21' in ema_vals:
                    # Bullish if short EMA > long EMA
                    features['ema_trend'] = 1 if ema_vals['ema_9'] > ema_vals['ema_21'] else -1
            
            # Bollinger position
            bb = technical_analysis.get('bollinger_bands', {})
            if bb and 'position' in bb:
                features['bb_position'] = bb['position']
        
        return features

    def save_model(self, file_path: str = "models/rl_q_table.json") -> str:
        """
        Save Q-table and hyperparameters to disk.

        Args:
            file_path: Target JSON file path

        Returns:
            Saved file path
        """
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "metadata": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "actions": self.actions,
                "action_values": self.action_values,
            },
            "q_table": {
                state: [float(value) for value in values.tolist()]
                for state, values in self.q_table.items()
            },
        }

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return file_path

    def load_model(self, file_path: str = "models/rl_q_table.json") -> bool:
        """
        Load Q-table and hyperparameters from disk.

        Args:
            file_path: Source JSON file path

        Returns:
            True if model was loaded, False if file does not exist
        """
        if not os.path.exists(file_path):
            return False

        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata", {})
        self.learning_rate = float(metadata.get("learning_rate", self.learning_rate))
        self.discount_factor = float(metadata.get("discount_factor", self.discount_factor))
        self.epsilon = float(metadata.get("epsilon", self.epsilon))

        self.actions = metadata.get("actions", self.actions)
        self.action_values = metadata.get("action_values", self.action_values)

        raw_q_table = payload.get("q_table", {})
        self.q_table = {
            state: np.array(values, dtype=float)
            for state, values in raw_q_table.items()
        }

        return True
    
    def update_from_outcome(self, state_key: str, action_idx: int, 
                           reward: float, next_state_key: str):
        """
        Update Q-values based on actual outcome (for online learning).
        
        Args:
            state_key: Previous state
            action_idx: Action taken
            reward: Reward received
            next_state_key: Resulting state
        """
        # Q-learning update rule
        current_q = self._get_q_value(state_key, action_idx)
        
        # Max Q-value for next state
        next_q_values = [self._get_q_value(next_state_key, i) 
                        for i in range(len(self.actions))]
        max_next_q = max(next_q_values)
        
        # Update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        self.q_table[state_key][action_idx] = new_q


def format_rl_predictions(prediction_data: Dict) -> str:
    """
    Format RL predictions for display.
    
    Args:
        prediction_data: RL prediction results
        
    Returns:
        Formatted string
    """
    output = []
    
    output.append("🤖 RL-BASED PREDICTIONS (Q-Learning):")
    output.append(f"  Signal: {prediction_data['signal'].upper()}")
    output.append(f"  Confidence: {prediction_data['confidence'].upper()} ({prediction_data.get('confidence_score', 0):.2f})")
    output.append(f"  Expected Return: {prediction_data['expected_return']:.2f}%")
    
    output.append(f"\n  Q-Values (learned preferences):")
    for action, q_val in prediction_data.get('q_values', {}).items():
        indicator = "👉" if action == prediction_data['signal'] else "  "
        output.append(f"    {indicator} {action:12s}: {q_val:6.3f}")
    
    if 'predictions' in prediction_data:
        output.append(f"\n  Price Predictions:")
        predictions = prediction_data['predictions']
        for i, (pred, lower, upper) in enumerate(zip(
            predictions,
            prediction_data['confidence_lower'],
            prediction_data['confidence_upper']
        ), 1):
            output.append(f"    Day {i}: ${pred:,.2f} (range: ${lower:,.2f} - ${upper:,.2f})")
    
    return "\n".join(output)


# Create global predictor instance
_rl_predictor = None

def get_rl_predictor() -> RLPredictor:
    """Get or create RL predictor instance."""
    global _rl_predictor
    if _rl_predictor is None:
        _rl_predictor = RLPredictor(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1  # 10% exploration
        )
        model_loaded = _rl_predictor.load_model(settings.rl_model_path)
        if model_loaded:
            print(f"✅ Loaded RL Q-table from {settings.rl_model_path}")
    return _rl_predictor
