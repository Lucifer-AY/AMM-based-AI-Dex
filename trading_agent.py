"""LangGraph-based Crypto Trading Agent Workflow.

This module implements the complete trading agent using LangGraph with:
- State management
- Multi-step analysis workflow
- Integration with Grok LLM
- LangSmith tracing
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
import operator
from datetime import datetime
import json

from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer, format_analysis_for_llm
from simple_predictor import format_predictions_for_analysis as format_predictions_for_llm
from rl_predictor import get_rl_predictor, format_rl_predictions
from web_search import get_web_search
from config import settings
from mcp_tools import get_mcp_toolbox


# ===== STATE DEFINITION =====

class TradingAgentState(TypedDict):
    """State for the crypto trading agent workflow."""
    
    # User input
    messages: Annotated[List, operator.add]
    crypto_symbol: str
    investment_amount: Optional[float]
    risk_tolerance: str  # low, medium, high
    investment_horizon: int  # days
    
    # Analysis results
    current_price_data: Optional[dict]
    technical_analysis: Optional[dict]
    price_predictions: Optional[dict]
    market_sentiment: Optional[dict]
    
    # MCP Tool enhancement
    mcp_analysis: Optional[dict]
    tool_calls_made: Annotated[List[str], operator.add]
    
    # Final recommendations
    trading_recommendation: Optional[dict]
    final_report: Optional[str]
    
    # Workflow control
    error: Optional[str]
    step_completed: Annotated[List[str], operator.add]


# ===== LLM INITIALIZATION =====

def get_llm(temperature: float = None, enable_tools: bool = False):
    """Initialize Groq LLM (Fast and Free!) with optional MCP tools."""
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=temperature or settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )
    
    # Bind MCP tools if enabled
    if enable_tools and settings.enable_mcp_tools:
        mcp_toolbox = get_mcp_toolbox(
            toolbox_url=settings.mcp_toolbox_url if settings.mcp_toolbox_url else None
        )
        tool_schemas = mcp_toolbox.get_tool_schemas()
        if tool_schemas:
            llm = llm.bind_tools(tool_schemas)
            print(f"🔧 MCP Toolbox enabled with {len(tool_schemas)} tools")
    
    return llm


# ===== WORKFLOW NODES =====

def extract_crypto_info(state: TradingAgentState) -> TradingAgentState:
    """
    Extract cryptocurrency symbol and investment parameters from user message.
    """
    print("📝 Step 1: Extracting crypto information...")
    
    messages = state['messages']
    last_message = messages[-1].content if messages else ""
    
    # Use LLM to extract information
    llm = get_llm(temperature=0)
    
    extraction_prompt = f"""
    Extract the following information from the user's message:
    1. Cryptocurrency symbol (e.g., BTC, ETH, ADA)
       - If user mentions "Bitcoin" or "BIT coin", use BTC
       - If user mentions "Ethereum", use ETH  
       - If user mentions "Solana", use SOL
       - Convert full cryptocurrency names to their symbols
    2. Investment amount (if mentioned, otherwise return null)
    3. Risk tolerance (low/medium/high, default to medium if not mentioned)
    4. Investment horizon in days (if mentioned, otherwise default to 7)
    
    User message: {last_message}
    
    Respond in this exact format:
    SYMBOL: <symbol>
    AMOUNT: <amount or null>
    RISK: <low/medium/high>
    HORIZON: <days>
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    response_text = response.content
    
    # Parse response
    symbol = None
    amount = state.get('investment_amount')
    risk = state.get('risk_tolerance', settings.default_risk_tolerance)
    horizon = state.get('investment_horizon', settings.default_investment_horizon)
    
    for line in response_text.split('\n'):
        if 'SYMBOL:' in line:
            symbol = line.split('SYMBOL:')[1].strip().upper()
            # Normalize common name variations to symbols
            symbol_map = {
                'BITCOIN': 'BTC', 'BIT': 'BTC', 'BITC': 'BTC',
                'ETHEREUM': 'ETH', 'ETHER': 'ETH',
                'SOLANA': 'SOL',
                'CARDANO': 'ADA',
                'RIPPLE': 'XRP',
                'DOGECOIN': 'DOGE', 'DOGE COIN': 'DOGE',
                'POLKADOT': 'DOT',
                'POLYGON': 'MATIC',
                'AVALANCHE': 'AVAX'
            }
            symbol = symbol_map.get(symbol, symbol)
        elif 'AMOUNT:' in line:
            amount_str = line.split('AMOUNT:')[1].strip()
            if amount_str.lower() != 'null':
                try:
                    amount = float(amount_str.replace('$', '').replace(',', ''))
                except:
                    pass
        elif 'RISK:' in line:
            risk = line.split('RISK:')[1].strip().lower()
        elif 'HORIZON:' in line:
            try:
                horizon = int(line.split('HORIZON:')[1].strip().split()[0])
            except:
                pass
    
    if not symbol:
        # Default to BTC if not specified
        symbol = 'BTC'
    
    return {
        **state,
        'crypto_symbol': symbol,
        'investment_amount': amount,
        'risk_tolerance': risk,
        'investment_horizon': horizon,
        'step_completed': ['extract_info']
    }


def fetch_current_data(state: TradingAgentState) -> TradingAgentState:
    """
    Fetch current price and comprehensive market data from CoinGecko.
    """
    print(f"💰 Step 2: Fetching current data for {state['crypto_symbol']}...")
    
    try:
        fetcher = get_data_fetcher()
        web_search = get_web_search()
        
        # Get comprehensive coin details (includes all indicators)
        coin_data = fetcher.get_coin_details(state['crypto_symbol'])
        
        # Get fear & greed index
        fear_greed = fetcher.get_fear_greed_index()
        
        # Get web search sentiment and news (if Tavily is configured)
        web_sentiment = web_search.search_crypto_news(state['crypto_symbol']) if web_search.available else None
        
        market_sentiment = {
            'fear_greed_index': fear_greed.get('index', 50),
            'sentiment': fear_greed.get('classification', 'Neutral'),
            'timestamp': fear_greed.get('timestamp', ''),
            'market_cap_rank': coin_data.get('market_cap_rank', 0),
            'ath_distance': coin_data.get('ath_change_percentage', 0),
            'atl_distance': coin_data.get('atl_change_percentage', 0)
        }
        
        # Add web search sentiment if available
        if web_sentiment:
            market_sentiment['web_sentiment'] = web_sentiment.get('sentiment', 'neutral')
            market_sentiment['news_articles'] = len(web_sentiment.get('articles', []))
            market_sentiment['web_summary'] = web_sentiment.get('summary', '')
        
        # Add current price data - use 'price' key from coin_data
        current_price_data = {
            'symbol': coin_data.get('symbol', state['crypto_symbol']),
            'price': coin_data.get('price', 0),
            'change_24h': coin_data.get('price_change_percentage_24h', 0),
            'volume_24h': coin_data.get('volume_24h', 0),
            'market_cap': coin_data.get('market_cap', 0),
            'high_24h': coin_data.get('high_24h', 0),
            'low_24h': coin_data.get('low_24h', 0)
        }
        
        return {
            **state,
            'current_price_data': current_price_data,
            'market_sentiment': market_sentiment,
            'step_completed': ['fetch_data']
        }
        
    except Exception as e:
        print(f"⚠️  Error in fetch_current_data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return with fallback data on error
        fallback_price_data = {
            'symbol': state['crypto_symbol'],
            'price': 0,
            'change_24h': 0,
            'volume_24h': 0,
            'market_cap': 0,
            'high_24h': 0,
            'low_24h': 0
        }
        fallback_sentiment = {
            'fear_greed_index': 50,
            'sentiment': 'neutral',
            'market_cap_rank': 0,
            'ath_distance': 0,
            'atl_distance': 0
        }
        return {
            **state,
            'current_price_data': fallback_price_data,
            'market_sentiment': fallback_sentiment,
            'error': f"Error fetching data: {str(e)}",
            'step_completed': ['fetch_data_error']
        }


def perform_technical_analysis(state: TradingAgentState) -> TradingAgentState:
    """
    Perform comprehensive technical analysis.
    """
    print(f"📊 Step 3: Performing technical analysis...")
    
    try:
        fetcher = get_data_fetcher()
        
        # Fetch historical data (90 days)
        historical_data = fetcher.get_historical_data(
            state['crypto_symbol'],
            days=90,
            interval='1d'
        )
        
        if historical_data.empty:
            raise ValueError("No historical data available")
        
        # Perform technical analysis
        analyzer = TechnicalAnalyzer(historical_data)
        analysis = analyzer.get_complete_analysis()
        
        return {
            **state,
            'technical_analysis': analysis,
            'step_completed': ['technical_analysis']
        }
        
    except Exception as e:
        # Return empty analysis on error - generate_recommendation will handle it
        return {
            **state,
            'technical_analysis': None,
            'error': f"Error in technical analysis: {str(e)}",
            'step_completed': ['technical_analysis_error']
        }


def execute_mcp_tools(state: TradingAgentState) -> TradingAgentState:
    """
    Execute MCP tools for enhanced market analysis.
    Provides deep insights using advanced analytical tools.
    """
    print(f"🔧 Step 4: Running MCP Toolbox for enhanced analysis...")
    
    if not settings.enable_mcp_tools:
        print("  ⏭️  MCP Tools disabled, skipping...")
        return {
            **state,
            'mcp_analysis': None,
            'tool_calls_made': [],
            'step_completed': ['mcp_tools_skipped']
        }
    
    try:
        mcp_toolbox = get_mcp_toolbox()
        symbol = state['crypto_symbol']
        mcp_results = {}
        tools_used = []
        
        # Execute key analysis tools
        print(f"  📊 Calculating risk metrics for {symbol}...")
        risk_metrics = mcp_toolbox.execute_tool("calculate_risk_metrics", {"symbol": symbol, "days": 90})
        if risk_metrics.get("success"):
            mcp_results["risk_metrics"] = risk_metrics["result"]
            tools_used.append("calculate_risk_metrics")
        
        print(f"  📈 Identifying support/resistance levels...")
        support_resistance = mcp_toolbox.execute_tool("get_support_resistance_levels", {"symbol": symbol, "days": 30})
        if support_resistance.get("success"):
            mcp_results["support_resistance"] = support_resistance["result"]
            tools_used.append("get_support_resistance_levels")
        
        print(f"  📉 Analyzing volume profile...")
        volume_analysis = mcp_toolbox.execute_tool("analyze_volume_profile", {"symbol": symbol, "days": 30})
        if volume_analysis.get("success"):
            mcp_results["volume_profile"] = volume_analysis["result"]
            tools_used.append("analyze_volume_profile")
        
        print(f"  🔗 Checking BTC correlation...")
        correlation = mcp_toolbox.execute_tool("get_correlation_analysis", {"symbol": symbol, "days": 30})
        if correlation.get("success"):
            mcp_results["btc_correlation"] = correlation["result"]
            tools_used.append("get_correlation_analysis")
        
        print(f"  📰 Searching latest news and sentiment...")
        news = mcp_toolbox.execute_tool("search_latest_news", {"symbol": symbol})
        if news.get("success"):
            mcp_results["news_sentiment"] = news["result"]
            tools_used.append("search_latest_news")
        
        print(f"  ✅ MCP Analysis complete: {len(tools_used)} tools executed")
        
        return {
            **state,
            'mcp_analysis': mcp_results,
            'tool_calls_made': tools_used,
            'step_completed': ['mcp_tools_executed']
        }
        
    except Exception as e:
        print(f"  ⚠️  Error in MCP tools execution: {str(e)}")
        return {
            **state,
            'mcp_analysis': None,
            'tool_calls_made': [],
            'step_completed': ['mcp_tools_error']
        }


def predict_prices(state: TradingAgentState) -> TradingAgentState:
    """
    Predict future prices using DUAL predictors:
    1. Momentum-based predictor (simple & fast)
    2. RL Q-Learning predictor (adaptive & pattern-based)
    
    Combines both for final decision.
    """
    print(f"🔮 Step 5: Predicting future prices (Dual Predictor Mode)...")
    
    try:
        fetcher = get_data_fetcher()
        rl_predictor = get_rl_predictor()
        
        # Get comprehensive coin data with historical changes
        coin_details = fetcher.get_coin_details(state['crypto_symbol'])
        
        if not coin_details or 'price' not in coin_details:
            raise ValueError(f"Invalid coin data received for {state['crypto_symbol']}")
        
        current_price = coin_details['price']
        
        # ===== PREDICTOR 1: MOMENTUM-BASED =====
        print("  📊 Running Momentum Predictor...")
        change_7d = coin_details.get('price_change_percentage_7d', 0)
        change_30d = coin_details.get('price_change_percentage_30d', 0)
        change_1y = coin_details.get('price_change_percentage_1y', 0)
        
        # Calculate weighted momentum
        momentum_7d = change_7d * 0.5
        momentum_30d = change_30d * 0.3  
        momentum_1y = (change_1y / 52) * 0.2
        combined_momentum = momentum_7d + momentum_30d + momentum_1y
        
        # Generate momentum-based predictions
        momentum_predictions = []
        for i in range(1, state['investment_horizon'] + 1):
            decay_factor = 0.9 ** (i - 1)
            predicted_change = (combined_momentum / 7) * i * decay_factor
            predicted_price = current_price * (1 + predicted_change / 100)
            momentum_predictions.append(predicted_price)
        
        momentum_return = ((momentum_predictions[-1] - current_price) / current_price) * 100
        
        # Momentum signal
        if momentum_return > 10:
            momentum_signal = "strong_buy"
        elif momentum_return > 5:
            momentum_signal = "buy"
        elif momentum_return < -10:
            momentum_signal = "strong_sell"
        elif momentum_return < -5:
            momentum_signal = "sell"
        else:
            momentum_signal = "hold"
        
        # ===== PREDICTOR 2: RL Q-LEARNING =====
        print("  🤖 Running RL Predictor (Q-Learning)...")
        
        # Prepare market data for RL
        market_data = {
            'price': current_price,
            'change_24h': coin_details.get('price_change_percentage_24h', 0),
            'price_change_percentage_7d': change_7d,
            'volume_24h': coin_details.get('total_volume', 0),
            'high_24h': coin_details.get('high_24h', current_price),
            'low_24h': coin_details.get('low_24h', current_price)
        }
        
        # Get technical analysis if available
        technical_analysis = state.get('technical_analysis')
        
        # RL predictions
        rl_result = rl_predictor.predict_prices(
            current_price=current_price,
            market_data=market_data,
            technical_analysis=technical_analysis,
            horizon=state['investment_horizon']
        )
        
        # ===== COMBINE BOTH PREDICTORS =====
        print("  🔄 Combining predictions...")
        
        # Weighted average: 50% momentum, 50% RL
        combined_predictions = []
        for mom_pred, rl_pred in zip(momentum_predictions, rl_result['predictions']):
            combined = (mom_pred * 0.5) + (rl_pred * 0.5)
            combined_predictions.append(combined)
        
        # Calculate confidence intervals based on volatility
        volatility = abs(coin_details['high_24h'] - coin_details['low_24h']) / current_price
        confidence_margin = max(0.05, min(0.15, volatility * 2))
        
        confidence_lower = [p * (1 - confidence_margin) for p in combined_predictions]
        confidence_upper = [p * (1 + confidence_margin) for p in combined_predictions]
        
        # Combined signal logic
        # Convert signals to numeric scores
        signal_scores = {
            'strong_sell': -2, 'sell': -1, 'hold': 0, 
            'buy': 1, 'strong_buy': 2
        }
        
        momentum_score = signal_scores.get(momentum_signal, 0)
        rl_score = signal_scores.get(rl_result['signal'], 0)
        combined_score = (momentum_score + rl_score) / 2
        
        # Determine final signal
        if combined_score >= 1.5:
            final_signal = "strong_buy"
            confidence = "high"
        elif combined_score >= 0.5:
            final_signal = "buy"
            confidence = "medium"
        elif combined_score <= -1.5:
            final_signal = "strong_sell"
            confidence = "high"
        elif combined_score <= -0.5:
            final_signal = "sell"
            confidence = "medium"
        else:
            final_signal = "hold"
            confidence = "low"
        
        # Calculate returns
        avg_prediction = sum(combined_predictions) / len(combined_predictions)
        expected_return = ((avg_prediction - current_price) / current_price) * 100
        
        prediction_data = {
            # Combined predictions
            'predictions': combined_predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'periods': len(combined_predictions),
            'current_price': current_price,
            'method': 'dual_predictor',
            'signal': final_signal,
            'confidence': confidence,
            'short_term_return': ((combined_predictions[0] - current_price) / current_price) * 100,
            'medium_term_return': ((combined_predictions[min(3, len(combined_predictions)-1)] - current_price) / current_price) * 100,
            'long_term_return': ((combined_predictions[-1] - current_price) / current_price) * 100,
            'predicted_prices': {
                'day_1': round(combined_predictions[0], 2),
                'day_3': round(combined_predictions[min(3, len(combined_predictions)-1)], 2),
                'day_7': round(combined_predictions[-1], 2)
            },
            # Individual predictor results
            'momentum_predictor': {
                'signal': momentum_signal,
                'predictions': momentum_predictions,
                'return': momentum_return
            },
            'rl_predictor': {
                'signal': rl_result['signal'],
                'predictions': rl_result['predictions'],
                'return': rl_result['expected_return'],
                'q_values': rl_result.get('q_values', {}),
                'confidence': rl_result.get('confidence', 'medium')
            },
            'combined_score': combined_score
        }
        
        return {
            **state,
            'price_predictions': prediction_data,
            'step_completed': ['price_prediction']
        }
        
    except Exception as e:
        print(f"Error in price prediction: {str(e)}")
        # Create fallback prediction
        current_price = state['current_price_data']['price']
        prediction_data = {
            'predictions': [current_price] * state['investment_horizon'],
            'confidence_lower': [current_price * 0.95] * state['investment_horizon'],
            'confidence_upper': [current_price * 1.05] * state['investment_horizon'],
            'periods': state['investment_horizon'],
            'current_price': current_price,
            'method': 'fallback',
            'signal': 'hold',
            'confidence': 'low',
            'short_term_return': 0,
            'medium_term_return': 0,
            'long_term_return': 0
        }
        
        return {
            **state,
            'price_predictions': prediction_data,
            'step_completed': ['price_prediction_fallback']
        }


def generate_recommendation(state: TradingAgentState) -> TradingAgentState:
    """
    Generate final trading recommendation using all analysis including MCP insights.
    """
    print(f"💡 Step 6: Generating AI-powered trading recommendation...")
    
    try:
        llm = get_llm(temperature=0.3)
        
        # Add safe access to state data with defaults
        current_price = state.get('current_price_data', {}).get('price', 0)
        if current_price == 0:
            # Use fallback analysis when price data unavailable
            error_msg = "Unable to fetch current price data. Please check your internet connection and try again."
            return {
                **state,
                'trading_recommendation': {
                    'recommendation': error_msg,
                    'symbol': state.get('crypto_symbol', 'Unknown'),
                    'error': True
                },
                'final_report': error_msg,
                'error': 'price_data_unavailable',
                'step_completed': ['generate_recommendation_no_data']
            }
        
        # Get technical analysis or use fallback
        technical_analysis = state.get('technical_analysis')
        if technical_analysis:
            tech_analysis_text = format_analysis_for_llm(technical_analysis)
        else:
            tech_analysis_text = "Technical analysis unavailable"
        
        # Get price predictions or use fallback
        price_predictions = state.get('price_predictions')
        if price_predictions:
            predictions_text = format_predictions_for_llm(price_predictions)
        else:
            predictions_text = "Price predictions unavailable"
        
        # Get MCP analysis if available
        mcp_analysis = state.get('mcp_analysis')
        mcp_text = ""
        if mcp_analysis:
            mcp_text = "\n## MCP TOOLBOX ENHANCED ANALYSIS\n"
            
            if 'risk_metrics' in mcp_analysis:
                rm = mcp_analysis['risk_metrics']
                if 'error' not in rm:
                    mcp_text += f"""\n### Risk Assessment
- Sharpe Ratio: {rm.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {rm.get('max_drawdown_pct', 0):.2f}%
- Value at Risk (95%): {rm.get('value_at_risk_95_pct', 0):.2f}%
- Volatility: {rm.get('volatility_pct', 0):.2f}%
- Risk Rating: {rm.get('risk_rating', 'unknown').upper()}
"""
            
            if 'support_resistance' in mcp_analysis:
                sr = mcp_analysis['support_resistance']
                if 'error' not in sr:
                    mcp_text += f"""\n### Key Price Levels
- Nearest Resistance: ${sr.get('nearest_resistance', 'N/A')}
- Nearest Support: ${sr.get('nearest_support', 'N/A')}
- Resistance Levels: {', '.join([f'${r:.2f}' for r in sr.get('resistance_levels', [])])}
- Support Levels: {', '.join([f'${s:.2f}' for s in sr.get('support_levels', [])])}
"""
            
            if 'volume_profile' in mcp_analysis:
                vp = mcp_analysis['volume_profile']
                if 'error' not in vp:
                    mcp_text += f"""\n### Volume Analysis
- Average Volume: {vp.get('average_volume', 0):,.0f}
- Current Volume: {vp.get('current_volume', 0):,.0f}
- Volume Trend: {vp.get('volume_trend', 'unknown').upper()}
- Volume Change: {vp.get('volume_change_pct', 0):.2f}%
- Volume Spikes (30d): {vp.get('volume_spikes_count', 0)}
"""
            
            if 'btc_correlation' in mcp_analysis:
                bc = mcp_analysis['btc_correlation']
                if 'error' not in bc:
                    mcp_text += f"""\n### Market Correlation
- BTC Correlation: {bc.get('correlation_with_btc', 0):.2f}
- {bc.get('interpretation', '')}
"""
            
            if 'news_sentiment' in mcp_analysis:
                ns = mcp_analysis['news_sentiment']
                if ns.get('news_available'):
                    mcp_text += f"""\n### News Sentiment
- Sentiment: {ns.get('sentiment', 'neutral').upper()}
- Articles Analyzed: {ns.get('article_count', 0)}
- Summary: {ns.get('summary', 'N/A')}
"""
        
        # Compile all analysis data
        analysis_summary = f"""
# CRYPTO TRADING ANALYSIS REPORT
**Symbol:** {state['crypto_symbol']}
**Current Price:** ${current_price}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## MARKET SENTIMENT
- Fear & Greed Index: {state['market_sentiment'].get('fear_greed_index', 'N/A')}/100 ({state['market_sentiment'].get('sentiment', 'Unknown')})
- 24h Change: {state['current_price_data'].get('change_24h', 0):.2f}%
- 24h Volume: ${state['current_price_data'].get('volume_24h', 0):,.0f}

## TECHNICAL ANALYSIS
{tech_analysis_text}

## PRICE PREDICTIONS
{predictions_text}
{mcp_text}

## MARKET CONTEXT (CoinGecko Data)
- Market Cap Rank: #{state['market_sentiment'].get('market_cap_rank', 'N/A')}
- Distance from ATH: {state['market_sentiment'].get('ath_distance', 0):.2f}%
- Distance from ATL: {state['market_sentiment'].get('atl_distance', 0):.2f}%
- 24h High/Low: ${state['current_price_data'].get('high_24h', 0):.2f} / ${state['current_price_data'].get('low_24h', 0):.2f}

## INVESTMENT PARAMETERS
- Investment Amount: ${state['investment_amount'] if state['investment_amount'] else 'Not specified'}
- Risk Tolerance: {state['risk_tolerance'].upper()}
- Investment Horizon: {state['investment_horizon']} days
"""
        
        recommendation_prompt = f"""
Based on the comprehensive analysis above (including MCP Toolbox insights), provide a detailed investment recommendation.
Consider:
1. All technical indicators (trend, RSI, volume, EMA, support/resistance, Bollinger Bands, liquidity)
2. Price predictions and confidence intervals
3. Market sentiment (Fear & Greed Index)
4. MCP Analysis (risk metrics, support/resistance, volume profile, BTC correlation, news sentiment)
5. User's risk tolerance: {state['risk_tolerance']}
6. Investment horizon: {state['investment_horizon']} days

Provide a structured recommendation with:
- BUY/SELL/HOLD decision with clear reasoning
- OPTIMAL ENTRY PRICE (or price range)
- POSITION SIZING (percentage of total amount to invest)
- ENTRY STRATEGY (all at once vs. DCA)
- EXIT STRATEGY (target prices and stop-loss)
- RISK ASSESSMENT
- Timeline for investment

Be specific with numbers and actionable advice.

Analysis:
{analysis_summary}
"""
        
        response = llm.invoke([
            SystemMessage(content="You are an expert cryptocurrency trading advisor with deep knowledge of technical analysis and risk management."),
            HumanMessage(content=recommendation_prompt)
        ])
        
        recommendation = response.content
        
        # Create structured recommendation
        trading_rec = {
            'recommendation': recommendation,
            'symbol': state['crypto_symbol'],
            'current_price': current_price,
            'analysis_date': datetime.now().isoformat(),
            'risk_level': state['risk_tolerance'],
            'horizon': state['investment_horizon']
        }
        
        return {
            **state,
            'trading_recommendation': trading_rec,
            'final_report': recommendation,
            'step_completed': ['generate_recommendation']
        }
    
    except Exception as e:
        # Return with error message if recommendation generation fails
        error_recommendation = f"""
Unable to generate detailed recommendation due to: {str(e)}

However, based on available data:
- Symbol: {state.get('crypto_symbol', 'Unknown')}
- Current analysis suggests exercising caution until full data is available.
- Consider waiting for complete market data before making investment decisions.
"""
        return {
            **state,
            'trading_recommendation': {
                'recommendation': error_recommendation,
                'symbol': state.get('crypto_symbol', 'Unknown'),
                'current_price': state.get('current_price_data', {}).get('price', 0),
                'analysis_date': datetime.now().isoformat(),
                'risk_level': state.get('risk_tolerance', 'moderate'),
                'horizon': state.get('investment_horizon', 7),
                'error': True
            },
            'final_report': error_recommendation,
            'error': f"Error in recommendation generation: {str(e)}",
            'step_completed': ['generate_recommendation_error']
        }


def format_final_output(state: TradingAgentState) -> TradingAgentState:
    """
    Format and present final output to user.
    """
    print(f"✅ Step 7: Formatting final output...")
    
    if state.get('error'):
        final_message = f"❌ Error: {state['error']}"
    else:
        final_message = f"""
# 🎯 CRYPTO INVESTMENT RECOMMENDATION

**{state['crypto_symbol']}** - ${state['current_price_data']['price']:.2f}

{state['final_report']}

---

📊 **Analysis Powered by:**
- 🚀 Groq AI (Lightning-fast LLM)
- � MCP Toolbox ({len(state.get('tool_calls_made', []))} advanced tools)
- 📈 CoinGecko API (Real-time data & indicators)
- 🔮 Dual Predictors (Momentum + RL Q-Learning)
- 📊 Technical Indicators (RSI, EMA, Bollinger, Volume, S/R, Liquidity)
- 🧠 LangGraph Multi-Agent Workflow
- 📉 Fear & Greed Index + Market Sentiment

⚡ **Analysis completed in seconds!**

⚠️ **Disclaimer:** This is an AI-generated analysis for educational purposes only. 
Not financial advice. Always do your own research and consult with financial advisors.
"""
    
    return {
        **state,
        'messages': [AIMessage(content=final_message)],
        'step_completed': ['format_output']
    }


# ===== WORKFLOW CONSTRUCTION =====

def create_trading_agent():
    """Create and compile the LangGraph trading agent workflow."""
    
    # Initialize graph
    workflow = StateGraph(TradingAgentState)
    
    # Add nodes
    workflow.add_node("extract_info", extract_crypto_info)
    workflow.add_node("fetch_data", fetch_current_data)
    workflow.add_node("technical_analysis", perform_technical_analysis)
    workflow.add_node("mcp_tools", execute_mcp_tools)
    workflow.add_node("predict_prices", predict_prices)
    workflow.add_node("generate_recommendation", generate_recommendation)
    workflow.add_node("format_output", format_final_output)
    
    # Define edges (workflow sequence)
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "fetch_data")
    workflow.add_edge("fetch_data", "technical_analysis")
    workflow.add_edge("technical_analysis", "mcp_tools")
    workflow.add_edge("mcp_tools", "predict_prices")
    workflow.add_edge("predict_prices", "generate_recommendation")
    workflow.add_edge("generate_recommendation", "format_output")
    workflow.add_edge("format_output", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app


# ===== MAIN AGENT FUNCTION =====

async def run_trading_agent(user_message: str) -> str:
    """
    Run the complete trading agent workflow.
    
    Args:
        user_message: User's investment query
        
    Returns:
        Final recommendation text
    """
    # Create agent
    agent = create_trading_agent()
    
    # Initialize state
    initial_state = {
        'messages': [HumanMessage(content=user_message)],
        'crypto_symbol': '',
        'investment_amount': None,
        'risk_tolerance': settings.default_risk_tolerance,
        'investment_horizon': settings.default_investment_horizon,
        'current_price_data': None,
        'technical_analysis': None,
        'price_predictions': None,
        'market_sentiment': None,
        'mcp_analysis': None,
        'tool_calls_made': [],
        'trading_recommendation': None,
        'final_report': None,
        'error': None,
        'step_completed': []
    }
    
    # Run workflow
    print("\n🚀 Starting Crypto Trading Agent...\n")
    
    final_state = await agent.ainvoke(initial_state)
    
    print("\n✨ Analysis complete!\n")
    
    # Return final message
    if final_state['messages']:
        return final_state['messages'][-1].content
    else:
        return "Error: No recommendation generated"


def run_trading_agent_sync(user_message: str) -> str:
    """
    Synchronous version of run_trading_agent.
    
    Args:
        user_message: User's investment query
        
    Returns:
        Final recommendation text
    """
    # Create agent
    agent = create_trading_agent()
    
    # Initialize state
    initial_state = {
        'messages': [HumanMessage(content=user_message)],
        'crypto_symbol': '',
        'investment_amount': None,
        'risk_tolerance': settings.default_risk_tolerance,
        'investment_horizon': settings.default_investment_horizon,
        'current_price_data': None,
        'technical_analysis': None,
        'price_predictions': None,
        'market_sentiment': None,
        'mcp_analysis': None,
        'tool_calls_made': [],
        'trading_recommendation': None,
        'final_report': None,
        'error': None,
        'step_completed': []
    }
    
    # Run workflow
    print("\n🚀 Starting Crypto Trading Agent...\n")
    
    final_state = agent.invoke(initial_state)
    
    print("\n✨ Analysis complete!\n")
    
    # Return final message
    if final_state['messages']:
        return final_state['messages'][-1].content
    else:
        return "Error: No recommendation generated"
