"""MCP (Model Context Protocol) Tools Integration for Trading Agent.

This module provides MCP toolbox integration using toolbox_langchain client to enhance 
the model's capabilities with:
- Real-time market data tools
- Advanced technical analysis tools  
- Risk assessment tools
- Portfolio optimization tools
- News sentiment analysis tools
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

try:
    from toolbox_langchain import ToolboxClient
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("⚠️ toolbox_langchain not installed. MCP tools will use fallback implementation.")

from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer
from web_search import get_web_search


class MCPToolbox:
    """MCP Toolbox providing enhanced tools for the trading agent using ToolboxClient."""
    
    def __init__(self, toolbox_url: Optional[str] = None):
        """Initialize MCP Toolbox with ToolboxClient or fallback to custom tools.
        
        Args:
            toolbox_url: Optional URL for ToolboxClient connection
        """
        self.data_fetcher = get_data_fetcher()
        self.web_search = get_web_search()
        self.toolbox_client = None
        self.custom_tools_registered = False
        
        # Try to initialize ToolboxClient if available
        if TOOLBOX_AVAILABLE:
            try:
                # ToolboxClient requires a URL - use empty string if not provided
                url = toolbox_url if toolbox_url else ""
                self.toolbox_client = ToolboxClient(url=url)
                print("✅ ToolboxClient initialized successfully")
            except Exception as e:
                print(f"⚠️ ToolboxClient initialization failed: {e}. Using fallback implementation.")
                self.toolbox_client = None
        
        # Register custom fallback tools if ToolboxClient not available
        if not self.toolbox_client:
            self._register_custom_tools()
    
    def _register_custom_tools(self):
        """Register custom fallback tools when ToolboxClient is not available."""
        self.custom_tools = {
            "calculate_risk_metrics": self.calculate_risk_metrics,
            "get_support_resistance_levels": self.get_support_resistance_levels,
            "analyze_volume_profile": self.analyze_volume_profile,
            "get_correlation_analysis": self.get_correlation_analysis,
            "search_latest_news": self.search_latest_news,
        }
        self.custom_tools_registered = True
        print(f"📦 Registered {len(self.custom_tools)} custom fallback tools")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for function calling.
        
        Returns schemas from ToolboxClient if available, otherwise custom tool schemas.
        """
        # Try to get schemas from ToolboxClient first
        if self.toolbox_client:
            try:
                # ToolboxClient provides get_tools() method that returns tool definitions
                if hasattr(self.toolbox_client, 'get_tools'):
                    toolbox_tools = self.toolbox_client.get_tools()
                    if toolbox_tools:
                        print(f"📦 Loaded {len(toolbox_tools)} tools from ToolboxClient")
                        return toolbox_tools
            except Exception as e:
                print(f"⚠️ Error getting tools from ToolboxClient: {e}. Using fallback.")
        
        # Fallback to custom tool schemas
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_risk_metrics",
                    "description": "Calculate comprehensive risk metrics including Sharpe ratio, maximum drawdown, Value at Risk (VaR), and volatility for a cryptocurrency.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Cryptocurrency symbol (e.g., 'BTC', 'ETH')"},
                            "days": {"type": "integer", "description": "Lookback period in days (default: 90)", "default": 90}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_support_resistance_levels",
                    "description": "Identify key support and resistance price levels based on historical data for trading decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Cryptocurrency symbol"},
                            "days": {"type": "integer", "description": "Lookback period in days (default: 30)", "default": 30}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_volume_profile",
                    "description": "Analyze trading volume patterns, trends, and detect unusual volume spikes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Cryptocurrency symbol"},
                            "days": {"type": "integer", "description": "Analysis period in days (default: 30)", "default": 30}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_correlation_analysis",
                    "description": "Analyze price correlation between a cryptocurrency and Bitcoin to understand market coupling.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Cryptocurrency symbol to analyze"},
                            "days": {"type": "integer", "description": "Correlation period in days (default: 30)", "default": 30}
                        },
                        "required": ["symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_latest_news",
                    "description": "Search for latest news and sentiment about a cryptocurrency using web search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Cryptocurrency symbol"},
                            "query": {"type": "string", "description": "Optional specific search query", "default": ""}
                        },
                        "required": ["symbol"]
                    }
                }
            }
        ]
    
    def get_as_langchain_tools(self) -> List[Any]:
        """Get tools as LangChain-compatible tool objects.
        
        Returns LangChain tools from ToolboxClient if available.
        """
        if self.toolbox_client:
            try:
                # ToolboxClient has as_langchain_tools() method
                if hasattr(self.toolbox_client, 'as_langchain_tools'):
                    langchain_tools = self.toolbox_client.as_langchain_tools()
                    print(f"🔗 Loaded {len(langchain_tools)} LangChain tools from ToolboxClient")
                    return langchain_tools
            except Exception as e:
                print(f"⚠️ Error getting LangChain tools: {e}")
        
        # Return empty list if ToolboxClient not available
        return []
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments.
        
        Tries ToolboxClient first, then falls back to custom implementation.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary
            
        Returns:
            Dictionary with success status and result/error
        """
        # Try ToolboxClient first
        if self.toolbox_client:
            try:
                if hasattr(self.toolbox_client, 'call_tool'):
                    result = self.toolbox_client.call_tool(tool_name, arguments)
                    return {"success": True, "result": result}
            except Exception as e:
                print(f"⚠️ ToolboxClient execution failed for {tool_name}: {e}. Using fallback.")
        
        # Fallback to custom tools
        if self.custom_tools_registered and tool_name in self.custom_tools:
            try:
                tool_function = self.custom_tools[tool_name]
                result = tool_function(**arguments)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Tool {tool_name} not found"}
    
    # ===== CUSTOM TOOL IMPLEMENTATIONS (Fallback) =====
    
    def calculate_risk_metrics(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            historical_data = self.data_fetcher.get_historical_data(symbol, days=days)
            
            if historical_data.empty:
                return {"error": "No historical data available"}
            
            prices = historical_data['price']
            daily_returns = prices.pct_change().dropna()
            
            # Sharpe Ratio (assuming 0% risk-free rate)
            avg_return = daily_returns.mean()
            std_return = daily_returns.std()
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            
            # Maximum Drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Value at Risk (95% confidence)
            var_95 = daily_returns.quantile(0.05) * 100
            
            # Downside Deviation
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() * (252 ** 0.5) * 100 if len(negative_returns) > 0 else 0
            
            return {
                "symbol": symbol.upper(),
                "period_days": days,
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown_pct": float(max_drawdown),
                "value_at_risk_95_pct": float(var_95),
                "volatility_pct": float(std_return * (252 ** 0.5) * 100),
                "downside_deviation_pct": float(downside_deviation),
                "risk_rating": "low" if sharpe_ratio > 1.0 and max_drawdown > -20 else "medium" if sharpe_ratio > 0.5 else "high"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_support_resistance_levels(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        try:
            historical_data = self.data_fetcher.get_historical_data(symbol, days=days)
            
            if historical_data.empty:
                return {"error": "No historical data available"}
            
            prices = historical_data['price']
            highs = historical_data['high'] if 'high' in historical_data.columns else prices
            lows = historical_data['low'] if 'low' in historical_data.columns else prices
            
            # Use percentile-based levels
            resistance_levels = [
                float(highs.quantile(0.95)),
                float(highs.quantile(0.75)),
                float(highs.quantile(0.60))
            ]
            
            support_levels = [
                float(lows.quantile(0.40)),
                float(lows.quantile(0.25)),
                float(lows.quantile(0.05))
            ]
            
            current_price = float(prices.iloc[-1])
            
            return {
                "symbol": symbol.upper(),
                "current_price": current_price,
                "resistance_levels": sorted(resistance_levels, reverse=True),
                "support_levels": sorted(support_levels, reverse=True),
                "nearest_resistance": min([r for r in resistance_levels if r > current_price], default=None),
                "nearest_support": max([s for s in support_levels if s < current_price], default=None)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_volume_profile(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Analyze volume patterns."""
        try:
            historical_data = self.data_fetcher.get_historical_data(symbol, days=days)
            
            if historical_data.empty or 'volume' not in historical_data.columns:
                return {"error": "No volume data available"}
            
            volumes = historical_data['volume']
            avg_volume = volumes.mean()
            
            # Detect volume spikes (> 2x average)
            volume_spikes = volumes[volumes > avg_volume * 2]
            
            # Recent volume trend
            recent_volume = volumes.tail(7).mean()
            volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
            
            return {
                "symbol": symbol.upper(),
                "period_days": days,
                "average_volume": float(avg_volume),
                "current_volume": float(volumes.iloc[-1]),
                "volume_trend": volume_trend,
                "volume_spikes_count": int(len(volume_spikes)),
                "volume_change_pct": float(((volumes.iloc[-1] - avg_volume) / avg_volume) * 100),
                "high_volume_days": int((volumes > avg_volume * 1.5).sum())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_correlation_analysis(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Analyze correlation with Bitcoin."""
        try:
            if symbol.upper() == 'BTC':
                return {
                    "symbol": "BTC",
                    "correlation_with_btc": 1.0,
                    "interpretation": "Bitcoin correlates perfectly with itself",
                    "period_days": days
                }
            
            # Get data for both coins
            target_data = self.data_fetcher.get_historical_data(symbol, days=days)
            btc_data = self.data_fetcher.get_historical_data('BTC', days=days)
            
            if target_data.empty or btc_data.empty:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Calculate correlation
            target_returns = target_data['price'].pct_change().dropna()
            btc_returns = btc_data['price'].pct_change().dropna()
            
            # Align indices
            common_index = target_returns.index.intersection(btc_returns.index)
            if len(common_index) < 5:
                return {"error": "Not enough overlapping data points"}
            
            correlation = target_returns.loc[common_index].corr(btc_returns.loc[common_index])
            
            # Interpret correlation
            if correlation > 0.7:
                interpretation = "Strong positive correlation - moves closely with BTC"
            elif correlation > 0.3:
                interpretation = "Moderate positive correlation"
            elif correlation > -0.3:
                interpretation = "Weak or no correlation - independent movement"
            elif correlation > -0.7:
                interpretation = "Moderate negative correlation"
            else:
                interpretation = "Strong negative correlation - moves opposite to BTC"
            
            return {
                "symbol": symbol.upper(),
                "correlation_with_btc": float(correlation),
                "interpretation": interpretation,
                "period_days": days,
                "data_points": len(common_index)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_latest_news(self, symbol: str, query: str = "") -> Dict[str, Any]:
        """Search for latest news and sentiment."""
        try:
            if not self.web_search or not self.web_search.available:
                return {
                    "symbol": symbol.upper(),
                    "news_available": False,
                    "message": "Web search not configured. Set TAVILY_API_KEY to enable news search."
                }
            
            search_query = query if query else symbol
            results = self.web_search.search_crypto_news(search_query)
            
            return {
                "symbol": symbol.upper(),
                "news_available": True,
                "sentiment": results.get("sentiment", "neutral"),
                "article_count": len(results.get("articles", [])),
                "summary": results.get("summary", ""),
                "articles": results.get("articles", [])[:5]  # Top 5 articles
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_mcp_toolbox = None


def get_mcp_toolbox(toolbox_url: Optional[str] = None, reset: bool = False) -> MCPToolbox:
    """Get singleton MCP Toolbox instance.
    
    Args:
        toolbox_url: Optional URL for ToolboxClient connection
        reset: Force recreation of singleton instance
        
    Returns:
        MCPToolbox instance
    """
    global _mcp_toolbox
    if _mcp_toolbox is None or reset:
        _mcp_toolbox = MCPToolbox(toolbox_url=toolbox_url)
    return _mcp_toolbox
