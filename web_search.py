"""Web Search Module using Tavily API.

Provides market sentiment, news, and real-time information about cryptocurrencies
using Tavily's search API for enhanced trading analysis.
"""

from typing import Dict, List, Optional
from config import settings


class TavilyWebSearch:
    """Search crypto market news and sentiment using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily search client.
        
        Args:
            api_key: Tavily API key (uses config if not provided)
        """
        self.api_key = api_key or settings.tavily_api_key
        self.available = bool(self.api_key and self.api_key != "")
        
        if self.available:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Tavily client initialization failed: {e}")
                self.available = False
    
    def search_crypto_news(self, symbol: str, days: int = 7) -> Dict:
        """
        Search recent news and sentiment about a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'Bitcoin', 'Ethereum')
            days: Days of news to search
            
        Returns:
            Dict with news articles and sentiment
        """
        if not self.available:
            return {
                'sentiment': 'neutral',
                'articles': [],
                'summary': 'Tavily API not configured',
                'error': 'No API key provided'
            }
        
        try:
            query = f"{symbol} cryptocurrency price news sentiment last {days} days"
            
            response = self.client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=5
            )
            
            sentiment = self._analyze_sentiment(response)
            articles = self._extract_articles(response)
            
            return {
                'sentiment': sentiment,
                'articles': articles,
                'summary': response.get('answer', 'No summary available'),
                'source': 'Tavily Search'
            }
            
        except Exception as e:
            print(f"Error searching crypto news: {e}")
            return {
                'sentiment': 'neutral',
                'articles': [],
                'summary': str(e),
                'error': 'Search failed'
            }
    
    def search_market_analysis(self, query: str) -> Dict:
        """
        Search for specific crypto market analysis.
        
        Args:
            query: Custom search query
            
        Returns:
            Dict with search results and analysis
        """
        if not self.available:
            return {
                'results': [],
                'summary': 'Tavily API not configured'
            }
        
        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=3
            )
            
            return {
                'results': self._extract_articles(response),
                'summary': response.get('answer', 'No analysis available')
            }
            
        except Exception as e:
            print(f"Error searching market analysis: {e}")
            return {
                'results': [],
                'summary': f'Search error: {str(e)}'
            }
    
    def search_trading_signals(self, symbol: str) -> Dict:
        """
        Search for recent trading signals and technical analysis.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict with trading signals and analysis
        """
        if not self.available:
            return {
                'signals': [],
                'analysis': 'Tavily API not configured'
            }
        
        try:
            query = f"{symbol} crypto trading signals buy sell technical analysis forecast"
            
            response = self.client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=5
            )
            
            articles = self._extract_articles(response)
            
            return {
                'signals': articles,
                'analysis': response.get('answer', 'No analysis available'),
                'query': query
            }
            
        except Exception as e:
            print(f"Error searching trading signals: {e}")
            return {
                'signals': [],
                'analysis': f'Search error: {str(e)}'
            }
    
    def _analyze_sentiment(self, response: Dict) -> str:
        """
        Analyze sentiment from search results.
        
        Args:
            response: Tavily search response
            
        Returns:
            Sentiment classification: bullish, bearish, neutral
        """
        try:
            answer = response.get('answer', '').lower()
            
            bullish_keywords = [
                'surge', 'rally', 'pump', 'moon', 'bull', 'bullish',
                'rise', 'gain', 'profit', 'up', 'green', 'positive'
            ]
            bearish_keywords = [
                'crash', 'dump', 'collapse', 'bear', 'bearish',
                'fall', 'decline', 'loss', 'down', 'red', 'negative'
            ]
            
            bullish_count = sum(answer.count(kw) for kw in bullish_keywords)
            bearish_count = sum(answer.count(kw) for kw in bearish_keywords)
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _extract_articles(self, response: Dict) -> List[Dict]:
        """
        Extract articles from Tavily response.
        
        Args:
            response: Tavily search response
            
        Returns:
            List of article dicts with title and URL
        """
        articles = []
        
        try:
            if 'results' in response:
                for result in response['results']:
                    articles.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('source', '')
                    })
        except Exception as e:
            print(f"Error extracting articles: {e}")
        
        return articles
    
    def get_sentiment_summary(self, symbol: str) -> str:
        """
        Get a brief sentiment summary for a crypto.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Sentiment string with context
        """
        if not self.available:
            return "Tavily search not available - configure TAVILY_API_KEY"
        
        news_data = self.search_crypto_news(symbol)
        sentiment = news_data.get('sentiment', 'neutral')
        articles_count = len(news_data.get('articles', []))
        
        sentiment_emoji = {
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '🟡'
        }
        
        emoji = sentiment_emoji.get(sentiment, '🟡')
        
        return f"{emoji} Market Sentiment: {sentiment.upper()} ({articles_count} recent articles)"


# Global search instance
_web_search = None

def get_web_search() -> TavilyWebSearch:
    """Get or create global web search instance."""
    global _web_search
    if _web_search is None:
        _web_search = TavilyWebSearch()
    return _web_search
