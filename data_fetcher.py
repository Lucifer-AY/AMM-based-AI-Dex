"""Crypto Data Fetcher Module.

Fetches real-time and historical cryptocurrency data from CoinGecko API (100% Free).
Includes technical indicators, market data, and sentiment analysis.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time


# Rate limiting for CoinGecko free tier (10-50 calls/minute)
class RateLimiter:
    def __init__(self, min_interval=3.0):
        """Initialize rate limiter.
        
        Args:
            min_interval: Minimum seconds between requests (default 3s for safety)
        """
        self.min_interval = min_interval
        self.last_request_time = 0
    
    def wait(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class CryptoDataFetcher:
    """Fetch cryptocurrency data and indicators from CoinGecko API."""
    
    def __init__(self):
        """Initialize data fetcher with CoinGecko API."""
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limiter = RateLimiter(min_interval=3.0)
        # Simple cache to prevent duplicate calls within same session
        self._cache = {}
        self._cache_timestamp = {}
    
    def get_current_price(self, symbol: str) -> Dict:
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dict with current price and market data
        """
        try:
            # Check cache (valid for 30 seconds)
            cache_key = f"price_{symbol}"
            if cache_key in self._cache:
                cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
                if cache_age < 30:
                    print(f"  ✓ Using cached price data for {symbol}")
                    return self._cache[cache_key]
            
            # CoinGecko API (free tier)
            coin_id = self._symbol_to_coingecko_id(symbol)
            url = f"{self.coingecko_base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            
            # Apply rate limiting
            self.rate_limiter.wait()
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id in data:
                coin_data = data[coin_id]
                result = {
                    'symbol': symbol.upper(),
                    'price': coin_data.get('usd', 0),
                    'change_24h': coin_data.get('usd_24h_change', 0),
                    'volume_24h': coin_data.get('usd_24h_vol', 0),
                    'market_cap': coin_data.get('usd_market_cap', 0),
                    'timestamp': datetime.now().isoformat()
                }
                # Cache the result
                self._cache[cache_key] = result
                self._cache_timestamp[cache_key] = time.time()
                return result
            else:
                raise ValueError(f"No data found for {symbol}")
                
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return {
                'symbol': symbol.upper(),
                'price': 0,
                'change_24h': 0,
                'volume_24h': 0,
                'market_cap': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_historical_data(
        self,
        symbol: str,
        days: int = 90,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from CoinGecko.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days of historical data (max 365 for free tier)
            interval: Time interval ('daily' supported for free tier)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache (valid for 60 seconds since historical data changes less frequently)
            cache_key = f"historical_{symbol}_{days}"
            if cache_key in self._cache:
                cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
                if cache_age < 60:
                    print(f"  ✓ Using cached historical data for {symbol}")
                    return self._cache[cache_key]
            
            coin_id = self._symbol_to_coingecko_id(symbol)
            
            # CoinGecko OHLC endpoint (free tier: max 365 days, daily candles)
            url = f"{self.coingecko_base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': min(days, 365)  # Free tier limit
            }
            
            # Apply rate limiting
            self.rate_limiter.wait()
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return self._get_fallback_data(symbol, days)
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['volume'] = 0  # CoinGecko OHLC doesn't include volume in free tier
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the result
            self._cache[cache_key] = df
            self._cache_timestamp[cache_key] = time.time()
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self._get_fallback_data(symbol, days)
    
    def get_coin_details(self, symbol: str) -> Dict:
        """
        Get detailed information about a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dict with comprehensive market data
        """
        try:
            # Check cache (valid for 30 seconds)
            cache_key = f"details_{symbol}"
            if cache_key in self._cache:
                cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
                if cache_age < 30:
                    print(f"  ✓ Using cached details for {symbol}")
                    return self._cache[cache_key]
            
            coin_id = self._symbol_to_coingecko_id(symbol)
            url = f"{self.coingecko_base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            # Apply rate limiting
            self.rate_limiter.wait()
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            result = {
                'symbol': symbol.upper(),
                'name': data.get('name', ''),
                'price': market_data.get('current_price', {}).get('usd', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'market_cap_rank': data.get('market_cap_rank'),
                'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                'fully_diluted_valuation': market_data.get('fully_diluted_valuation', {}).get('usd', 0),
                'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'price_change_percentage_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_percentage_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_percentage_30d': market_data.get('price_change_percentage_30d', 0),
                'price_change_percentage_1y': market_data.get('price_change_percentage_1y', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                'atl': market_data.get('atl', {}).get('usd', 0),
                'atl_change_percentage': market_data.get('atl_change_percentage', {}).get('usd', 0),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'total_supply': market_data.get('total_supply', 0),
                'max_supply': market_data.get('max_supply', 0),
                'timestamp': datetime.now().isoformat()
            }
            # Cache the result
            self._cache[cache_key] = result
            self._cache_timestamp[cache_key] = time.time()
            return result
            
        except Exception as e:
            print(f"Error fetching coin details: {e}")
            # Fallback to simple price data
            return self.get_current_price(symbol)
    
    def get_fear_greed_index(self) -> Dict:
        """
        Get the Fear & Greed Index for crypto markets.
        
        Returns:
            Dict with current fear/greed data
        """
        try:
            # Using alternative fear/greed index API
            url = "https://api.alternative.me/fng/"
            params = {'limit': 1, 'format': 'json'}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('data'):
                latest = data['data'][0]
                return {
                    'index': int(latest.get('value', 50)),
                    'classification': latest.get('value_classification', 'Neutral'),
                    'timestamp': latest.get('timestamp'),
                    'time_until_update': latest.get('time_until_update')
                }
            else:
                return {'index': 50, 'classification': 'Neutral', 'error': 'No data'}
                
        except Exception as e:
            print(f"Error fetching fear/greed index: {e}")
            return {'index': 50, 'classification': 'Neutral', 'error': str(e)}
    
    def get_price_7d_change(self, symbol: str) -> float:
        """Get 7-day price change percentage."""
        try:
            data = self.get_coin_details(symbol)
            return data.get('price_change_percentage_7d', 0)
        except:
            return 0
    
    def get_price_30d_change(self, symbol: str) -> float:
        """Get 30-day price change percentage."""
        try:
            data = self.get_coin_details(symbol)
            return data.get('price_change_percentage_30d', 0)
        except:
            return 0
    
    def get_price_1y_change(self, symbol: str) -> float:
        """Get 1-year price change percentage."""
        try:
            data = self.get_coin_details(symbol)
            return data.get('price_change_percentage_1y', 0)
        except:
            return 0
    
    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """
        Convert symbol to CoinGecko ID.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            CoinGecko ID (e.g., 'bitcoin', 'ethereum')
        """
        # Mapping of common symbols to CoinGecko IDs
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'XRP': 'ripple',
            'LINK': 'chainlink',
            'MATIC': 'matic-network',
            'XLM': 'stellar',
            'BCH': 'bitcoin-cash',
            'LTC': 'litecoin',
            'AVAX': 'avalanche-2',
            'FTM': 'fantom',
            'ATOM': 'cosmos',
            'CRO': 'crypto-com-chain',
            'NEAR': 'near',
            'VET': 'vechain',
            'HBAR': 'hedera-hashgraph',
            'ZEC': 'zcash',
            'SHIB': 'shiba-inu',
            'PEPE': 'pepe',
            'MEME': 'meme-token'
        }
        
        return symbol_map.get(symbol.upper(), symbol.lower())
    
    def _get_fallback_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate fallback data if API fails."""
        print(f"Using fallback data for {symbol}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


# Singleton instance
_data_fetcher = None

def get_data_fetcher() -> CryptoDataFetcher:
    """Get or create global data fetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = CryptoDataFetcher()
    return _data_fetcher
