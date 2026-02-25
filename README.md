# 🚀 Crypto Trading Agent

An advanced AI-powered cryptocurrency trading agent that provides comprehensive market analysis and personalized investment recommendations using **Groq AI** (lightning-fast and free!), **LangGraph**, and **CoinGecko API**.

GROQ_API_KEY=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

# ===== OPTIONAL =====
# LangSmith for monitoring agent runs (free tier available)
# Get key from: https://smith.langchain.com
LANGSMITH_API_KEY=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


## ✨ Features

### � DUAL PREDICTOR SYSTEM (NEW!)
Combines TWO independent predictors for superior accuracy:

1. **Momentum-Based Predictor** ⚡
   - Multi-timeframe momentum (7d, 30d, 1y)
   - Fast mathematical calculations
   - Weighted trend analysis
   
2. **RL Q-Learning Predictor** 🤖
   - Reinforcement Learning approach
   - Adaptive pattern recognition
   - State-action-reward framework
   - Learns from market conditions

**Final Decision = 50% Momentum + 50% RL**  
✅ No training required - works instantly!  
✅ Higher confidence through consensus  
✅ Balanced reactive + predictive approach

### 📊 Comprehensive Technical Analysis (CoinGecko-Powered)
1. **Price Action Analysis** - Multi-timeframe trend detection
2. **RSI Zone Analysis** - Overbought/oversold conditions
3. **Volume Analysis** - Volume trends and patterns
4. **Exponential Moving Averages** - 9, 21, 50, 200 EMAs with crossover signals
5. **Support & Resistance** - Key price levels identification
6. **Bollinger Bands** - Volatility and squeeze detection
7. **Liquidity Areas** - High-volume zones for better entry/exit

### 🤖 LangGraph Multi-Agent Workflow
- **State Management** - Sophisticated workflow orchestration
- **LangSmith Integration** - Complete observability and debugging
- **Groq AI** - Lightning-fast inference (powered by Groq LPUs)
- **Error Handling** - Robust fallback mechanisms

### 💡 Smart Recommendations
- **Risk-Adjusted Strategies** - Tailored to your risk tolerance
- **Position Sizing** - Optimal investment allocation
- **Entry/Exit Strategies** - DCA vs. lump sum recommendations
- **Stop-Loss & Targets** - Clear risk management

## 🆓 100% Free APIs!

- **Groq AI**: Free tier with generous limits (faster than GPT-4!)
- **CoinGecko**: Free API with 50 calls/minute
- **LangSmith**: Free tier for monitoring (optional)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd "c:\Users\Anurag Yadav\Desktop\Dex Agent"

# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
# source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your API keys:

```env
# Required: Get FREE key from https://console.groq.com/
GROQ_API_KEY=your_actual_groq_api_key

# Optional but Recommended: Get from https://smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key

# Enable LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=crypto-trading-agent
```

### Where to Get API Keys

#### Groq API Key (Required - 100% FREE!)
1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up with Google/GitHub (takes 30 seconds)
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

**Why Groq?**
- ⚡ **10x Faster** than GPT-4
- 🆓 **Completely Free** tier
- 🚀 **Powered by LPUs** (Language Processing Units)
- 🎯 **Excellent for analysis** tasks

#### LangSmith API Key (Optional - for monitoring)
1. Visit [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up for a free account
3. Go to Settings → API Keys
4. Create a new API key
5. Copy and paste into `.env` file

## 🚀 Usage

### Interactive Chat Mode (Recommended)

```bash
python main.py
```

This starts an interactive chat where you can ask multiple questions:

```
Your query: Should I invest in Bitcoin? I have $5000 and medium risk tolerance
Your query: Analyze Ethereum for short-term trading
Your query: Give me analysis for SOL with $2000 for 14 days
```

### Single Query Mode

```bash
python main.py "Should I invest $3000 in Bitcoin for 7 days?"
```

### REST API Server

```bash
python main.py api
```

Then access:
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**API Example Request:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should I invest in Bitcoin?",
    "amount": 5000,
    "risk_tolerance": "medium",
    "investment_horizon": 7
  }'
```

## 📖 Example Queries

```
✅ "Should I invest in Bitcoin? I have $5000 and high risk tolerance"

✅ "Give me analysis for Ethereum with $2000 investment for 14 days"

✅ "Analyze Solana for short-term trading with low risk"

✅ "What's the best time to buy ADA with $1000?"

✅ "Should I buy or sell BNB right now?"

✅ "Give me detailed analysis for MATIC"
```

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  LangGraph Workflow                 │
│  ┌──────CoinGecko API            │  │
│  │    - Price, Volume, Market    │  │
│  │    - Fear & Greed Index       │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ 3. Technical Analysis         │  │
│  │    - Price Action             │  │
│  │    - RSI, Volume, EMA         │  │
│  │    - S/R, Bollinger, Liquidity│  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ 4. Price Prediction           │  │
│  │    - Momentum Analysis        │  │
│  │    - No training needed!      │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ 5. Generate Recommendation    │  │
│  │    - Groq AI (Lightning Fast!)│  │
│  │    - 7-day forecast           │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ 5. Generate Recommendation    │  │
│  │    - Risk assessment          │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ 6. Format Output              │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Report   │
└─────────────────┘
```

## 📁 Project Structure

```
Dex Agent/
├── main.py                    # Main application entry point
├── trading_agent.py           # LangGraph workflow & agent logic
├── data_fetcher.py           # CoinGecko API integration
├── technical_analysis.py      # All technical indicators
├── simple_predictor.py        # Momentum-based predictions (no training!)
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies (lightweight!)
├── .env.example             # Environment variables template
├── .env                     # Your actual API keys (you create this)
└── README.md               # This file
```

## 🔍 LangSmith Monitoring

Once you have LangSmith configured, you can:

1. **View all agent runs** at [https://smith.langchain.com](https://smith.langchain.com)
2. **Debug workflows** - See each step of the LangGraph execution
3. **Monitor performance** - Track latency, token usage, costs
4. **Analyze traces** - Understand agent decision-making

## ⚙️ Configuration

Edit `config.py` or `.env` to customize:

```python
# Model settings
MODEL_NAME=grok-beta
TEMPERATURE=0.1
MAX_TOKENS=4096

# Trading parameters
DEFAULT_RISK_TOLERANCE=medium  # low, medium, high
DEFAULT_INVESTMENT_HORIZON=7   # days
```

## 🧪 Testing Individual Components

### Test Technical Analysis
```python
from data_fetcher import get_data_fetcher
from technical_analysis import TechnicalAnalyzer

fetcher = get_data_fetcher()
data = fetcher.get_historical_data('BTC', days=90)
analyzer = TechnicalAnalyzer(data)
analysis = analyzer.get_complete_analysis()
print(analysis)
```

### Test CoinGecko API
```python
from data_fetcher import get_data_fetcher

fetcher = get_data_fetcher()

# Current price
price = fetcher.get_current_price('BTC')
print(price)

# Detailed coin info with indicators
details = fetcher.get_coin_details('BTC')
print(details)

# Historical data
history = fetcher.get_historical_data('ETH', days=30)
print(history)
```

### Test Price Predictions
```python
from simple_predictor import predict_with_momentum
from data_fetcher import get_data_fetcher

fetcher = get_data_fetcher()
data = fetcher.get_historical_data('BTC', days=90)
predictions = predict_with_momentum(data, periods=7)
print(predictions)
```

## 🛡️ Disclaimer

**IMPORTANT:** This trading agent is for **educational and research purposes only**.

- ❌ NOT financial advice
- ❌ NOT a substitute for professional financial consultation
- ❌ NO guarantees of profit or accuracy
- ✅ Always do your own research (DYOR)
- ✅ Only invest what you can afford to lose
- ✅ Cryptocurrency investments carry significant risk

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional technical indicators
- More sophisticated RL strategies
- Multiple exchange integrations
- Portfolio optimization
- Backtesting framework
- Web UI interface

## 📝 License
Issue: "GROQ_API_KEY not configured"

**Solution:**
1. Check `.env` file exists (not `.env.example`)
2. Get FREE API key from https://console.groq.com/
3. Add to `.env` file: `GROQ_API_KEY=gsk_your_key_here`
4. Restart the application

### Issue: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
env\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "No historical data available"

**Possible causes:**
- Internet connection issue
- CoinGecko API rate limit (50 calls/minute on free tier)
- Invalid cryptocurrency symbol

**Solution:**
- Check internet connection
- Try Real-time WebSocket updates
- [ ] Advanced chart visualizations
- [ ] Integration with trading platforms
- [ ] Sentiment analysis from social media
- [ ] On-chain analytics integration
- [ ] Mobile app version

---

**Built with ❤️ using Groq AI, LangGraph, and CoinGecko
- The agent will automatically use fallback prediction methods

### Module import errors
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

## 📧 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review LangSmith traces for debugging
3. Check API key validity

## 🎯 Roadmap

- [ ] Web-based UI with React
- [ ] Multi-coin portfolio optimization
- [ ] Backtesting engine
- [ ] Real-time alerts and notifications
- [ ] Integration with trading platforms
- [ ] Advanced RL algorithms (PPO, A3C)
- [ ] Sentiment analysis from social media
- [ ] On-chain analytics integration

---

**Built with ❤️ using LangGraph, Groq AI, and CoinGecko**

*Happy Trading! 🚀📈*
