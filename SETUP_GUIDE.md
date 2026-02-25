# 🎯 Complete Setup Guide - Crypto Trading Agent

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [API Key Setup](#api-key-setup)
4. [First Run](#first-run)
5. [Usage Modes](#usage-modes)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## 🖥️ System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB+ recommended for model training)
- **Internet:** Required for fetching crypto data and API calls
- **Disk Space:** ~2GB for dependencies

---

## 🚀 Installation Steps

### Step 1: Verify Python Installation

```bash
python --version
```

Should show Python 3.8 or higher. If not installed, download from [python.org](https://python.org).

### Step 2: Navigate to Project Directory

```bash
cd "c:\Users\Anurag Yadav\Desktop\Dex Agent"
```

### Step 3: Activate Virtual Environment

The virtual environment is already created. Activate it:

**Windows (PowerShell):**
```powershell
.\env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
env\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source env/bin/activate
```

You should see `(env)` before your command prompt.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages (~1-2GB download). May take 5-10 minutes.

### Step 5: Verify Installation

```bash
python test_setup.py
```

This runs comprehensive tests on all components.

---

## 🔑 API Key Setup

### Get Grok API Key (Required)

1. Visit [https://console.x.ai/](https://console.x.ai/)
2. Sign up or log in with your account
3. Navigate to **API Keys** section
4. Click **"Create API Key"**
5. Copy the generated key (starts with `xai-...`)

**Note:** Grok offers free tier with generous limits for testing.

### Get LangSmith API Key (Optional but Recommended)

1. Visit [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up for free account
3. Go to **Settings** → **API Keys**
4. Click **"Create API Key"**
5. Copy the key (starts with `ls...`)

**Benefits:**
- Track all agent runs
- Debug workflows visually
- Monitor performance and costs
- View detailed traces

### Configure Environment File

1. **Copy the example file:**
   ```bash
   copy .env.example .env
   ```

2. **Edit `.env` file** (use Notepad or any text editor):
   ```env
   # Replace with your actual keys
   XAI_API_KEY=xai-your-actual-key-here
   LANGSMITH_API_KEY=ls-your-actual-key-here
   
   # Keep these settings
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=crypto-trading-agent
   ```

3. **Save the file**

---

## 🎬 First Run

### Option 1: Test Components First (Recommended)

```bash
python test_setup.py
```

Expected output:
```
✅ PASS - Package Imports
✅ PASS - Configuration
✅ PASS - Data Fetcher
✅ PASS - Technical Analysis
✅ PASS - GRU Model

Results: 5/5 tests passed
🎉 All tests passed! Your agent is ready to use.
```

### Option 2: Run Interactive Mode

```bash
python main.py
```

You'll see the welcome screen. Try these queries:

```
Your query: Should I invest in Bitcoin? I have $5000 and medium risk tolerance

Your query: Analyze Ethereum for short-term trading

Your query: Give me analysis for Solana with $2000 for 14 days
```

Type `quit` to exit.

### Option 3: Single Query Mode

```bash
python main.py "Should I invest $3000 in Bitcoin?"
```

### Option 4: Run API Server

```bash
python main.py api
```

Then visit: `http://localhost:8000/docs` for interactive API documentation.

---

## 💡 Usage Modes

### 1. Interactive Chat (Recommended for Beginners)

```bash
python main.py
```

**Pros:**
- Easy to use
- Can ask multiple questions
- Beautiful formatted output
- Most user-friendly

**Example Session:**
```
🚀 Crypto Trading Agent

Your query: Should I invest in Bitcoin?
[Analysis results...]

Your query: Compare Ethereum and Solana
[Analysis results...]

Your query: quit
👋 Thank you for using Crypto Trading Agent!
```

### 2. Single Query (Good for Scripts)

```bash
python main.py "Analyze Bitcoin for $5000 investment with high risk"
```

**Pros:**
- Quick one-off analysis
- Can be used in shell scripts
- Saves output to variable

### 3. REST API (For Applications)

```bash
# Terminal 1: Start server
python main.py api

# Terminal 2: Make requests
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should I invest in Bitcoin?",
    "amount": 5000,
    "risk_tolerance": "medium"
  }'
```

**Pros:**
- Integrate with web apps
- Multiple concurrent requests
- RESTful interface
- Interactive docs at `/docs`

### 4. Example Scripts

```bash
python examples.py
```

Provides 10 different usage examples:
1. Simple Query
2. Compare Cryptos
3. Technical Analysis Only
4. Price Prediction Only
5. Custom Risk Tolerance
6. Market Sentiment
7. Generate Charts
8. API Integration
9. Batch Analysis
10. Save/Load Model

---

## 🔧 Troubleshooting

### Issue: "XAI_API_KEY not configured"

**Solution:**
1. Check `.env` file exists (not `.env.example`)
2. Open `.env` and verify API key is present
3. Make sure there are no quotes around the key
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
- API rate limit reached (wait 5 minutes)
- Invalid cryptocurrency symbol

**Solution:**
- Check internet connection
- Try a different symbol (BTC, ETH, SOL)
- Wait a few minutes if rate limited

### Issue: Model training is slow

**Solution:**
- Reduce epochs in code (default is 30)
- Use smaller GRU units
- Upgrade to more powerful hardware
- The first run is always slower (compiling)

### Issue: "Insufficient data for GRU predictions"

**Cause:** Less than 60 days of historical data

**Solution:**
- The agent automatically uses fallback prediction
- This is expected and handled gracefully
- Predictions are based on trend analysis instead

### Issue: API server won't start

**Solution:**
```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Check if port 8000 is in use
# Windows:
netstat -ano | findstr :8000

# macOS/Linux:
lsof -i :8000

# Kill the process or use different port
```

---

## ⚙️ Advanced Configuration

### Customize Model Parameters

Edit [config.py](config.py):

```python
# Model settings
MODEL_NAME = "grok-beta"
TEMPERATURE = 0.1  # Lower = more focused, Higher = more creative
MAX_TOKENS = 4096

# Trading parameters
DEFAULT_RISK_TOLERANCE = "medium"  # low, medium, high
DEFAULT_INVESTMENT_HORIZON = 7     # days
```

### Customize GRU Model

Edit in [trading_agent.py](trading_agent.py), `predict_prices` function:

```python
predictor = GRUPricePredictor(
    sequence_length=60,    # Length of input sequences
    features=5,            # OHLCV features
    gru_units=[64, 32],   # Layers (increase for more complexity)
    dropout_rate=0.2       # Regularization (0.2-0.5)
)

predictor.train(
    historical_data,
    epochs=30,             # Increase for better accuracy (slower)
    batch_size=32,        # Larger = faster, more memory
    verbose=0
)
```

### Change Data Sources

Edit [data_fetcher.py](data_fetcher.py) to:
- Add more exchanges
- Use different API providers
- Cache data locally
- Add new indicators

### Enable Binance API (Optional)

If you have Binance account:

1. Get API keys from Binance
2. Add to `.env`:
   ```env
   BINANCE_API_KEY=your_binance_key
   BINANCE_SECRET_KEY=your_binance_secret
   ```

3. Modify initialization in code:
   ```python
   fetcher = get_data_fetcher(
       use_binance=True,
       api_key=settings.binance_api_key,
       api_secret=settings.binance_secret_key
   )
   ```

---

## 📊 Interpreting Results

### Technical Indicators Guide

**RSI (Relative Strength Index):**
- < 30: Oversold (potential buy signal)
- 30-70: Neutral
- > 70: Overbought (potential sell signal)

**Bollinger Bands:**
- Price at lower band: Oversold
- Price at upper band: Overbought
- Squeeze (narrow bands): Breakout coming

**EMA Crossovers:**
- Golden Cross: Short EMA crosses above long EMA (bullish)
- Death Cross: Short EMA crosses below long EMA (bearish)

**Volume:**
- High volume + price increase: Strong bullish signal
- High volume + price decrease: Strong bearish signal
- Low volume: Weak signals

### Risk Tolerance Impact

**Low Risk:**
- Smaller position sizes
- More conservative entry points
- Tighter stop-losses
- Preference for stable coins

**Medium Risk:**
- Balanced approach
- Moderate position sizing
- Standard stop-losses
- Mix of stable and growth coins

**High Risk:**
- Larger positions
- Aggressive entries
- Wider stop-losses
- Focus on high-growth potential

---

## 🎓 Learning Resources

### Understanding Crypto Trading
- [Investopedia: Cryptocurrency](https://www.investopedia.com/cryptocurrency-4427699)
- [TradingView: Education](https://www.tradingview.com/education/)

### Technical Analysis
- [Babypips: Technical Analysis](https://www.babypips.com/learn/forex/technical-analysis)

### LangGraph & LangChain
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

### Machine Learning for Trading
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

## 📞 Support & Community

### Getting Help
1. Check this guide first
2. Run `python test_setup.py` for diagnostics
3. Review LangSmith traces for debugging
4. Check error messages carefully

### Best Practices
- ✅ Start with small test amounts
- ✅ Always verify API keys are valid
- ✅ Use LangSmith for debugging
- ✅ Keep your API keys secret (.env in .gitignore)
- ✅ Update dependencies regularly: `pip install --upgrade -r requirements.txt`

---

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Backup Your Work
```bash
# Backup .env file (contains keys)
copy .env .env.backup

# Backup trained models
xcopy /E /I models models_backup
```

### Monitor LangSmith
Visit [https://smith.langchain.com](https://smith.langchain.com) regularly to:
- Check token usage
- Monitor costs
- Debug failed runs
- Optimize performance

---

## ✅ Quick Checklist

Before using the agent, ensure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with valid API keys
- [ ] Test script passed (`python test_setup.py`)
- [ ] Internet connection active
- [ ] LangSmith project created (optional)

---

**You're all set! Start exploring crypto markets with AI-powered analysis! 🚀📈**

For quick reference, see [QUICKSTART.md](QUICKSTART.md)
