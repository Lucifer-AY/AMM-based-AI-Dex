# ⚡ Super Quick Start - 3 Minutes to Running!

## Step 1: Get Groq API Key (1 minute) 🔑

1. Go to **[https://console.groq.com/](https://console.groq.com/)**
2. Click **"Sign In"** → Sign up with Google/GitHub
3. Click **"API Keys"** → **"Create API Key"**
4. Copy the key (starts with `gsk_...`)

**✅ Done! Groq is 100% FREE and super fast!**

---

## Step 2: Setup Project (1 minute) ⚙️

```powershell
# Navigate to project
cd "c:\Users\Anurag Yadav\Desktop\Dex Agent"

# Activate environment
.\env\Scripts\Activate.ps1

# Install packages (one-time, ~2 min)
pip install -r requirements.txt

# Create .env file
copy .env.example .env
```

Now edit `.env` file and paste your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## Step 3: Run! (30 seconds) 🚀

```powershell
python main.py
```

Type your query:
```
Your query: Should I invest $5000 in Bitcoin?
```

**That's it! The agent will analyze and give you recommendations!**

---

## 💡 Example Queries

```
✅ Should I invest in Bitcoin? I have $5000 and medium risk tolerance

✅ Analyze Ethereum for short-term trading

✅ Give me analysis for Solana with $3000 for 14 days

✅ Compare BTC and ETH, which is better?

✅ Is now a good time to buy crypto?
```

---

## 🎯 What You'll Get

Each analysis provides:
- ✅ Current price & market data (from CoinGecko)
- ✅ Technical indicators (RSI, EMA, Bollinger Bands, Volume)
- ✅ 7-day price predictions with confidence intervals
- ✅ **BUY/SELL/HOLD** recommendation
- ✅ Entry strategy (when and how much to invest)
- ✅ Risk assessment & stop-loss levels
- ✅ Position sizing recommendations

All powered by:
- **Groq AI** (10x faster than GPT-4, 100% free!)
- **CoinGecko API** (Real-time crypto data, free!)
- **LangGraph** (Advanced multi-agent workflow)

---

## 🔧 Quick Test

```powershell
# Test all components
python test_setup.py
```

Should see:
```
✅ PASS - Package Imports
✅ PASS - Configuration
✅ PASS - Data Fetcher
✅ PASS - Technical Analysis
✅ PASS - Price Prediction

Results: 5/5 tests passed
🎉 All tests passed! Your agent is ready to use.
```

---

## 🆘 Issues?

**"GROQ_API_KEY not configured"**
→ Make sure you created `.env` file and pasted your Groq key

**"Module not found"**
→ Run: `pip install -r requirements.txt`

**"No data for crypto"**
→ Try: BTC, ETH, SOL, ADA (use popular coins)

---

## 🎓 Learn More

- Full docs: [README.md](README.md)
- Detailed setup: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Example scripts: `python examples.py`

---

**Ready to analyze crypto markets! 🚀📈**

*Remember: This is educational only, not financial advice!*
