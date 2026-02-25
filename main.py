"""Main Application - Crypto Trading Agent Interface.

This provides multiple interfaces to interact with the trading agent:
1. CLI (Command Line Interface)
2. Interactive chat mode
3. REST API (FastAPI)
"""

import asyncio
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from trading_agent import run_trading_agent_sync, run_trading_agent
from config import settings


# Initialize Rich console for beautiful output
console = Console()


def display_welcome():
    """Display welcome message."""
    welcome_text = """
# 🚀 Crypto Trading Agent

**Powered by:**
- 🧠 Groq AI (Lightning Fast & Free!)
- 📊 LangGraph Multi-Agent Workflow
- 📈 CoinGecko API (Real-time Data & Indicators)
- 🔮 Advanced Momentum-Based Predictions
- 🔍 LangSmith Monitoring

**Features:**
✅ Real-time crypto data analysis
✅ 8-step comprehensive technical analysis
✅ AI-powered price predictions
✅ Personalized investment recommendations
✅ Risk-adjusted strategies

Type your investment query or 'quit' to exit.
    """
    console.print(Panel(Markdown(welcome_text), style="bold blue"))


def display_result(result: str):
    """Display analysis result."""
    console.print("\n")
    console.print(Panel(Markdown(result), title="📊 Analysis Result", style="green"))


def display_error(error: str):
    """Display error message."""
    console.print(Panel(f"❌ Error: {error}", style="bold red"))


def interactive_mode():
    """Run agent in interactive chat mode."""
    display_welcome()
    
    console.print("\n💡 Example queries:", style="bold yellow")
    console.print("   - 'Should I invest in Bitcoin? I have $5000 and medium risk tolerance'")
    console.print("   - 'Give me analysis for ETH with $2000 investment for 14 days'")
    console.print("   - 'Analyze Solana for short-term trading'\n")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]Your query[/bold cyan]")
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("\n👋 Thank you for using Crypto Trading Agent!", style="bold blue")
                break
            
            if not user_input.strip():
                continue
            
            # Show processing message
            with console.status("[bold green]🔍 Analyzing market data...", spinner="dots"):
                # Run agent
                result = run_trading_agent_sync(user_input)
            
            # Display result
            display_result(result)
            
        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="bold blue")
            break
        except Exception as e:
            display_error(str(e))


def single_query_mode(query: str):
    """Run agent for a single query."""
    console.print(f"\n🔍 Analyzing: {query}\n", style="bold cyan")
    
    try:
        with console.status("[bold green]Processing...", spinner="dots"):
            result = run_trading_agent_sync(query)
        
        display_result(result)
        
    except Exception as e:
        display_error(str(e))


def run_api_server():
    """Run FastAPI REST API server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(
            title="Crypto Trading Agent API",
            description="AI-powered cryptocurrency trading analysis and recommendations",
            version="1.0.0"
        )
        
        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        class QueryRequest(BaseModel):
            query: str
            symbol: Optional[str] = None
            amount: Optional[float] = None
            risk_tolerance: Optional[str] = "medium"
            investment_horizon: Optional[int] = 7
        
        class QueryResponse(BaseModel):
            success: bool
            recommendation: Optional[str] = None
            error: Optional[str] = None
        
        @app.get("/")
        async def root():
            return {
                "message": "Crypto Trading Agent API",
                "version": "1.0.0",
                "endpoints": {
                    "analyze": "/analyze (POST)",
                    "health": "/health (GET)"
                }
            }
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "langsmith_enabled": settings.langchain_tracing_v2}
        
        @app.post("/analyze", response_model=QueryResponse)
        async def analyze(request: QueryRequest):
            """Analyze cryptocurrency and provide trading recommendation."""
            try:
                # Build query from parameters
                query = request.query
                if request.symbol:
                    query += f" for {request.symbol}"
                if request.amount:
                    query += f" with ${request.amount}"
                
                # Run agent
                result = await run_trading_agent(query)
                
                return QueryResponse(success=True, recommendation=result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        console.print("\n🚀 Starting API Server...\n", style="bold green")
        console.print("📡 API Documentation: http://localhost:8000/docs", style="cyan")
        console.print("🔍 LangSmith Dashboard: https://smith.langchain.com\n", style="cyan")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except ImportError:
        console.print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn", style="bold red")
        sys.exit(1)


def main():
    """Main entry point."""
    # Check for API key
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        console.print("\n❌ Error: GROQ_API_KEY not configured!", style="bold red")
        console.print("\n📝 Please set your Groq API key in .env file:", style="yellow")
        console.print("   1. Copy .env.example to .env")
        console.print("   2. Get FREE API key from https://console.groq.com/")
        console.print("   3. Add GROQ_API_KEY=your_key_here to .env file\n")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "api":
            # Run API server
            run_api_server()
        elif command == "help":
            console.print("""
Usage:
    python main.py                    # Interactive chat mode
    python main.py api                # Run REST API server
    python main.py "your query here"  # Single query mode
            """)
        else:
            # Single query mode
            query = " ".join(sys.argv[1:])
            single_query_mode(query)
    else:
        # Interactive mode (default)
        interactive_mode()


if __name__ == "__main__":
    main()
