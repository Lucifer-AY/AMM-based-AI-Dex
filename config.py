"""Configuration management for Crypto Trading Agent."""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    
    # LangSmith
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGCHAIN_ENDPOINT"
    )
    langchain_project: str = Field(
        default="crypto-trading-agent",
        alias="LANGCHAIN_PROJECT"
    )
    
    # Model Configuration (Groq - Fast and Free!)
    model_name: str = Field(default="mixtral-8x7b-32768", alias="MODEL_NAME")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    max_tokens: int = Field(default=8192, alias="MAX_TOKENS")
    
    # Trading Parameters
    default_risk_tolerance: Literal["low", "medium", "high"] = Field(
        default="medium",
        alias="DEFAULT_RISK_TOLERANCE"
    )
    default_investment_horizon: int = Field(default=7, alias="DEFAULT_INVESTMENT_HORIZON")
    
    # Binance API (optional - NOT USED)
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", alias="BINANCE_SECRET_KEY")
    
    # Tavily Web Search (for market sentiment and news)
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    
    # MCP (Model Context Protocol) Settings
    enable_mcp_tools: bool = Field(default=True, alias="ENABLE_MCP_TOOLS")
    mcp_tool_choice: str = Field(default="auto", alias="MCP_TOOL_CHOICE")  # auto, required, none
    mcp_toolbox_url: str = Field(default="", alias="MCP_TOOLBOX_URL")  # Optional ToolboxClient URL

    # Local Query Logging
    enable_query_logging: bool = Field(default=True, alias="ENABLE_QUERY_LOGGING")
    query_log_db_path: str = Field(default="logs/query_logs.db", alias="QUERY_LOG_DB_PATH")

    # Prediction Feedback Loop (predict -> store -> evaluate -> improve)
    enable_prediction_feedback_loop: bool = Field(default=True, alias="ENABLE_PREDICTION_FEEDBACK_LOOP")
    prediction_feedback_store_path: str = Field(default="logs/prediction_feedback.json", alias="PREDICTION_FEEDBACK_STORE_PATH")
    prediction_feedback_hold_band_pct: float = Field(default=1.5, alias="PREDICTION_FEEDBACK_HOLD_BAND_PCT")
    rl_model_path: str = Field(default="models/rl_q_table.json", alias="RL_MODEL_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Set environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
