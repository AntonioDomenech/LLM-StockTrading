from pydantic import BaseModel, Field, validator
from typing import Optional, List

class AppConfig(BaseModel):
    symbol: str = Field("AAPL", description="Ticker symbol")
    start_date: str = Field("2022-01-01")
    end_date: str = Field("2023-01-01")
    mode: str = Field("Train", description="Train or Test")
    persona: str = Field("Balanced", description="Secure | Balanced | Risk")
    look_back_window: int = Field(30, ge=1, le=365)
    k_memory: int = Field(8, ge=1, le=64)
    model: str = Field("gpt-4o-mini", description="OpenAI model")
    embed_model: str = Field("text-embedding-3-small")
    news_source: str = Field("Auto", description="Auto | NewsAPI | Alpaca | None")
    initial_cash: float = Field(10000.0, ge=100.0)
    max_position: int = Field(1, ge=0, le=10)
    allow_short: bool = Field(False)
    random_seed: int = Field(42)

    _normalized_warnings: list = []

    @validator("persona")
    def valid_persona(cls, v):
        allowed = {"Secure", "Balanced", "Risk"}
        if v not in allowed:
            raise ValueError(f"persona must be one of {allowed}")
        return v

    @validator("mode")
    def valid_mode(cls, v):
        allowed = {"Train", "Test"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

def normalize_and_validate_cfg(cfg: "AppConfig") -> "AppConfig":
    warnings = []
    if cfg.look_back_window < 5:
        warnings.append("look_back_window < 5 may be too small to be informative.")
    if cfg.k_memory < 4:
        warnings.append("k_memory < 4 may reduce retrieval quality.")
    cfg._normalized_warnings = warnings
    return cfg
