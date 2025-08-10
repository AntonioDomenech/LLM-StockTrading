import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yfinance as yf
from datetime import datetime, timedelta

class MarketEnvironment:
    """
    Daily-step environment.
    Position model: integer number of shares (long-only by default).
    """
    def __init__(self, df: pd.DataFrame, initial_cash: float = 10000.0, allow_short: bool=False, max_position:int=1):
        assert "Close" in df.columns, "DataFrame must have a 'Close' column."
        self.df = df.copy()
        self.df.sort_index(inplace=True)
        self.idx = 0
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0
        self.allow_short = allow_short
        self.max_position = max_position
        self.equity_curve = []  # (date, equity)
        self.actions = []       # list of dicts per step
        self.done = False

    @staticmethod
    def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            raise RuntimeError(f"yfinance returned no data for {symbol} between {start} and {end}")
        return df

    @property
    def current_date(self):
        return self.df.index[self.idx]

    @property
    def current_price(self):
        return float(self.df.iloc[self.idx]["Close"])

    def step(self, action: str):
        """
        action: 'buy', 'sell', 'hold'
        Executes at current day's close and advances one day.
        """
        if self.done:
            return

        price = self.current_price
        date = self.current_date

        # Execute action
        if action == "buy":
            # buy 1 unit if allowed
            if self.position < self.max_position:
                self.cash -= price
                self.position += 1
        elif action == "sell":
            # sell 1 unit if we have one (shorting optional)
            if self.position > 0:
                self.cash += price
                self.position -= 1
            elif self.allow_short:
                self.cash += price
                self.position -= 1
        # else hold

        equity = self.cash + self.position * price
        self.equity_curve.append((date, equity))
        self.actions.append({"date": date, "price": price, "action": action, "position": self.position, "equity": equity})

        self.idx += 1
        if self.idx >= len(self.df.index):
            self.done = True

    def reset_portfolio(self):
        self.cash = self.initial_cash
        self.position = 0
        self.equity_curve.clear()
        self.actions.clear()
        self.idx = 0
        self.done = False
