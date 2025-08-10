import numpy as np
import pandas as pd

def equity_stats(actions_df: pd.DataFrame, initial_cash: float) -> dict:
    if actions_df.empty:
        return {"cum_return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0}
    equity = actions_df["equity"].values.astype(float)
    rets = np.diff(equity) / equity[:-1]
    cum_return = (equity[-1] / initial_cash) - 1.0
    sharpe = 0.0
    if len(rets) > 2 and np.std(rets) > 0:
        sharpe = np.sqrt(252) * np.mean(rets) / (np.std(rets) + 1e-12)
    dd = drawdown(equity)
    trades = int((actions_df["action"] != "hold").sum())
    return {"cum_return": float(cum_return), "sharpe": float(sharpe), "max_dd": float(dd), "trades": trades}

def drawdown(curve):
    peak = -1e18
    maxdd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-12)
        if dd > maxdd:
            maxdd = dd
    return float(maxdd)
