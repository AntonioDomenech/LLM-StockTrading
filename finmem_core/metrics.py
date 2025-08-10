import pandas as pd, numpy as np
from typing import Any, Dict, Optional
def extract_portfolio_timeseries(portfolio: Any) -> pd.DataFrame:
    candidates = [n for n in dir(portfolio) if not n.startswith("_")]
    for name in candidates:
        try: val = getattr(portfolio, name)
        except Exception: continue
        if "polars" in type(val).__module__:
            try: return _normalize_cols(val.to_pandas())
            except Exception: pass
    for name in ["df","data","frame","portfolio_df"]:
        if hasattr(portfolio, name):
            try:
                df = getattr(portfolio, name)
                if hasattr(df, "to_pandas"): df = df.to_pandas()
                return _normalize_cols(df)
            except Exception: pass
    fields = {}
    for key in ["date_series","record_series","share_series","price_series","holding_series"]:
        if hasattr(portfolio, key): fields[key] = getattr(portfolio, key)
    if fields:
        df = pd.DataFrame({k.replace("_series",""): list(v) for k,v in fields.items()})
        return _normalize_cols(df)
    return pd.DataFrame(columns=["date","price","share","record"])
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame()
    for k in ["date","dates","day"]:
        if k in cols: out["date"] = pd.to_datetime(df[cols[k]]).dt.date; break
    for k in ["price","close","adj close","adj_close"]:
        if k in cols: out["price"] = df[cols[k]]; break
    for k in ["share","position","holding","shares"]:
        if k in cols: out["share"] = df[cols[k]]; break
    for k in ["record","equity","cum_return","cumulative_return","pnl"]:
        if k in cols: out["record"] = df[cols[k]]; break
    return out
def compute_buy_hold(df_price: pd.DataFrame) -> pd.Series:
    s = df_price["Close"].astype(float).dropna(); base = float(s.iloc[0]); return s / base
def compute_agent_equity_from_portfolio(port_df: pd.DataFrame) -> Optional[pd.Series]:
    if "record" in port_df.columns and len(port_df["record"])>0:
        s = pd.Series(port_df["record"].values, index=pd.to_datetime(port_df["date"]))
        if s.max() <= 3 and s.min() >= -1 and (s.abs().mean() < 0.2):
            return (1 + s).cumprod()
        return s
    return None
def compute_drawdown(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax(); return equity/roll_max - 1.0
def cagr(equity: pd.Series) -> float:
    if len(equity) < 2: return 0.0
    days = (equity.index[-1] - equity.index[0]).days or 1; years = days/365.0
    final = float(equity.iloc[-1]); return final**(1/years) - 1 if final>0 and years>0 else 0.0
def ann_vol(equity: pd.Series) -> float:
    ret = equity.pct_change().dropna(); return float(ret.std() * (252**0.5))
def sharpe(equity: pd.Series, rf_annual: float=0.0) -> float:
    ret = equity.pct_change().dropna(); 
    if ret.empty: return 0.0
    excess = ret - (rf_annual/252.0); vol = ret.std()
    return float(excess.mean()/vol * (252**0.5)) if vol>0 else 0.0
def max_drawdown(equity: pd.Series) -> float:
    return float(compute_drawdown(equity).min())
def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity)); return cagr(equity)/dd if dd>0 else 0.0
def exposure_pct(port_df: pd.DataFrame) -> float:
    if "share" not in port_df.columns or len(port_df)==0: return 0.0
    invested = (port_df["share"]!=0).sum(); return 100.0 * invested / len(port_df)
def estimate_trades(port_df: pd.DataFrame) -> Dict[str, float]:
    if "share" not in port_df.columns or "price" not in port_df.columns: 
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_gain": 0.0, "avg_loss": 0.0}
    shares = port_df["share"].fillna(0).values; prices = port_df["price"].astype(float).fillna(method="ffill").values
    trades = []; pos = 0; entry_price = None
    for i in range(1, len(shares)):
        if pos==0 and shares[i]!=0: pos = shares[i]; entry_price = prices[i]
        elif pos!=0 and shares[i]==0:
            direction = 1 if pos>0 else -1; pnl = direction * (prices[i] - entry_price) / entry_price
            trades.append(pnl); pos = 0; entry_price = None
        elif pos!=0 and (shares[i]>0) != (pos>0):
            direction = 1 if pos>0 else -1; pnl = direction * (prices[i] - entry_price) / entry_price
            trades.append(pnl); pos = shares[i]; entry_price = prices[i]
    if not trades: return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_gain": 0.0, "avg_loss": 0.0}
    gains = [t for t in trades if t>0]; losses = [-t for t in trades if t<0]
    win_rate = 100.0 * len(gains) / len(trades)
    profit_factor = (sum(gains)/sum(losses)) if sum(losses)>0 else float('inf')
    import numpy as np
    avg_gain = float(np.mean(gains)) if gains else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return {"trades": len(trades), "win_rate": win_rate, "profit_factor": profit_factor, "avg_gain": avg_gain, "avg_loss": avg_loss}
