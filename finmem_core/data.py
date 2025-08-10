import pandas as pd, yfinance as yf, datetime as dt
def download_ohlcv(symbol: str, start: dt.datetime, end: dt.datetime, interval: str="1d") -> pd.DataFrame:
    end_inclusive = end + dt.timedelta(days=1)
    df = yf.download(symbol, start=start, end=end_inclusive, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=lambda c: c.title() if isinstance(c, str) else c)
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' in df.columns:
        df = df.dropna(subset=['Close']).copy()
    return df
def build_env(symbol: str, df: pd.DataFrame, articles: list) -> dict:
    import pandas as pd
    by_date = {}
    for a in (articles or []):
        ts = a.get("published_at") or a.get("publishedAt")
        if not ts: continue
        d = pd.to_datetime(ts, errors="coerce")
        if pd.isna(d): continue
        by_date.setdefault(d.date(), []).append((a.get("title") or a.get("description") or "")[:500])
    env = {}
    for ts, row in df.iterrows():
        dd = ts.date()
        price = float(row.get("Close", row.iloc[0]))
        env[dd] = {"price": {symbol: price}, "filing_k": {}, "filing_q": {}, "news": {symbol: by_date.get(dd, [])}}
    return dict(sorted(env.items(), key=lambda kv: kv[0]))
