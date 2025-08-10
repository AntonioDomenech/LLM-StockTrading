from typing import List, Dict, Optional
import os, requests
from datetime import datetime, timedelta

def _have(k: str) -> bool:
    return bool(os.getenv(k, "").strip())

def fetch_news(symbol: str, date_str: str, source_preference: str="Auto", limit: int=10) -> List[Dict]:
    """
    Returns a list of {title, published_at, url, source}
    date_str is ISO (YYYY-MM-DD); we fetch +/- 1 day window to ensure coverage.
    """
    if source_preference not in {"Auto","NewsAPI","Alpaca","None"}:
        source_preference = "Auto"

    if source_preference == "None":
        return []

    if source_preference == "Auto":
        if _have("NEWSAPI_KEY"):
            return _newsapi(symbol, date_str, limit)
        elif _have("ALPACA_API_KEY") and _have("ALPACA_SECRET_KEY"):
            return _alpaca_news(symbol, date_str, limit)
        else:
            return []
    elif source_preference == "NewsAPI":
        if _have("NEWSAPI_KEY"):
            return _newsapi(symbol, date_str, limit)
        return []
    elif source_preference == "Alpaca":
        if _have("ALPACA_API_KEY") and _have("ALPACA_SECRET_KEY"):
            return _alpaca_news(symbol, date_str, limit)
        return []
    return []

def _newsapi(symbol: str, date_str: str, limit: int) -> List[Dict]:
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        return []
    # Search company news vaguely by ticker
    base = "https://newsapi.org/v2/everything"
    from_date = date_str
    to_date = date_str
    params = {
        "q": symbol,
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "pageSize": str(limit),
        "language": "en",
        "apiKey": api_key
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        out = []
        for a in j.get("articles", []):
            out.append({
                "title": a.get("title"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name", "NewsAPI")
            })
        return out
    except Exception:
        return []

def _alpaca_news(symbol: str, date_str: str, limit: int) -> List[Dict]:
    key = os.getenv("ALPACA_API_KEY", "")
    sec = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not sec:
        return []
    # Alpaca News API v2
    base = "https://data.alpaca.markets/v1beta1/news"
    # Pull the date only
    start = date_str + "T00:00:00Z"
    end = date_str + "T23:59:59Z"
    params = {
        "symbols": symbol,
        "start": start,
        "end": end,
        "limit": str(limit)
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    try:
        r = requests.get(base, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        j = r.json()
        out = []
        for a in j.get("news", []):
            out.append({
                "title": a.get("headline"),
                "published_at": a.get("created_at"),
                "url": a.get("url"),
                "source": a.get("source", "Alpaca")
            })
        return out
    except Exception:
        return []
