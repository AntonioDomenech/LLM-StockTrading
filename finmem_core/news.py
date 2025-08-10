import os, requests, datetime as dt, time
from typing import List, Dict, Tuple
def _norm(items):
    out = []
    for a in items:
        src = a.get("source")
        if isinstance(src, dict): src = src.get("name")
        out.append({"source": src or a.get("author"),
            "title": a.get("title") or a.get("headline"),
            "description": a.get("description") or a.get("summary"),
            "url": a.get("url"),
            "published_at": a.get("publishedAt") or a.get("updated_at") or a.get("created_at") or a.get("timestamp"),
            "content": a.get("content"),})
    return out
def fetch_news_newsapi(symbol: str, start: dt.datetime, end: dt.datetime, api_key: str) -> List[Dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": f'"{symbol}" OR {symbol}', "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "language": "en", "sortBy": "publishedAt", "pageSize": 100, "apiKey": api_key}
    out, page = [], 1
    while True:
        try:
            params["page"] = page
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (401,402,403,426,429) or r.status_code>=500: return []
            r.raise_for_status(); data = r.json()
        except requests.RequestException:
            return []
        out.extend(data.get("articles", []))
        if len(data.get("articles", [])) < 100 or page > 5: break
        page += 1; time.sleep(0.3)
    return _norm(out)
def fetch_news_alpaca(symbol: str, start: dt.datetime, end: dt.datetime, key: str, secret: str, base_url: str) -> List[Dict]:
    url = base_url.rstrip("/") + "/v1beta1/news"
    headers = {"Apca-Api-Key-Id": key, "Apca-Api-Secret-Key": secret}
    params = {"symbols": symbol, "start": start.isoformat()+"Z", "end": end.isoformat()+"Z", "limit": 50, "sort": "desc"}
    out = []; page_token = None
    while True:
        try:
            if page_token: params["page_token"] = page_token
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code in (401,403) or r.status_code>=500: return []
            r.raise_for_status(); data = r.json()
        except requests.RequestException:
            return []
        items = data.get("news", data.get("data", [])); out.extend(items)
        page_token = data.get("next_page_token")
        if not page_token: break
        time.sleep(0.2)
    return _norm(out)
def fetch_news(symbol: str, start: dt.datetime, end: dt.datetime, source_preference: str="auto") -> Tuple[str, List[Dict]]:
    newsapi_key = os.getenv("NEWSAPI_KEY")
    alpaca_key = os.getenv("ALPACA_API_KEY"); alpaca_secret = os.getenv("ALPACA_API_SECRET")
    alpaca_base = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")
    def try_newsapi(): return fetch_news_newsapi(symbol, start, end, newsapi_key) if newsapi_key else []
    def try_alpaca(): return fetch_news_alpaca(symbol, start, end, alpaca_key, alpaca_secret, alpaca_base) if alpaca_key and alpaca_secret else []
    pref = (source_preference or "auto").lower()
    if pref == "newsapi":
        items = try_newsapi(); 
        if items: return "newsapi", items
        items = try_alpaca(); return ("alpaca", items) if items else ("none", [])
    if pref == "alpaca":
        items = try_alpaca(); 
        if items: return "alpaca", items
        items = try_newsapi(); return ("newsapi", items) if items else ("none", [])
    items = try_newsapi(); 
    if items: return "newsapi", items
    items = try_alpaca(); 
    if items: return "alpaca", items
    return "none", []
