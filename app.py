import os, io, json
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from finmem_core.config import AppConfig, normalize_and_validate_cfg
from finmem_core.environment import MarketEnvironment
from finmem_core.news import fetch_news
from finmem_core.memorydb import BrainDB, make_embedder_openai_or_hash
from finmem_core.agent import LLMAgent
from finmem_core.metrics import equity_stats
from finmem_core.plot import plot_price_and_equity
from finmem_core.utils import ensure_dir, stamp_dir, save_json, print_warnings

load_dotenv()

st.set_page_config(page_title="FINMEM Streamlit Clean", layout="wide")
st.title("FINMEM — Clean Refactor (Daily Trading Agent)")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Symbol", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2022-01-01")).isoformat()
    with col2:
        end_date = st.date_input("End", value=pd.to_datetime("2023-01-01")).isoformat()

    mode = st.selectbox("Mode", ["Train", "Test"], index=0)
    persona = st.selectbox("Persona", ["Secure","Balanced","Risk"], index=1)
    look_back_window = st.slider("Look-back window (days)", 5, 120, 30)
    k_memory = st.slider("K memory retrieved", 4, 32, 8)
    model_name = st.text_input("OpenAI model", value=os.getenv("DEFAULT_MODEL","gpt-4o-mini"))
    embed_model = st.text_input("Embedding model", value=os.getenv("DEFAULT_EMBED_MODEL","text-embedding-3-small"))
    news_source = st.selectbox("News source", ["Auto","NewsAPI","Alpaca","None"], index=0)
    initial_cash = st.number_input("Initial cash", min_value=100.0, value=10000.0, step=100.0)
    max_position = st.number_input("Max position (units)", min_value=0, value=1, step=1)
    allow_short = st.checkbox("Allow short", value=False)

    st.divider()
    uploaded_env = st.file_uploader("Optional: load pickled price DataFrame", type=["pkl"])
    run_btn = st.button("Run")

# Build config
cfg = AppConfig(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    mode=mode,
    persona=persona,
    look_back_window=look_back_window,
    k_memory=k_memory,
    model=model_name,
    embed_model=embed_model,
    news_source=news_source,
    initial_cash=float(initial_cash),
    max_position=int(max_position),
    allow_short=bool(allow_short)
)
cfg = normalize_and_validate_cfg(cfg)

warn_text = print_warnings(getattr(cfg, "_normalized_warnings", []))
if warn_text:
    st.sidebar.warning(warn_text)

# Load prices
@st.cache_data(show_spinner=False)
def _load_prices_cached(symbol, start, end):
    return MarketEnvironment.load_prices(symbol, start, end)

if run_btn:
    st.toast("Starting run…")

    # Load price DF either from upload or yfinance
    if uploaded_env is not None:
        df = pd.read_pickle(uploaded_env)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    else:
        df = _load_prices_cached(cfg.symbol, cfg.start_date, cfg.end_date)

    # Build environment
    env = MarketEnvironment(df, initial_cash=cfg.initial_cash, allow_short=cfg.allow_short, max_position=cfg.max_position)

    # Memory + embedder
    import os as _os
    from openai import OpenAI as _OpenAI
    client = None
    try:
        if _os.getenv("OPENAI_API_KEY"):
            client = _OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None
    embed_fn = make_embedder_openai_or_hash(client, cfg.embed_model)
    memdb = BrainDB(embed_fn)

    # Agent
    agent = LLMAgent(cfg.persona, cfg.model)

    # Prepare output dirs
    run_dir = stamp_dir("data/runs")
    save_json(os.path.join(run_dir, "config.json"), cfg.model_dump())

    # Iterate
    st.subheader("Run Progress")
    progress = st.progress(0)
    total = len(df.index)
    table_rows = []
    news_cache = {}

    for i, current_date in enumerate(df.index):
        # Price window
        lb = max(0, i - cfg.look_back_window + 1)
        pw = [(d.strftime("%Y-%m-%d"), float(df.iloc[j]["Close"])) for j, d in enumerate(df.index[lb:i+1], start=lb)]

        # Fetch news once per date
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in news_cache:
            news_cache[date_str] = fetch_news(cfg.symbol, date_str, cfg.news_source, limit=8)
        news_list = news_cache[date_str]

        # Retrieve memories with last rationale + prices as query
        query = f"{cfg.symbol} {date_str} recent trend {pw[-1][1]:.2f}"
        retrieved = memdb.retrieve(query, k=cfg.k_memory)

        # Get decision
        out = agent.step(cfg.symbol, date_str, pw, news_list, retrieved)
        action = out["action"]
        # Step env
        env.step(action)

        # Update memories lightly: store rationale in short, summaries in reflections
        if out.get("rationale"):
            memdb.add("short", f"{date_str} action={action} rationale={out['rationale'][:500]}", meta={"date":date_str})
        # Periodically store reflection
        if i % max(1, cfg.look_back_window//5) == 0 and i > 0:
            memdb.add("reflections", f"Reflection up to {date_str}: trending equity={env.equity_curve[-1][1]:.2f}", meta={"date":date_str})

        # Accumulate table row
        env_row = env.actions[-1]
        row = {
            "date": env_row["date"],
            "price": env_row["price"],
            "action": env_row["action"],
            "position": env_row["position"],
            "equity": env_row["equity"],
            "confidence": out.get("confidence", 0.0),
            "rationale": out.get("rationale","")[:300]
        }
        table_rows.append(row)

        if i % 5 == 0 or i == total-1:
            progress.progress(int((i+1)/total * 100))

    # Results
    actions_df = pd.DataFrame(table_rows)
    actions_df.sort_values("date", inplace=True)
    st.success("Run complete ✅")

    # Metrics
    stats = equity_stats(actions_df, cfg.initial_cash)
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Cumulative Return", f"{stats['cum_return']*100:.2f}%")
    colB.metric("Sharpe (approx)", f"{stats['sharpe']:.2f}")
    colC.metric("Max Drawdown", f"{stats['max_dd']*100:.2f}%")
    colD.metric("# Trades", f"{stats['trades']}")

    # Plot
    buf = plot_price_and_equity(actions_df["date"].tolist(), actions_df["price"].tolist(), actions_df["equity"].tolist())
    st.image(buf, caption="Price vs. Equity")

    # Actions table
    st.subheader("Actions")
    st.dataframe(actions_df, use_container_width=True, hide_index=True)

    # News used: last day (expandable)
    st.subheader("News (example: last day)")
    last_day = actions_df["date"].iloc[-1]
    last_day_str = last_day.strftime("%Y-%m-%d")
    used = news_cache.get(last_day_str, [])
    if used:
        with st.expander(f"Articles on {last_day_str} ({len(used)})"):
            for n in used:
                st.markdown(f"- **{n.get('title','')}** — *{n.get('source','')}*  \n{n.get('url','')}")

    # Save artifacts
    actions_df.to_csv(os.path.join(run_dir, "actions.csv"), index=False)
    st.write("Artifacts saved to:", run_dir)
    st.code(os.path.abspath(run_dir), language="bash")
