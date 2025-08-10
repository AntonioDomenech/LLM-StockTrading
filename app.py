
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st


# --- Load environment variables from .env early ---
try:
    from dotenv import load_dotenv, find_dotenv
    # Try common locations: CWD, script dir
    _ = load_dotenv(find_dotenv(usecwd=True), override=False)
    _ = load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    _ = load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
except Exception as _e:
    pass

# ---- Core imports from your project ----
from finmem_core.agent import LLMAgent
from finmem_core.run_type import RunMode
from finmem_core.environment import MarketEnvironment
from finmem_core.config_utils import normalize_and_validate_cfg
from finmem_core.checkpoint_utils import resolve_test_checkpoint, list_run_candidates

APP_TITLE = "FinMem — Training & Evaluation"
RUNS_BASE = "data/runs"


# =============== Small utilities ===============

def stamp_dir(base: str) -> str:
    """Create a timestamped directory under base and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


# =============== Sidebar (clean layout) ===============

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Run")
    run_mode = st.radio("Mode", options=["Train", "Test"], horizontal=True, index=0)
    run_name = st.text_input("Run name", value="my_run")
    symbol = st.text_input("Trading symbol", value="AAPL")

    st.divider()
    st.header("Dates")
    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start", value=date(2022, 1, 3))
    with colB:
        end_date = st.date_input("End", value=date(2023, 1, 3))

    st.divider()
    st.header("Agent & Memory")
    agent_name = st.text_input("Agent name", value="agent_1")
    look_back = st.number_input("Look-back window (days)", min_value=1, max_value=365, value=30, step=1)
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=10, value=5, step=1)
    character_string = st.text_area("Character / System prompt", value="You are a helpful trading assistant.")

    st.divider()
    st.header("Embedding")
    emb_model = st.selectbox(
        "Embedding model",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        index=0,
    )
    chunk_chars = st.number_input("Chunk char size", min_value=256, max_value=16000, value=6000, step=256)

    st.divider()
    st.header("OpenAI / Chat")
    chat_model = st.text_input("Chat model", value="gpt-4o-mini")
    max_tokens = st.number_input("Max tokens per reply", min_value=64, max_value=8192, value=512, step=32)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)

    st.divider()
    run_btn = st.button("Run 🚀", use_container_width=True)


# =============== Build the config dict and normalize ===============

def build_cfg() -> Dict[str, Any]:
    cfg = {
        "general": {
            "agent_name": agent_name,
            "trading_symbol": symbol,
            "character_string": character_string,
            "look_back_window_size": int(look_back),
            "top_k": int(top_k),
        },
        "embedding": {
            "model_name": emb_model,
            "chunk_char_size": int(chunk_chars),
        },
        "chat": {
        "end_point": "openai",
        "model": chat_model,
        "system_message": character_string,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature)
    },
        # default thresholds per layer; config_utils will backfill if any missing
        "short": {},
        "mid": {},
        "long": {},
        "reflection": {},
    }
    return cfg

cfg = build_cfg()
cfg = normalize_and_validate_cfg(cfg)




# =============== TRAIN ===============



def _init_environment_robust(symbol: str, start_date: date, end_date: date) -> MarketEnvironment:
    """
    Build a MarketEnvironment reliably by generating the required `env_data_pkl` dict
    from market data and news, then instantiating with the full signature.
    This avoids signature drift across repo variants.
    """
    import datetime as dt
    from finmem_core import data as data_utils
    from finmem_core import news as news_utils

    # 1) Download OHLCV (inclusive end)
    start_dt = dt.datetime.combine(start_date, dt.time.min)
    end_dt = dt.datetime.combine(end_date, dt.time.max)
    df = data_utils.download_ohlcv(symbol, start_dt, end_dt, interval="1d")
    if df is None or len(df) == 0:
        raise ValueError(f"No OHLCV data returned for {symbol} between {start_date} and {end_date}")

    # 2) Fetch news from preferred source (auto chooses NEWSAPI if key present, else Alpaca if keys present)
    source_used, articles = news_utils.fetch_news(symbol, start_dt, end_dt, source_preference="auto")
    # Non-fatal if no news; we still build environment using prices only
    articles = articles or []

    # 3) Build env_data dictionary
    env_dict = data_utils.build_env(symbol, df, articles)

    # 4) Instantiate environment with the full, strict signature
    env = MarketEnvironment(
        env_data_pkl=env_dict,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )

    # 5) Small UI hint about news source used (optional)
    try:
        st.caption(f"News source: **{source_used}** · Articles fetched: {len(articles)}")
    except Exception:
        pass
    return env


def _train_once() -> Tuple[str, str]:
    # Prepare output folders
    out_dir = stamp_dir(RUNS_BASE)
    out_dir_named = os.path.join(out_dir, run_name)
    final_dir = os.path.join(out_dir_named, "final")
    ensure_dir(final_dir)

    st.subheader("Training")
    with st.status("Initializing...", expanded=False) as status:
        # Build env
        env = _init_environment_robust(symbol, start_date, end_date)

        # Build agent
        the_agent = LLMAgent.from_config(cfg)

        # Steps
        keys = getattr(env, "keys", [])
        sim_len = getattr(env, "simulation_length", _safe_len(keys) - 1)
        steps = max(int(sim_len), 1)

        progress = st.progress(0.0, text=f"Training... 0 / {steps}")
        completed = 0

        status.update(label="Running training loop...", state="running")
        for _ in range(steps):
            mi = env.step()
            # If env returns (...., done: bool), break when done
            if isinstance(mi, tuple) and len(mi) > 0 and isinstance(mi[-1], bool) and mi[-1]:
                break
            the_agent.step(market_info=mi, run_mode=RunMode.Train)
            completed += 1
            progress.progress(min(completed / steps, 1.0), text=f"Training... {completed} / {steps}")

        # Save
        status.update(label="Saving checkpoints...", state="running")
        save_dir = final_dir
        env.save_checkpoint(path=os.path.join(save_dir, "env"), force=True)
        the_agent.save_checkpoint(path=save_dir, force=True)

        status.update(label=f"Finished Train. Results saved under {save_dir}", state="complete")

    return (os.path.join(save_dir, "env"), save_dir)


# =============== TEST ===============

def _test_once() -> None:
    st.subheader("Test / Evaluation")
    # Show a quick list of latest runs for clarity
    cands = list_run_candidates(RUNS_BASE)
    if len(cands) > 0:
        st.caption("Latest checkpoints:")
        for i, c in enumerate(cands[:5], start=1):
            st.write(f"{i}. final: `{c['final']}`")

    try:
        env_dir, agent_dir = resolve_test_checkpoint(RUNS_BASE)
    except FileNotFoundError:
        st.error("No TRAIN checkpoint found. Run a Train first.")
        return

    st.info(f"Loading checkpoint:\n- env: `{env_dir}`\n- agent: `{agent_dir}`")
    env = MarketEnvironment.load_checkpoint(env_dir)
    the_agent = LLMAgent.load_checkpoint(agent_dir)

    # Simple sanity step (optional)
try:
    # 1) reset pointer if available
    if hasattr(env, "reset"):
        try:
            env.reset()
        except Exception:
            pass

    # 2) get one step robustly
    mi_raw = env.step()
    mi, done = None, False
    if isinstance(mi_raw, tuple) and mi_raw:
        # assume (payload, ..., done: bool) OR (payload, done)
        if isinstance(mi_raw[-1], bool):
            done = mi_raw[-1]
            mi = mi_raw[0] if len(mi_raw) >= 2 else None
        else:
            mi = mi_raw[0]
    else:
        mi = mi_raw

    # 3) run agent only if we actually have a market_info and not done
    if mi is None or done:
        st.info("Loaded checkpoint, but no testable step is available (end-of-episode). Skipping smoke test.")
    else:
        the_agent.step(market_info=mi, run_mode=RunMode.Test)
        st.success("Loaded checkpoint and executed one TEST step successfully.")
except Exception as e:
    st.warning(f"Loaded checkpoint, but a TEST step raised: {e!r}. Continuing to dashboard...")



# =============== Entry point ===============

if run_btn:
    if run_mode == "Train":
        _train_once()
        st.toast("Training completed and saved ✅")
    else:
        _test_once()