# FINMEM Streamlit — Clean Refactor (Daily Trading Agent)

A production-ready, simplified refactor of a FinMem-style daily trading agent with:

- **Personas**: Secure / Balanced / Risk (dropdown)
- **Train/Test** modes with **full evaluation loop**
- **Metrics**: cumulative return, Sharpe, max drawdown, #trades
- **Equity vs. Price** plots + actions table
- **News sources**: Auto / NewsAPI / Alpaca (fallback to none)
- **Hierarchical memory** scaffolding (short/mid/long + reflections)
- **OpenAI LLM wrapper** with robust fallback (Responses → Chat Completions)
- **Windows-friendly dependencies** (no hard FAISS requirement)
- Clear logs & checkpoints under `data/runs/<timestamp>/`

> **No secrets are committed.** Provide your keys via `.env` (see `.env.template`).

---

## Quickstart

1) **Install** (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

2) **Set environment** (create `.env` from template):
```bash
cp .env.template .env
# edit .env to add your keys (OPENAI_API_KEY, NEWSAPI_KEY, ALPACA_API_KEY/SECRET)
```

3) **Run the app**:
```bash
streamlit run app.py
```

---

## Notes

- **Data source**: Default price data via `yfinance` (no API keys). If you already have pickled env data, you can load it from the UI.
- **Personas**: Map to different risk appetites and prompt styles. You can still edit the system prompt if you wish.
- **News**: If you select **Auto**, we use **NewsAPI** when `NEWSAPI_KEY` is present; otherwise, try **Alpaca** if keys exist; else skip news.
- **Memory**: Uses OpenAI embeddings when available. If not, a deterministic hash-based local embedding kicks in (lower quality but functional).
- **LLM**: You can type any OpenAI model name (e.g., `gpt-4.1`, `o3`, `gpt-4o-mini`). Wrapper attempts `Responses` API and falls back to Chat Completions automatically.

---

## Project Structure

```
finmem_streamlit_clean/
├─ app.py
├─ requirements.txt
├─ .env.template
├─ README.md
├─ finmem_core/
│  ├─ __init__.py
│  ├─ agent.py
│  ├─ chat.py
│  ├─ config.py
│  ├─ environment.py
│  ├─ memorydb.py
│  ├─ metrics.py
│  ├─ news.py
│  ├─ plot.py
│  ├─ prompts.py
│  └─ utils.py
└─ data/
   ├─ runs/        # created automatically
   └─ cache/       # optional caches
```

---

## What’s Improved vs. Your Previous Iteration

- Proper **Test** loop (iterate until done; compute metrics, plot, and table).
- **Persona selector** with three presets and consistent mapping.
- Optional **news source selector** (Auto/NewsAPI/Alpaca/None) + article list.
- **Config warnings surfaced** in UI when present.
- **Duplicate utilities removed**; a single `utils.py` provides `ensure_dir`/`stamp_dir`.
- OS-friendly **requirements** (no forced `faiss-cpu`).
- **Logging and checkpoints** are saved and linked from the UI.

---

## Disclaimers

- This repo is designed for clarity and stability. It is **not** a perfect reproduction of every research nuance of the original FinMem paper; rather, it keeps the spirit (memory hierarchy + LLM decisioning + personas) while being practical.
- For real trading, add execution logic, slippage/fees, risk controls, and compliance checks.

Enjoy!
