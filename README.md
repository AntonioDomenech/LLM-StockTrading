# FINMEM Streamlit Plus 4 (stable)
- Dashboard final con KPIs claros (Agente vs Buy&Hold).
- Test con memoria **solo lectura** + **noticias efímeras** del día (no persistentes).
- Carga automática del último checkpoint de Train o el que elijas.
- Sin dependencias frágiles: **sin guardrails**, **sin langchain** (OpenAI SDK directo).
- 3 personalidades: balanced / risk-seeking / risk-averse.

## Uso
```
pip install -r requirements.txt
cp .env.example .env   # añade OPENAI_API_KEY y opcionalmente NEWSAPI/ALPACA
streamlit run app.py
```


---

## Quickstart (updated)

```bash
cd finmem_streamlit_plus4_fixed
pip install -r requirements.txt
streamlit run app.py
```

**Notes**

- If you select an OpenAI chat model as `tokenization_model_name` (e.g., `gpt-4.1`), the app now **skips** Hugging Face tokenization and uses a safe fallback to avoid crashes.
- FAISS is optional. If `faiss` isn't available on your system, the app now falls back to a pure-NumPy cosine index. It's slower but works out-of-the-box.
- News provider selection: if both Alpaca and NewsAPI keys are missing, news will be disabled. Set `NEWSAPI_KEY` to auto-use NewsAPI.