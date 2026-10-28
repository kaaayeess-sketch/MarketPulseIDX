# MarketPulseIDX

Streamlit dashboard for BBCA.JK, UNVR.JK, TLKM.JK, and PGAS.JK on the Indonesia Stock Exchange.

## Features

- Daily adjusted OHLCV prices from Yahoo Finance, with local CSV fallbacks in `data/`.
- Technical indicators and a rule-based technical score.
- Yahoo Finance fundamentals and recent Google News headline sentiment.
- Decision Support with adjustable score weights and risk profiles, comparative risk measures, key reference levels, stock ranking, and a technical-only three-year backtest.

## Run locally

1. Install Python 3.9 or later.
2. Clone this repository and open a terminal in the project directory.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed by Streamlit in your browser.

The sentiment model is downloaded by Hugging Face when first used. If it cannot load, the app falls back to its built-in keyword lexicon. Price and fundamental requests use Yahoo Finance; headlines use Google News RSS. Daily data is cached for 24 hours.

## Notes

- Fundamental metrics may be unavailable for some IDX tickers; missing inputs are omitted and remaining score weights are renormalized.
- The backtest uses only technical history because the sentiment history is short. It applies the previous session's signal to the next session's return and ignores transaction costs and slippage.
- Support, resistance, stance labels, and backtest results are analytical references, not forecasts or instructions.

For educational and research purposes only, not investment advice. Data refreshed daily, not real-time. Past performance does not guarantee future results.