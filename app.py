from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


STOCKS = {
    "BBCA.JK": "Bank Central Asia",
    "UNVR.JK": "Unilever Indonesia",
    "TLKM.JK": "Telkom Indonesia",
    "PGAS.JK": "Perusahaan Gas Negara",
}
DATA_DIR = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parent
NEWS_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
NEWS_DAYS = 30
DEFAULT_WEIGHTS = {"technical": 50, "sentiment": 30, "fundamental": 20}
WEIGHT_MIN = 0
WEIGHT_MAX = 100
WEIGHT_STEP = 5
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
TECHNICAL_TREND_SMA_WINDOW = 200
SMA_WINDOWS = (20, 50, TECHNICAL_TREND_SMA_WINDOW)
SENTIMENT_NEUTRAL_BAND = 0.15
RISK_TERCILE_CUTOFFS = (1 / 3, 2 / 3)
RISK_PROFILES = ("Conservative", "Moderate", "Aggressive")
STANCE_THRESHOLDS = {
    "Conservative": {"accumulate": 0.65, "hold": 0.20, "reduce": -0.20},
    "Moderate": {"accumulate": 0.45, "hold": 0.10, "reduce": -0.25},
    "Aggressive": {"accumulate": 0.30, "hold": 0.00, "reduce": -0.30},
}
FUNDAMENTAL_RULES = {
    "roe_strong": 0.15,
    "roe_moderate": 0.08,
    "der_low": 100.0,
    "der_high": 200.0,
}
RISK_LOOKBACK_YEARS = 1
RISK_DRAWDOWN_YEARS = (1, 3)
SWING_LOOKBACK_DAYS = 60
ATR_STOP_MULTIPLIER = 2.0
BACKTEST_YEARS = 3
BACKTEST_TECHNICAL_THRESHOLD = 0.3
TRADING_DAYS_PER_YEAR = 252
STANCE_COLORS = {
    "Accumulate": "#21d07a",
    "Hold": "#94a3b8",
    "Reduce": "#f59e0b",
    "Avoid": "#f05260",
}
RISK_COLORS = {"Low": "#21d07a", "Medium": "#f59e0b", "High": "#f05260"}


def _column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = next((part for part in column if str(part).lower() in {
            "open", "high", "low", "close", "volume"
        }), column[0])
    return str(column).strip().lower().replace(" ", "")


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        ticker = next((value for value in frame.columns.get_level_values(0).unique()
                       if str(value).endswith(".JK")), None)
        if ticker is None:
            ticker = next((value for value in frame.columns.get_level_values(-1).unique()
                           if str(value).endswith(".JK")), None)
        if ticker is not None:
            for level in range(frame.columns.nlevels):
                if ticker in frame.columns.get_level_values(level):
                    frame = frame.xs(ticker, axis=1, level=level)
                    break
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

    frame = frame.copy()
    frame.columns = [_column_name(column) for column in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not all(name in frame.columns for name in required):
        return pd.DataFrame()
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].dropna(subset=["open", "high", "low", "close"])
    frame.index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    return frame.sort_index()


def _parse_local_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace('"', "")
    multiplier = 1
    if text.endswith("%"):
        text = text[:-1]
    if text[-1:].upper() in {"K", "M", "B"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[text[-1].upper()]
        text = text[:-1]
    text = text.replace(".", "").replace(",", ".").replace(" ", "")
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def _load_local_history(ticker: str) -> pd.DataFrame:
    local_file = DATA_DIR / f"{ticker}.csv"
    if local_file.exists():
        raw = pd.read_csv(local_file, encoding="utf-8-sig", dtype=str)
        columns = {str(column).strip().lower(): column for column in raw.columns}
        if all(key in columns for key in ("date", "open", "high", "low", "close", "volume")):
            result = raw.rename(columns={
                columns["date"]: "Date",
                columns["open"]: "Open",
                columns["high"]: "High",
                columns["low"]: "Low",
                columns["close"]: "Close",
                columns["volume"]: "Volume",
            })
        else:
            aliases = {
                "tanggal": "Date", "terakhir": "Close", "pembukaan": "Open",
                "tertinggi": "High", "terendah": "Low", "vol.": "Volume",
            }
            result = raw.rename(columns={
                column: aliases.get(str(column).strip().lower(), column)
                for column in raw.columns
            })
        date_column = next((column for column in result.columns
                            if str(column).lower() == "date"), None)
        if date_column is None:
            return pd.DataFrame()
        result = result.set_index(date_column)
        result.index = pd.to_datetime(result.index, dayfirst=True, errors="coerce")
        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column not in result.columns:
                return pd.DataFrame()
            result[column] = result[column].map(_parse_local_number)
        return _normalize_history(result)

    legacy_names = {
        "BBCA.JK": "Data Historis BBCA (1).csv",
        "UNVR.JK": "Data Historis UNVR (1).csv",
        "TLKM.JK": "Data Historis TLKM (1).csv",
        "PGAS.JK": "Data Historis PGAS (1).csv",
    }
    legacy_file = ROOT / legacy_names[ticker]
    if not legacy_file.exists():
        return pd.DataFrame()
    raw = pd.read_csv(legacy_file, encoding="utf-8-sig", dtype=str)
    aliases = {
        "tanggal": "Date", "terakhir": "Close", "pembukaan": "Open",
        "tertinggi": "High", "terendah": "Low", "vol.": "Volume",
    }
    raw = raw.rename(columns={
        column: aliases.get(str(column).strip().lower(), column)
        for column in raw.columns
    })
    if "Date" not in raw.columns:
        return pd.DataFrame()
    raw = raw.set_index("Date")
    raw.index = pd.to_datetime(raw.index, dayfirst=True, errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in raw.columns:
            return pd.DataFrame()
        raw[column] = raw[column].map(_parse_local_number)
    return _normalize_history(raw)


@st.cache_data(ttl=86400)
def load_prices(ticker: str) -> tuple[pd.DataFrame, str | None]:
    through_date = date.today()
    try:
        downloaded = yf.download(
            ticker,
            start="2019-01-01",
            end=(through_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        prices = _normalize_history(downloaded)
        if not prices.empty:
            return prices, None
        warning = "Yahoo Finance returned no price rows."
    except Exception as error:
        warning = f"Yahoo Finance price request failed: {error}"

    try:
        fallback = _load_local_history(ticker)
        if not fallback.empty:
            latest = fallback.index.max().date().isoformat()
            return fallback, f"{warning} Using the local CSV fallback (latest available date: {latest})."
        return pd.DataFrame(), f"{warning} No usable local CSV fallback was found."
    except Exception as error:
        return pd.DataFrame(), f"{warning} Local CSV fallback failed: {error}"


def add_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    for window in SMA_WINDOWS:
        data[f"SMA{window}"] = data["close"].rolling(window).mean()
    change = data["close"].diff()
    gain = change.clip(lower=0).ewm(
        alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False
    ).mean()
    loss = -change.clip(upper=0).ewm(
        alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False
    ).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - 100 / (1 + relative_strength)
    data.loc[(loss == 0) & (gain > 0), "RSI"] = 100

    fast = data["close"].ewm(span=12, adjust=False).mean()
    slow = data["close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = fast - slow
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    middle = data["close"].rolling(20).mean()
    deviation = data["close"].rolling(20).std()
    data["BB_upper"] = middle + 2 * deviation
    data["BB_lower"] = middle - 2 * deviation
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()
    return data


def technical_score(data: pd.DataFrame) -> float | None:
    if data.empty:
        return None
    latest = data.iloc[-1]
    rules: list[float] = []
    if pd.notna(latest[f"SMA{TECHNICAL_TREND_SMA_WINDOW}"]):
        rules.append(
            1.0
            if latest["close"] > latest[f"SMA{TECHNICAL_TREND_SMA_WINDOW}"]
            else -1.0
        )
    if pd.notna(latest["RSI"]):
        rules.append(
            1.0 if latest["RSI"] < RSI_OVERSOLD_THRESHOLD
            else -1.0 if latest["RSI"] > RSI_OVERBOUGHT_THRESHOLD
            else 0.0
        )
    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        rules.append(1.0 if latest["MACD"] > latest["MACD_signal"] else -1.0)
    return float(np.mean(rules)) if rules else None


def technical_score_series(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    rules = pd.DataFrame(index=data.index)
    trend_sma = data[f"SMA{TECHNICAL_TREND_SMA_WINDOW}"]
    rules[f"SMA{TECHNICAL_TREND_SMA_WINDOW}"] = np.where(
        trend_sma.notna(),
        np.where(data["close"] > trend_sma, 1.0, -1.0),
        np.nan,
    )
    rules["RSI"] = np.where(
        data["RSI"].notna(),
        np.select(
            [
                data["RSI"] < RSI_OVERSOLD_THRESHOLD,
                data["RSI"] > RSI_OVERBOUGHT_THRESHOLD,
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        np.nan,
    )
    rules["MACD"] = np.where(
        data["MACD"].notna() & data["MACD_signal"].notna(),
        np.where(data["MACD"] > data["MACD_signal"], 1.0, -1.0),
        np.nan,
    )
    return rules.mean(axis=1)


def _format_price(value: object) -> str:
    return f"{value:,.0f}" if pd.notna(value) else "N/A"


def technical_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.48, 0.14, 0.14, 0.14, 0.10],
        subplot_titles=("Price and Bollinger Bands", "Volume", "RSI (14)", "MACD (12, 26, 9)", "ATR (14)"),
    )
    figure.add_trace(
        go.Candlestick(
            x=data.index, open=data["open"], high=data["high"], low=data["low"],
            close=data["close"], name="Price",
        ),
        row=1, col=1,
    )
    sma_colors = ("#f59e0b", "#3b82f6", "#a855f7")
    for window, color in zip(SMA_WINDOWS, sma_colors):
        column = f"SMA{window}"
        figure.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line={"color": color}), row=1, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["BB_upper"], name="BB upper", line={"color": "#94a3b8", "dash": "dot", "width": 1}), row=1, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["BB_lower"], name="BB lower", line={"color": "#94a3b8", "dash": "dot", "width": 1}), row=1, col=1)
    figure.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color="#64748b"), row=2, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line={"color": "#14b8a6"}), row=3, col=1)
    for level in (RSI_OVERSOLD_THRESHOLD, RSI_OVERBOUGHT_THRESHOLD):
        figure.add_hline(y=level, line_dash="dot", line_color="#94a3b8", row=3, col=1)
    figure.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="MACD histogram", marker_color="#94a3b8"), row=4, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line={"color": "#2563eb"}), row=4, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal", line={"color": "#f97316"}), row=4, col=1)
    figure.add_trace(go.Scatter(x=data.index, y=data["ATR"], name="ATR", line={"color": "#8b5cf6"}), row=5, col=1)
    figure.update_layout(
        title=f"{ticker} — adjusted daily prices",
        height=1050,
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"t": 90, "b": 30},
    )
    figure.update_yaxes(title_text="IDR", row=1, col=1)
    figure.update_yaxes(title_text="Shares", row=2, col=1)
    figure.update_yaxes(range=[0, 100], row=3, col=1)
    return figure


@st.cache_data(ttl=86400)
def load_fundamentals(ticker: str) -> tuple[dict[str, object], str | None]:
    try:
        info = yf.Ticker(ticker).info
        return {
            "PER": info.get("trailingPE"),
            "PBV": info.get("priceToBook"),
            "ROE": info.get("returnOnEquity"),
            "DER": info.get("debtToEquity"),
            "Dividend Yield": info.get("dividendYield"),
            "Market Cap": info.get("marketCap"),
        }, None
    except Exception as error:
        return {}, f"Fundamental data could not be loaded for {ticker}: {error}"


def _format_metric(key: str, value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if key in {"ROE", "Dividend Yield"}:
        return f"{value:.2%}"
    if key == "Market Cap":
        return f"Rp {value:,.0f}"
    if key == "DER":
        return f"{value:,.2f}%"
    return f"{value:,.2f}x"


@st.cache_data(ttl=86400)
def load_news(ticker: str) -> tuple[list[dict[str, object]], list[str]]:
    headlines: list[dict[str, object]] = []
    warnings: list[str] = []
    queries = (
        (f'"{STOCKS[ticker]}" {ticker} saham when:30d', "id", "ID", "ID:id"),
        (f'"{STOCKS[ticker]}" {ticker} stock when:30d', "en-US", "US", "US:en"),
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS)
    for query, language, region, edition in queries:
        try:
            response = requests.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": language, "gl": region, "ceid": edition},
                headers={"User-Agent": "Mozilla/5.0 (compatible; IDXResearchDashboard/1.0)"},
                timeout=12,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                published = item.findtext("pubDate")
                if not title or not link or not published:
                    continue
                try:
                    published_at = datetime.strptime(
                        published, "%a, %d %b %Y %H:%M:%S %Z"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        published_at = datetime.strptime(
                            published, "%a, %d %b %Y %H:%M:%S %z"
                        ).astimezone(timezone.utc)
                    except ValueError:
                        continue
                if published_at < cutoff:
                    continue
                headlines.append({
                    "Date": published_at.date(),
                    "Published": published_at,
                    "Headline": title,
                    "Link": link,
                    "Source": urlparse(link).netloc,
                })
        except Exception as error:
            warnings.append(f"Google News request failed: {error}")
    deduplicated: dict[str, dict[str, object]] = {}
    for headline in sorted(headlines, key=lambda row: row["Published"], reverse=True):
        key = re.sub(r"\W+", "", str(headline["Headline"]).lower())
        deduplicated.setdefault(key, headline)
    return list(deduplicated.values()), warnings


@st.cache_resource
def load_sentiment_model():
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model=NEWS_MODEL, tokenizer=NEWS_MODEL)
    except Exception:
        return None


POSITIVE_WORDS = {
    "naik", "menguat", "untung", "laba", "positif", "tumbuh", "rebound",
    "bullish", "growth", "profit", "surge", "gain", "upgrade", "strong",
    "dividend", "beat", "record", "optimis", "membaik",
}
NEGATIVE_WORDS = {
    "turun", "melemah", "rugi", "negatif", "anjlok", "jatuh", "bearish",
    "loss", "decline", "fall", "drop", "downgrade", "weak", "crisis",
    "utang", "gagal", "warning", "koreksi", "tertekan",
}


def keyword_sentiment(text: str) -> tuple[str, float]:
    words = set(re.findall(r"\w+", text.lower()))
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    if score > 0:
        return "Positive", 1.0
    if score < 0:
        return "Negative", -1.0
    return "Neutral", 0.0


def _model_label(result: dict[str, object]) -> tuple[str, float]:
    label = str(result.get("label", "")).lower()
    if "positive" in label or label == "label_2":
        return "Positive", 1.0
    if "negative" in label or label == "label_0":
        return "Negative", -1.0
    return "Neutral", 0.0


def score_news(headlines: list[dict[str, object]], model) -> tuple[list[dict[str, object]], bool]:
    texts = [str(row["Headline"]) for row in headlines]
    if model is not None and texts:
        try:
            results = model(texts, truncation=True, batch_size=16)
            return [
                {**row, "Sentiment": _model_label(result)[0],
                 "Score": _model_label(result)[1]}
                for row, result in zip(headlines, results)
            ], True
        except Exception:
            pass
    return [
        {**row, "Sentiment": keyword_sentiment(str(row["Headline"]))[0],
         "Score": keyword_sentiment(str(row["Headline"]))[1]}
        for row in headlines
    ], False


def sentiment_score(rows: list[dict[str, object]]) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row["Score"]) for row in rows]))


def weighted_score(
    scores: dict[str, float | None],
    weights: dict[str, int],
) -> float | None:
    available = [
        (score, weights.get(name, 0))
        for name, score in scores.items()
        if score is not None and weights.get(name, 0) > 0
    ]
    total_weight = sum(weight for _, weight in available)
    if total_weight == 0:
        return None
    return float(sum(float(score) * weight for score, weight in available) / total_weight)


def suggested_stance(score: float | None, risk_profile: str) -> str:
    if score is None:
        return "N/A"
    thresholds = STANCE_THRESHOLDS[risk_profile]
    if score >= thresholds["accumulate"]:
        return "Accumulate"
    if score >= thresholds["hold"]:
        return "Hold"
    if score >= thresholds["reduce"]:
        return "Reduce"
    return "Avoid"


def fundamental_score(
    ticker: str,
    fundamentals: dict[str, dict[str, object]],
) -> tuple[float | None, dict[str, float | None]]:
    values = fundamentals.get(ticker, {})
    per = values.get("PER")
    pbv = values.get("PBV")
    roe = values.get("ROE")
    der = values.get("DER")

    def valid_number(value: object, positive: bool = False) -> float | None:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not np.isfinite(number) or (positive and number <= 0):
            return None
        return number

    per_value = valid_number(per, positive=True)
    pbv_value = valid_number(pbv, positive=True)
    roe_value = valid_number(roe)
    der_value = valid_number(der)
    valid_pers = [
        number for item in fundamentals.values()
        if (number := valid_number(item.get("PER"), positive=True)) is not None
    ]
    valid_pbvs = [
        number for item in fundamentals.values()
        if (number := valid_number(item.get("PBV"), positive=True)) is not None
    ]
    components: dict[str, float | None] = {
        "PER": (
            1.0 if per_value <= float(np.mean(valid_pers)) else -1.0
        ) if per_value is not None and valid_pers else None,
        "PBV": (
            1.0 if pbv_value <= float(np.mean(valid_pbvs)) else -1.0
        ) if pbv_value is not None and valid_pbvs else None,
        "ROE": (
            1.0 if roe_value >= FUNDAMENTAL_RULES["roe_strong"]
            else 0.0 if roe_value >= FUNDAMENTAL_RULES["roe_moderate"]
            else -1.0
        ) if roe_value is not None else None,
        "DER": (
            1.0 if der_value <= FUNDAMENTAL_RULES["der_low"]
            else 0.0 if der_value <= FUNDAMENTAL_RULES["der_high"]
            else -1.0
        ) if der_value is not None else None,
    }
    valid_components = [value for value in components.values() if value is not None]
    return (float(np.mean(valid_components)) if valid_components else None), components


def calculate_risk(prices: pd.DataFrame) -> dict[str, object]:
    empty: dict[str, object] = {
        "volatility": None,
        "max_drawdown_1y": None,
        "max_drawdown_3y": None,
        "atr_pct": None,
        "drawdown_series": pd.Series(dtype=float),
        "drawdown_dates": pd.DatetimeIndex([]),
    }
    if prices.empty:
        return empty
    latest_date = prices.index[-1]
    returns = prices["close"].pct_change().dropna()
    volatility_start = latest_date - pd.DateOffset(years=RISK_LOOKBACK_YEARS)
    volatility_returns = returns.loc[returns.index >= volatility_start]
    volatility = (
        float(volatility_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(volatility_returns) > 1
        else None
    )
    drawdown_values: dict[int, float | None] = {}
    drawdown_series: pd.Series = pd.Series(dtype=float)
    drawdown_dates = pd.DatetimeIndex([])
    for years in RISK_DRAWDOWN_YEARS:
        start = latest_date - pd.DateOffset(years=years)
        window = prices.loc[prices.index >= start, "close"]
        if window.empty:
            drawdown_values[years] = None
            continue
        drawdown = window / window.cummax() - 1
        drawdown_values[years] = float(drawdown.min())
        if years == 3:
            drawdown_series = drawdown
            drawdown_dates = window.index
    indicators = add_technicals(prices)
    latest_close = float(prices["close"].iloc[-1])
    latest_atr = indicators["ATR"].iloc[-1]
    atr_pct = (
        float(latest_atr / latest_close)
        if pd.notna(latest_atr) and latest_close != 0
        else None
    )
    return {
        "volatility": volatility,
        "max_drawdown_1y": drawdown_values.get(1),
        "max_drawdown_3y": drawdown_values.get(3),
        "atr_pct": atr_pct,
        "drawdown_series": drawdown_series,
        "drawdown_dates": drawdown_dates,
    }


def classify_risk(risks: dict[str, dict[str, object]]) -> dict[str, str]:
    available = {
        ticker: float(values["volatility"])
        for ticker, values in risks.items()
        if values["volatility"] is not None
    }
    if not available:
        return {ticker: "N/A" for ticker in risks}
    values = np.asarray(list(available.values()), dtype=float)
    lower_tercile, upper_tercile = np.quantile(values, RISK_TERCILE_CUTOFFS)
    if lower_tercile == upper_tercile:
        labels = {ticker: "Medium" for ticker in available}
    else:
        labels = {
            ticker: (
                "Low" if value <= lower_tercile
                else "Medium" if value <= upper_tercile
                else "High"
            )
            for ticker, value in available.items()
        }
    return {ticker: labels.get(ticker, "N/A") for ticker in risks}


def backtest_technical(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if prices.empty:
        return pd.DataFrame(), {}
    data = add_technicals(prices)
    scores = technical_score_series(data)
    daily_returns = data["close"].pct_change()
    position = (scores.shift(1) > BACKTEST_TECHNICAL_THRESHOLD).astype(float)
    start = data.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
    results = pd.DataFrame({
        "Buy-and-hold benchmark": daily_returns,
        "Technical strategy": daily_returns * position,
    }).loc[lambda frame: frame.index >= start].dropna()
    if results.empty:
        return results, {}
    equity = (1 + results).cumprod()
    metrics: dict[str, dict[str, float]] = {}
    for name in results.columns:
        curve = equity[name]
        drawdown = curve / curve.cummax().clip(lower=1.0) - 1
        standard_deviation = results[name].std(ddof=1)
        sharpe = (
            float(results[name].mean() / standard_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
            if standard_deviation > 0
            else 0.0
        )
        metrics[name] = {
            "return": float(curve.iloc[-1] - 1),
            "max_drawdown": float(drawdown.min()),
            "sharpe": sharpe,
        }
    return equity, metrics


def _score_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}"


def render_technical(ticker: str, prices: pd.DataFrame) -> float | None:
    st.subheader(f"{ticker} technical analysis")
    if prices.empty:
        st.warning("No price data is available. Check the Yahoo Finance connection or add a CSV under data/.")
        return None
    data = add_technicals(prices)
    latest = data.iloc[-1]
    first, second, third = st.columns(3)
    first.metric("Latest close", _format_price(latest["close"]))
    second.metric("RSI (14)", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
    score = technical_score(data)
    third.metric("Technical score", _score_label(score))
    st.plotly_chart(technical_chart(data, ticker), use_container_width=True)
    st.markdown(
        f"**Score rules:** price above SMA{TECHNICAL_TREND_SMA_WINDOW} = +1 (otherwise -1); "
        f"RSI below {RSI_OVERSOLD_THRESHOLD} = +1, above {RSI_OVERBOUGHT_THRESHOLD} = -1, otherwise 0; "
        "MACD above its signal line = +1 (otherwise -1). "
        "The technical score is the mean of the available rule values, bounded from -1 to +1."
    )
    return score


def render_fundamental(
    ticker: str,
    fundamentals: dict[str, dict[str, object]],
) -> None:
    st.subheader("Fundamentals")
    info = fundamentals.get(ticker, {})
    st.caption(f"Data as of {date.today().isoformat()}")
    keys = ["PER", "PBV", "ROE", "DER", "Dividend Yield", "Market Cap"]
    columns = st.columns(3)
    for index, key in enumerate(keys):
        columns[index % 3].metric(key, _format_metric(key, info.get(key)))
    score, _ = fundamental_score(ticker, fundamentals)
    st.metric("Fundamental score", _score_label(score))
    st.caption(
        f"Score rules: PER and PBV at or below the available 4-stock average = +1, above = −1; "
        f"ROE ≥ {FUNDAMENTAL_RULES['roe_strong']:.0%} = +1, "
        f"{FUNDAMENTAL_RULES['roe_moderate']:.0%}–{FUNDAMENTAL_RULES['roe_strong']:.0%} = 0, "
        f"below {FUNDAMENTAL_RULES['roe_moderate']:.0%} = −1; "
        f"DER ≤ {FUNDAMENTAL_RULES['der_low']:.0f} = +1, "
        f"{FUNDAMENTAL_RULES['der_low']:.0f}–{FUNDAMENTAL_RULES['der_high']:.0f} = 0, "
        f"above {FUNDAMENTAL_RULES['der_high']:.0f} = −1. "
        "Missing inputs are excluded from the average."
    )

    st.markdown("#### Four-stock comparison")
    comparison: list[dict[str, object]] = []
    for symbol in STOCKS:
        values = fundamentals.get(symbol, {})
        fundamental_value, _ = fundamental_score(symbol, fundamentals)
        comparison.append({
            "Ticker": symbol,
            **{key: _format_metric(key, values.get(key)) for key in keys},
            "Score": _score_label(fundamental_value),
        })
    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)


def render_sentiment(ticker: str, model) -> tuple[float | None, list[dict[str, object]], bool]:
    st.subheader("News sentiment (last 30 days)")
    headlines, warnings = load_news(ticker)
    for warning in warnings:
        st.warning(warning)
    scored, used_model = score_news(headlines, model)
    if model is None or (headlines and not used_model):
        st.info("The sentiment model could not be loaded or scored this batch; using the built-in keyword lexicon.")
    if not scored:
        st.warning("No recent Google News headlines were found for this stock.")
        return None, [], used_model

    frame = pd.DataFrame(scored)
    daily = frame.groupby("Date", as_index=False)["Score"].mean().sort_values("Date")
    line = go.Figure(go.Scatter(
        x=daily["Date"], y=daily["Score"], mode="lines+markers",
        line={"color": "#2563eb"}, name="Average sentiment",
    ))
    line.update_layout(
        title="Daily average sentiment", yaxis={"range": [-1, 1], "title": "Score"},
        xaxis_title="Date", height=330, margin={"t": 55, "b": 25},
    )
    sentiment_order = ["Positive", "Neutral", "Negative"]
    counts = frame["Sentiment"].value_counts().reindex(sentiment_order, fill_value=0)
    bars = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=["#16a34a", "#94a3b8", "#dc2626"],
    ))
    bars.update_layout(title="Headline sentiment counts", yaxis_title="Headlines", height=330, margin={"t": 55, "b": 25})
    left, right = st.columns(2)
    left.plotly_chart(line, use_container_width=True)
    right.plotly_chart(bars, use_container_width=True)

    st.markdown("#### Latest headlines")
    latest = sorted(scored, key=lambda row: row["Published"], reverse=True)[:10]
    table = pd.DataFrame([
        {
            "Date": row["Published"].strftime("%Y-%m-%d"),
            "Headline": row["Headline"],
            "Sentiment": row["Sentiment"],
            "Score": _score_label(float(row["Score"])),
            "Source": row["Source"],
            "Link": row["Link"],
        }
        for row in latest
    ])
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={"Link": st.column_config.LinkColumn("Article", display_text="Open")},
    )
    return sentiment_score(scored), scored, used_model


def _technical_reason(data: pd.DataFrame) -> str:
    if data.empty:
        return "Price history is unavailable."
    latest = data.iloc[-1]
    parts = []
    trend_sma = latest[f"SMA{TECHNICAL_TREND_SMA_WINDOW}"]
    if pd.notna(trend_sma):
        parts.append(
            f"price above SMA{TECHNICAL_TREND_SMA_WINDOW} (+)"
            if latest["close"] > trend_sma
            else f"price below SMA{TECHNICAL_TREND_SMA_WINDOW} (−)"
        )
    if pd.notna(latest["RSI"]):
        parts.append(
            "RSI is oversold (+)" if latest["RSI"] < RSI_OVERSOLD_THRESHOLD
            else "RSI is overbought (−)" if latest["RSI"] > RSI_OVERBOUGHT_THRESHOLD
            else "RSI is in its neutral range"
        )
    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_signal"]):
        parts.append("MACD is above its signal (+)" if latest["MACD"] > latest["MACD_signal"] else "MACD is below its signal (−)")
    return "; ".join(parts) if parts else "Not enough price history to calculate the technical rules."


def _fundamental_reason(
    ticker: str,
    fundamentals: dict[str, dict[str, object]],
) -> str:
    score, components = fundamental_score(ticker, fundamentals)
    if score is None:
        return "Fundamental inputs are unavailable and were excluded."
    values = fundamentals.get(ticker, {})
    pieces = []
    for metric, label in (("PER", "PER"), ("PBV", "PBV")):
        if components[metric] is None:
            continue
        group_values = [
            float(item[metric]) for item in fundamentals.values()
            if item.get(metric) is not None
            and pd.notna(item.get(metric))
            and float(item[metric]) > 0
        ]
        if group_values:
            relation = "at or below" if components[metric] > 0 else "above"
            pieces.append(
                f"{label} {_format_metric(metric, values.get(metric))} {relation} "
                f"group average {_format_metric(metric, float(np.mean(group_values)))}"
            )
    if components["ROE"] is not None:
        pieces.append(f"ROE {_format_metric('ROE', values.get('ROE'))}")
    if components["DER"] is not None:
        pieces.append(f"DER {_format_metric('DER', values.get('DER'))}")
    return f"score {_score_label(score)}; " + (", ".join(pieces) or "available metrics scored")


def _format_percent(value: object, multiplier: float = 100.0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * multiplier:.2f}%"


def _decision_data(
    active_ticker: str,
    current_prices: pd.DataFrame,
    model,
    current_technical: float | None,
    current_sentiment: float | None,
    fundamentals: dict[str, dict[str, object]],
    weights: dict[str, int],
    risk_profile: str,
) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    price_sets: dict[str, pd.DataFrame] = {active_ticker: current_prices}
    for ticker in STOCKS:
        if ticker != active_ticker:
            prices, _ = load_prices(ticker)
            price_sets[ticker] = prices

    risks = {ticker: calculate_risk(price_sets[ticker]) for ticker in STOCKS}
    risk_labels = classify_risk(risks)
    rows = []
    for ticker in STOCKS:
        prices = price_sets[ticker]
        if ticker == active_ticker:
            technical = current_technical
            sentiment = current_sentiment
        else:
            technical = technical_score(add_technicals(prices)) if not prices.empty else None
            headlines, _ = load_news(ticker)
            scored_headlines, _ = score_news(headlines, model)
            sentiment = sentiment_score(scored_headlines)
        fundamental, _ = fundamental_score(ticker, fundamentals)
        scores = {
            "technical": technical,
            "sentiment": sentiment,
            "fundamental": fundamental,
        }
        combined = weighted_score(scores, weights)
        rows.append({
            "Ticker": ticker,
            "scores": scores,
            "combined": combined,
            "risk": risk_labels[ticker],
            "stance": suggested_stance(combined, risk_profile),
        })
    rows.sort(key=lambda row: row["combined"] if row["combined"] is not None else -2, reverse=True)
    return rows, price_sets, risks


def _render_risk_panel(
    ticker: str,
    prices: pd.DataFrame,
    risk: dict[str, object],
    risk_label: str,
) -> None:
    st.markdown("#### Risk profile")
    columns = st.columns(4)
    columns[0].metric("Annualized volatility", _format_percent(risk["volatility"]))
    columns[1].metric("Max drawdown · 1Y", _format_percent(risk["max_drawdown_1y"]))
    columns[2].metric("Max drawdown · 3Y", _format_percent(risk["max_drawdown_3y"]))
    columns[3].metric("ATR / price", _format_percent(risk["atr_pct"]))
    color = RISK_COLORS.get(risk_label, "#94a3b8")
    st.markdown(
        f'<div class="risk-tag" style="--tag-color:{color}">Relative risk: <b>{risk_label}</b>'
        f' <span>(volatility tercile across available IDX stocks)</span></div>',
        unsafe_allow_html=True,
    )
    drawdown = risk["drawdown_series"]
    dates = risk["drawdown_dates"]
    if isinstance(drawdown, pd.Series) and not drawdown.empty:
        figure = go.Figure(go.Scatter(
            x=dates,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            line={"color": "#f05260", "width": 1.8},
            fill="tozeroy",
            fillcolor="rgba(240,82,96,0.12)",
            hovertemplate="%{x|%d %b %Y}<br>Drawdown %{y:.2f}%<extra></extra>",
        ))
        _overview_chart_layout(figure, 230)
        figure.update_layout(title=f"{ticker} drawdown · 3 years", showlegend=False)
        figure.update_yaxes(title="Drawdown (%)", ticksuffix="%")
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_backtest(prices: pd.DataFrame) -> None:
    st.markdown("#### Simple technical-only backtest · 3 years")
    st.caption(
        f"Position is held only when the previous session's technical score was above "
        f"{BACKTEST_TECHNICAL_THRESHOLD:.1f}; otherwise it stays in cash. "
        "Sentiment history is too short for this test. Signals are shifted one day to avoid "
        "look-ahead bias. Transaction costs and slippage are ignored."
    )
    equity, metrics = backtest_technical(prices)
    if equity.empty or not metrics:
        st.warning("Not enough historical prices to calculate the 3-year backtest.")
        return
    names = ["Technical strategy", "Buy-and-hold benchmark"]
    metric_rows = []
    for name in names:
        values = metrics[name]
        metric_rows.append({
            "Strategy": name,
            "Cumulative return": _format_percent(values["return"]),
            "Maximum drawdown": _format_percent(values["max_drawdown"]),
            "Sharpe ratio": f"{values['sharpe']:.2f}",
        })
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    figure = go.Figure()
    colors = {"Technical strategy": "#21d07a", "Buy-and-hold benchmark": "#60a5fa"}
    for name in names:
        figure.add_trace(go.Scatter(
            x=equity.index,
            y=equity[name],
            mode="lines",
            name=name,
            line={"color": colors[name], "width": 2},
            hovertemplate="%{x|%d %b %Y}<br>Indexed equity %{y:.3f}<extra>%{fullData.name}</extra>",
        ))
    _overview_chart_layout(figure, 300)
    figure.update_layout(title="Growth of 1.0 unit", yaxis_title="Portfolio value")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_decision_support(
    active_ticker: str,
    current_prices: pd.DataFrame,
    model,
    current_technical: float | None,
    current_sentiment: float | None,
    fundamentals: dict[str, dict[str, object]],
    weights: dict[str, int],
    risk_profile: str,
) -> None:
    st.subheader("Decision Support")
    st.warning(
        "Educational and research use only, not investment advice. "
        "Past performance does not guarantee future results."
    )
    st.caption(
        f"Risk profile: {risk_profile}. Weights are renormalized over available scores "
        f"(technical {weights['technical']}% · sentiment {weights['sentiment']}% · "
        f"fundamental {weights['fundamental']}%)."
    )
    rows, price_sets, risks = _decision_data(
        active_ticker,
        current_prices,
        model,
        current_technical,
        current_sentiment,
        fundamentals,
        weights,
        risk_profile,
    )
    selected = next(row for row in rows if row["Ticker"] == active_ticker)
    combined = selected["combined"]
    stance = str(selected["stance"])
    stance_color = STANCE_COLORS.get(stance, "#94a3b8")
    technical_value = selected["scores"]["technical"]
    st.markdown(
        f"""
        <div class="decision-card" style="--stance-color:{stance_color}">
          <div class="decision-label">SUGGESTED STANCE · {active_ticker}</div>
          <div class="decision-value">{stance}</div>
          <div class="decision-score">Combined score &nbsp; {_score_label(combined)}</div>
          <div class="decision-caveat">Available conditions lean toward this stance; it is a research reference, not a forecast or instruction.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Why")
    active_data = add_technicals(current_prices) if not current_prices.empty else pd.DataFrame()
    technical_note = (
        f"Technical score {_score_label(technical_value)}: {_technical_reason(active_data)}."
        if technical_value is not None else "Technical score is unavailable and was excluded."
    )
    sentiment_note = (
        f"Recent news sentiment is {current_sentiment:+.2f} "
        f"({'positive' if current_sentiment > SENTIMENT_NEUTRAL_BAND else 'negative' if current_sentiment < -SENTIMENT_NEUTRAL_BAND else 'neutral'})."
        if current_sentiment is not None else "No recent headline sentiment is available; it was excluded."
    )
    fundamental_value, _ = fundamental_score(active_ticker, fundamentals)
    fundamental_note = (
        f"Fundamental score {_score_label(fundamental_value)}: "
        f"{_fundamental_reason(active_ticker, fundamentals)}."
        if fundamental_value is not None else "Fundamental data is unavailable and was excluded."
    )
    with st.container(border=True):
        st.markdown(
            f"- {technical_note}\n"
            f"- {sentiment_note}\n"
            f"- {fundamental_note}"
        )

    st.markdown("#### Risk panel")
    selected_risk = risks[active_ticker]
    _render_risk_panel(active_ticker, current_prices, selected_risk, str(selected["risk"]))

    st.markdown("#### Key levels")
    if current_prices.empty:
        st.info("Price history is unavailable; reference levels cannot be calculated.")
    else:
        recent = current_prices.tail(SWING_LOOKBACK_DAYS)
        indicators = add_technicals(current_prices)
        latest_close = float(current_prices["close"].iloc[-1])
        latest_atr = indicators["ATR"].iloc[-1]
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())
        reference_stop = (
            latest_close - ATR_STOP_MULTIPLIER * float(latest_atr)
            if pd.notna(latest_atr)
            else None
        )
        level_columns = st.columns(3)
        level_columns[0].metric(f"Support · last {SWING_LOOKBACK_DAYS} sessions", f"Rp {_format_price(support)}")
        level_columns[1].metric(f"Resistance · last {SWING_LOOKBACK_DAYS} sessions", f"Rp {_format_price(resistance)}")
        level_columns[2].metric(
            f"ATR reference · close − {ATR_STOP_MULTIPLIER:.0f}× ATR",
            f"Rp {_format_price(reference_stop)}" if reference_stop is not None else "N/A",
        )
    st.caption("Swing levels and ATR-based reference levels are estimates from historical data, not trade instructions.")

    st.markdown("#### Four-stock comparison and ranking")
    table_rows = [{
        "Rank": rank,
        "Ticker": row["Ticker"],
        "Technical": _score_label(row["scores"]["technical"]),
        "Sentiment": _score_label(row["scores"]["sentiment"]),
        "Fundamental": _score_label(row["scores"]["fundamental"]),
        "Combined": _score_label(row["combined"]),
        "Risk label": row["risk"],
        "Suggested stance": row["stance"],
    } for rank, row in enumerate(rows, start=1)]
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    scatter_rows = [row for row in rows if row["combined"] is not None and risks[row["Ticker"]]["volatility"] is not None]
    if scatter_rows:
        scatter = go.Figure()
        for risk_label in ("Low", "Medium", "High"):
            group = [row for row in scatter_rows if row["risk"] == risk_label]
            if not group:
                continue
            scatter.add_trace(go.Scatter(
                x=[row["combined"] for row in group],
                y=[risks[row["Ticker"]]["volatility"] * 100 for row in group],
                mode="markers+text",
                text=[row["Ticker"] for row in group],
                textposition="top center",
                name=f"{risk_label} risk",
                marker={"color": RISK_COLORS[risk_label], "size": 13, "line": {"color": "#0b1422", "width": 1}},
                hovertemplate="%{text}<br>Combined %{x:+.2f}<br>Annualized volatility %{y:.1f}%<extra></extra>",
            ))
        _overview_chart_layout(scatter, 350)
        scatter.update_layout(
            title="Combined score vs. annualized volatility",
            xaxis={"title": "Combined score", "range": [-1.1, 1.1], "zeroline": True, "zerolinecolor": "#64748b"},
            yaxis_title="Annualized volatility (%)",
            legend={"orientation": "h", "y": 1.14, "x": 0},
        )
        st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

    _render_backtest(current_prices)


def _overview_card(label: str, value: str, note: str, positive: bool | None = None) -> None:
    tone = "neutral" if positive is None else "positive" if positive else "negative"
    st.markdown(
        f"""
        <div class="overview-card">
          <div class="card-label">{label}</div>
          <div class="card-value">{value}</div>
          <div class="card-note {tone}">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _overview_chart_layout(figure: go.Figure, height: int) -> None:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1", "family": "Inter, sans-serif", "size": 11},
        margin={"l": 8, "r": 8, "t": 42, "b": 8},
        height=height,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(showgrid=False, linecolor="#263447", zeroline=False)
    figure.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
        tickfont={"color": "#8290a3"},
    )


def render_overview(ticker: str, prices: pd.DataFrame) -> None:
    last_price_date = prices.index[-1] if not prices.empty else None
    data_label = last_price_date.strftime("%d %b %Y").upper() if last_price_date is not None else "NO PRICE DATA"
    st.markdown(
        f"""
        <div class="overview-heading">
          <div>
            <div class="eyebrow">Market overview</div>
            <h1>{STOCKS[ticker]}</h1>
            <div class="overview-subtitle">{ticker} · Adjusted daily prices and market snapshot</div>
          </div>
          <div class="asof-chip">● &nbsp; DATA THROUGH {data_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prices.empty:
        st.warning("No price data is available. Check the Yahoo Finance connection or add a CSV under data/.")
        return

    periods = {
        "1 month": pd.DateOffset(months=1),
        "3 months": pd.DateOffset(months=3),
        "6 months": pd.DateOffset(months=6),
        "1 year": pd.DateOffset(years=1),
        "5 years": pd.DateOffset(years=5),
    }
    selector, latest_note = st.columns([1, 3])
    with selector:
        selected_period = st.selectbox(
            "Chart period", list(periods), index=3, label_visibility="collapsed"
        )

    latest = prices.iloc[-1]
    latest_date = prices.index[-1].strftime("%d %b %Y")
    cutoff = prices.index[-1] - periods[selected_period]
    period_prices = prices.loc[prices.index >= cutoff]
    if period_prices.empty:
        period_prices = prices.tail(1)
    start_close = float(period_prices["close"].iloc[0])
    period_return = (float(latest["close"]) / start_close - 1) * 100 if start_close else 0
    previous_close = prices["close"].iloc[-2] if len(prices) > 1 else np.nan
    daily_return = (
        (float(latest["close"]) / float(previous_close) - 1) * 100
        if pd.notna(previous_close) and previous_close != 0
        else None
    )
    indicators = add_technicals(prices)
    score = technical_score(indicators)
    year_start = prices.index[-1] - pd.DateOffset(years=1)
    year_prices = prices.loc[prices.index >= year_start]
    year_high = float(year_prices["high"].max()) if not year_prices.empty else np.nan
    year_low = float(year_prices["low"].min()) if not year_prices.empty else np.nan
    latest_note.markdown(
        f'<div class="latest-note">Latest session: <b>{latest_date}</b>'
        f' &nbsp;·&nbsp; 52W range: <b>Rp {_format_price(year_low)} – {_format_price(year_high)}</b></div>',
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        _overview_card("LATEST CLOSE", f"Rp {_format_price(latest['close'])}", latest_date)
    with metric_columns[1]:
        _overview_card(
            "DAILY CHANGE",
            f"{daily_return:+.2f}%" if daily_return is not None else "N/A",
            "vs. previous session",
            daily_return >= 0 if daily_return is not None else None,
        )
    with metric_columns[2]:
        _overview_card(
            f"{selected_period.upper()} RETURN",
            f"{period_return:+.2f}%",
            f"from Rp {_format_price(start_close)}",
            period_return >= 0,
        )
    with metric_columns[3]:
        _overview_card(
            "TECHNICAL SCORE",
            _score_label(score),
            "Rule-based · scale −1 to +1",
            score >= 0 if score is not None else None,
        )

    chart_column, distribution_column = st.columns([2.35, 1])
    with chart_column:
        with st.container(border=True):
            st.markdown(f"#### Price performance · {selected_period}")
            performance = go.Figure()
            performance.add_trace(go.Scatter(
                x=period_prices.index,
                y=period_prices["close"],
                mode="lines",
                name=ticker,
                line={"color": "#21d07a", "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(33,208,122,0.12)",
                hovertemplate="%{x|%d %b %Y}<br>Rp %{y:,.0f}<extra></extra>",
            ))
            _overview_chart_layout(performance, 370)
            performance.update_layout(showlegend=False)
            performance.update_yaxes(title="Price (IDR)", tickformat=",.0f")
            st.plotly_chart(performance, use_container_width=True, config={"displayModeBar": False})

    with distribution_column:
        with st.container(border=True):
            st.markdown("#### Daily price moves")
            returns = period_prices["close"].pct_change().dropna()
            counts = [
                int((returns > 0).sum()),
                int((returns == 0).sum()),
                int((returns < 0).sum()),
            ]
            movement = go.Figure(go.Pie(
                labels=["Up days", "Unchanged", "Down days"],
                values=counts,
                hole=0.72,
                sort=False,
                marker={"colors": ["#21d07a", "#64748b", "#f05260"], "line": {"color": "#0b1220", "width": 3}},
                textinfo="none",
                hovertemplate="%{label}: %{value} days<extra></extra>",
            ))
            _overview_chart_layout(movement, 275)
            movement.update_layout(
                showlegend=True,
                legend={"orientation": "v", "y": 0.5, "x": 0.82, "font": {"size": 10}},
                margin={"l": 0, "r": 70, "t": 12, "b": 10},
                annotations=[{
                    "text": f"<b>{len(returns)}</b><br><span style='font-size:10px'>sessions</span>",
                    "x": 0.39, "y": 0.5, "showarrow": False,
                    "font": {"size": 20, "color": "#f8fafc"},
                }],
            )
            st.plotly_chart(movement, use_container_width=True, config={"displayModeBar": False})

    monthly_column, comparison_column = st.columns([1, 1.55])
    with monthly_column:
        with st.container(border=True):
            st.markdown("#### Monthly performance")
            monthly_close = prices["close"].groupby(prices.index.to_period("M")).last()
            monthly_returns = (monthly_close.pct_change() * 100).dropna().tail(6)
            month_colors = ["#21d07a" if value >= 0 else "#f05260" for value in monthly_returns]
            monthly = go.Figure(go.Bar(
                x=[period.strftime("%b %y") for period in monthly_returns.index],
                y=monthly_returns.values,
                marker_color=month_colors,
                hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
            ))
            _overview_chart_layout(monthly, 250)
            monthly.update_layout(showlegend=False)
            monthly.update_yaxes(title="Return (%)", ticksuffix="%")
            st.plotly_chart(monthly, use_container_width=True, config={"displayModeBar": False})

    with comparison_column:
        with st.container(border=True):
            st.markdown("#### One-year stock comparison")
            peer_rows: list[dict[str, object]] = []
            for symbol in STOCKS:
                peer_prices = prices if symbol == ticker else load_prices(symbol)[0]
                if peer_prices.empty:
                    value = None
                else:
                    peer_prices = peer_prices.loc[peer_prices.index <= prices.index[-1]]
                    if len(peer_prices) > 1:
                        peer_cutoff = peer_prices.index[-1] - pd.DateOffset(years=1)
                        window = peer_prices.loc[peer_prices.index >= peer_cutoff]
                        if len(window) > 1 and window["close"].iloc[0] != 0:
                            value = (peer_prices["close"].iloc[-1] / window["close"].iloc[0] - 1) * 100
                        else:
                            value = None
                    else:
                        value = None
                peer_rows.append({"Ticker": symbol, "1Y return": value})
            valid_rows = [row for row in peer_rows if row["1Y return"] is not None]
            peer_chart = go.Figure(go.Bar(
                x=[row["1Y return"] for row in valid_rows],
                y=[row["Ticker"] for row in valid_rows],
                orientation="h",
                marker_color=[
                    "#21d07a" if row["1Y return"] >= 0 else "#f05260"
                    for row in valid_rows
                ],
                text=[f"{row['1Y return']:+.1f}%" for row in valid_rows],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
            ))
            _overview_chart_layout(peer_chart, 250)
            peer_chart.update_layout(showlegend=False, margin={"l": 8, "r": 45, "t": 12, "b": 8})
            peer_chart.update_xaxes(title="Return (%)", ticksuffix="%")
            st.plotly_chart(peer_chart, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="IDX Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --page: #080e18;
            --panel: #0e1724;
            --panel-raised: #121e2d;
            --border: #243244;
            --text: #e5eaf1;
            --muted: #91a0b2;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--page) !important;
            color: var(--text);
        }
        [data-testid="stHeader"] { background: var(--page); }
        [data-testid="stToolbar"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        [data-testid="stSidebar"] {
            background: #0b121d;
            border-right: 1px solid #202b39;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
        .block-container { max-width: 1540px; padding-top: 1.7rem; padding-bottom: 2rem; }
        h1, h2, h3, h4 { color: #edf1f6 !important; letter-spacing: -0.015em; }
        .overview-heading {
            display: flex; justify-content: space-between; align-items: center;
            gap: 1rem; margin: 0.2rem 0 1.25rem;
        }
        .overview-heading h1 { font-size: 1.72rem; margin: 0.28rem 0 0.15rem; }
        .eyebrow { color: #9babbd; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.015em; }
        .overview-subtitle, .latest-note { color: #91a0b2; font-size: 0.82rem; }
        .latest-note { padding-top: 0.62rem; text-align: right; }
        .latest-note b { color: #e1e7ef; }
        .asof-chip {
            border: 1px solid var(--border); border-radius: 7px; padding: 0.58rem 0.8rem;
            color: #aab8ca; font-size: 0.68rem;
            white-space: nowrap; background: var(--panel);
        }
        .overview-card {
            height: 105px; padding: 0.92rem 1rem 0.8rem; margin-bottom: 1rem;
            background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
        }
        .card-label { color: #9aa8b8; font-size: 0.67rem; font-weight: 600; letter-spacing: 0.025em; }
        .card-value { color: #edf1f6; font-size: 1.32rem; font-weight: 620; margin-top: 0.38rem; white-space: nowrap; }
        .card-note { color: #91a0b2; font-size: 0.7rem; line-height: 1.35; margin-top: 0.18rem; }
        .card-note.positive { color: #49c98b; }
        .card-note.negative { color: #e5767f; }
        .decision-card {
            padding: 1.15rem 1.35rem; border: 1px solid var(--border);
            border-left: 3px solid var(--stance-color); border-radius: 8px;
            background: var(--panel);
        }
        .decision-label { color: #9aa8bb; font-size: 0.68rem; letter-spacing: 0.035em; font-weight: 600; }
        .decision-value { color: var(--stance-color); font-size: 1.65rem; font-weight: 680; margin-top: 0.25rem; }
        .decision-score { color: #e1e7ef; font-size: 0.88rem; margin-top: 0.12rem; }
        .decision-caveat { color: #98a6ba; font-size: 0.78rem; margin-top: 0.35rem; }
        .risk-tag {
            display: inline-block; border: 1px solid var(--border); border-radius: 6px;
            color: var(--tag-color); padding: 0.38rem 0.65rem; margin: 0.2rem 0 0.65rem;
            font-size: 0.78rem;
        }
        .risk-tag span { color: var(--muted); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel) !important; border-color: var(--border) !important; border-radius: 8px;
        }
        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid #1b2a3d; gap: 0.35rem;
        }
        [data-testid="stTabs"] button[role="tab"] { color: #a1adbb; }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #21d07a; border-bottom-color: #21d07a;
        }
        [data-testid="stDataFrame"] {
            color-scheme: dark; background: var(--panel) !important;
            border: 1px solid var(--border); border-radius: 8px;
        }
        [data-testid="stDataFrame"] iframe { background: var(--panel) !important; color-scheme: dark; }
        [data-testid="stMetric"] {
            background: var(--panel) !important; border: 1px solid var(--border);
            padding: 0.85rem; border-radius: 8px;
        }
        [data-testid="stMetricLabel"] { color: #98a6ba; }
        [data-testid="stMetricValue"] { color: #edf1f6; }
        [data-testid="stAlert"] {
            color-scheme: dark; background: #111b29 !important;
            border: 1px solid var(--border); color: var(--text);
        }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: #d4dce6 !important; }
        [data-testid="stAlert"] svg { fill: #aab8c8; }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            color-scheme: dark; background: var(--panel-raised) !important;
            border-color: var(--border) !important; color: var(--text);
        }
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"], [role="listbox"] {
            color-scheme: dark; background: var(--panel-raised) !important;
            border-color: var(--border) !important; color: var(--text);
        }
        [role="option"] { background: var(--panel-raised); color: var(--text); }
        [role="option"]:hover { background: #1b2a3d; }
        code, pre, [data-testid="stCode"] {
            color-scheme: dark; background: #111b29 !important; color: #d4dce6 !important;
            border-color: var(--border) !important;
        }
        .modebar, .modebar-group { background: transparent !important; }
        hr { border-color: var(--border); }
        @media (max-width: 900px) {
            .overview-heading { align-items: flex-start; flex-direction: column; }
            .latest-note { text-align: left; }
            .card-label { font-size: 0.58rem; }
            .card-value { font-size: 1.02rem; }
            .overview-card { height: auto; min-height: 100px; padding-left: 0.65rem; padding-right: 0.65rem; }
            .card-note { font-size: 0.62rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("## IDX Dashboard")
        st.caption("Stock market overview")
        st.divider()
        st.markdown("#### Market overview")
        ticker = st.selectbox("Select stock", list(STOCKS), format_func=lambda value: f"{value} — {STOCKS[value]}")
        st.divider()
        st.markdown("#### Portfolio settings")
        weights = {
            "technical": st.slider(
                "Technical weight (%)", WEIGHT_MIN, WEIGHT_MAX, DEFAULT_WEIGHTS["technical"], WEIGHT_STEP,
                key="decision_technical_weight",
            ),
            "sentiment": st.slider(
                "Sentiment weight (%)", WEIGHT_MIN, WEIGHT_MAX, DEFAULT_WEIGHTS["sentiment"], WEIGHT_STEP,
                key="decision_sentiment_weight",
            ),
            "fundamental": st.slider(
                "Fundamental weight (%)", WEIGHT_MIN, WEIGHT_MAX, DEFAULT_WEIGHTS["fundamental"], WEIGHT_STEP,
                key="decision_fundamental_weight",
            ),
        }
        risk_profile = st.selectbox(
            "Risk profile", RISK_PROFILES, index=1, key="decision_risk_profile"
        )
        st.caption("Available score weights are automatically renormalized.")
        st.divider()
        st.markdown("#### Data sources")
        st.markdown("- [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`")
        st.markdown("- [Google News](https://news.google.com/) RSS")
        st.caption(
            f"Historical prices: 1 Jan 2019 to {date.today():%d %b %Y} "
            "(adjusted daily OHLCV; latest available market session)."
        )

    prices, price_warning = load_prices(ticker)
    if price_warning:
        st.warning(price_warning)
    if not prices.empty:
        prices = prices.loc[
            (prices.index >= pd.Timestamp("2019-01-01"))
            & (prices.index < pd.Timestamp(date.today() + timedelta(days=1)))
        ]
    fundamentals: dict[str, dict[str, object]] = {}
    fundamental_errors: dict[str, str] = {}
    for symbol in STOCKS:
        values, issue = load_fundamentals(symbol)
        fundamentals[symbol] = values
        if issue:
            fundamental_errors[symbol] = issue
    render_overview(ticker, prices)
    st.markdown("### Detailed analysis")
    sentiment_model = load_sentiment_model()

    technical_tab, fundamental_tab, sentiment_tab, decision_tab = st.tabs(
        ["Technical", "Fundamental", "Sentiment", "Decision Support"]
    )
    with technical_tab:
        technical = render_technical(ticker, prices)
    with fundamental_tab:
        for issue in fundamental_errors.values():
            st.warning(issue)
        render_fundamental(ticker, fundamentals)
    with sentiment_tab:
        sentiment, _, _ = render_sentiment(ticker, sentiment_model)
    with decision_tab:
        render_decision_support(
            ticker,
            prices,
            sentiment_model,
            technical,
            sentiment,
            fundamentals,
            weights,
            risk_profile,
        )

    st.divider()
    st.caption(
        "For educational and research purposes only, not investment advice. "
        "Data refreshed daily, not real-time."
    )


if __name__ == "__main__":
    main()
