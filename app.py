# -----------------------------
# Hotlist AI — Mobile (Auto-Discover + Resilient Pushshift Fallback)
# Finds the 15 most-talked tickers on Reddit over a recent window,
# runs VADER sentiment, validates symbols via Yahoo Finance,
# enriches with RSI/SMA/Volume, and ranks them.
# -----------------------------

import re, time, datetime as dt
from typing import Optional, List

import requests
import numpy as np
import pandas as pd
import streamlit as st

# Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Market data (no key)
import yfinance as yf

# ------- One-time NLTK bootstrap (cached on Streamlit Cloud) -------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ------- Defaults -------
DEFAULT_SUBREDDITS = ["stocks", "pennystocks", "themadinvestor", "TheRaceTo10Million"]

# Looks-like-a-ticker heuristic (2–5 uppercase letters)
TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")

# Primary + fallback endpoints
PRIMARY_BASE = "https://api.pullpush.io/reddit/search"
FALLBACK_BASE = "https://api.pushshift.io/reddit/search"

# Small stopword list to avoid obvious non-tickers
STOP_TICKER_WORDS = {
    "USD","CEO","DD","ATH","IPO","ETF","AI","YOLO","OTC","TOS","API",
    "GDP","CPI","FOMC","SEC","DTCC","RSI","SMA","EMA","EPS","PE","PEG"
}

# -------------------- Time window helpers --------------------
def last_two_market_days_start_ts(tz: str = "US/Eastern") -> int:
    """
    Unix timestamp at 00:00 of the EARLIER of the last two business days.
    (Mon–Fri; simple holiday-agnostic.)
    """
    now_local = pd.Timestamp.now(tz).normalize()
    found = 0
    day = 0
    start_candidate = now_local
    while found < 2:
        d = now_local - pd.Timedelta(days=day)
        if d.weekday() < 5:  # Mon-Fri
            start_candidate = d
            found += 1
        day += 1
    # convert to UTC epoch seconds
    return int(start_candidate.tz_convert("UTC").timestamp())

def five_day_window_start_ts() -> int:
    """Unix ts for ~5 calendar days ago (UTC)."""
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=5)).timestamp())

# -------------------- Extraction helpers --------------------
def clean_whitelist(raw: str) -> List[str]:
    return sorted({t.strip().upper() for t in re.split(r"[,\s]+", raw) if t.strip()})

def extract_tickers(text: str, whitelist: Optional[List[str]]):
    if not text:
        return []
    found = set(re.findall(TICKER_PATTERN, text.upper()))
    if whitelist is None:
        return [t for t in found if t not in STOP_TICKER_WORDS and 2 <= len(t) <= 5]
    return [t for t in found if t in whitelist]

# -------------------- HTTP + Pushshift wrappers --------------------
def http_get(url, params, timeout=20, retries=2, backoff=0.8):
    """GET with retries/backoff, returning JSON or None."""
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(backoff * (i + 1))
    return None

def pp_fetch(kind: str, params: dict):
    """
    Try PRIMARY (PullPush). If empty or error, try FALLBACK (api.pushshift.io).
    Returns list of items (dicts).
    """
    # Primary
    j = http_get(f"{PRIMARY_BASE}/{kind}", params)
    data = (j or {}).get("data") if isinstance(j, dict) else None
    if isinstance(data, list) and len(data) > 0:
        return data

    # Fallback
    j2 = http_get(f"{FALLBACK_BASE}/{kind}", params)
    data2 = (j2 or {}).get("data") if isinstance(j2, dict) else None
    if isinstance(data2, list):
        return data2

    return []

@st.cache_data(ttl=900)
def pullpush_fetch_sub(subreddit: str, limit_total: int = 150, after_ts: Optional[int] = None):
    """
    Fetch recent submissions + comments text for a subreddit via Pushshift-like APIs.
    Returns list[str] of text blobs. Uses primary mirror with fallback automatically.
    """
    texts = []
    base_params = {
        "subreddit": subreddit,
        "size": min(100, max(1, limit_total)),
        "sort_type": "created_utc",
        "sort": "desc",
    }
    if after_ts:
        base_params["after"] = after_ts

    # Submissions (titles + bodies)
    params1 = {**base_params, "is_video": "false"}
    subs = pp_fetch("submission", params1)
    for s in subs:
        title = s.get("title") or ""
        body  = s.get("selftext") or ""
        if title or body:
            texts.append(f"{title}\n{body}")

    # Comments (up to the remaining amount)
    remain = max(0, limit_total - len(texts))
    if remain:
        params2 = {**base_params, "size": min(100, remain)}
        coms = pp_fetch("comment", params2)
        for c in coms:
            body = c.get("body") or ""
            if body:
                texts.append(body)

    return texts

# -------------------- Scan + selection --------------------
@st.cache_data(ttl=900)
def scan_pushshift(subreddits, limit_per_sub, whitelist, after_ts=None):
    """Mentions & sentiment by ticker using Pushshift over an optional time window."""
    sid = SentimentIntensityAnalyzer()
    rows = []

    progress = st.progress(0)
    total = max(1, len(subreddits))
    for idx, sub in enumerate(subreddits, start=1):
        texts = pullpush_fetch_sub(sub, limit_total=limit_per_sub, after_ts=after_ts)
        if not texts:
            st.info(f"Skipping r/{sub} (no data returned from Pushshift)")
        else:
            for t in texts:
                tickers = extract_tickers(t, whitelist)
                if not tickers:
                    continue
                s = sid.polarity_scores(t)
                for tk in tickers:
                    rows.append({"ticker": tk, **s})
        progress.progress(idx / total)
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["ticker","mentions","pos","neg","neu","compound"])

    df = pd.DataFrame(rows)
    agg = df.groupby("ticker").agg(
        mentions=("ticker","count"),
        pos=("pos","mean"),
        neg=("neg","mean"),
        neu=("neu","mean"),
        compound=("compound","mean"),
    ).reset_index()
    return agg.sort_values(["mentions","compound"], ascending=[False, False])

@st.cache_data(ttl=900)
def is_valid_symbol_yf(ticker: str) -> bool:
    """Validate a ticker by checking if Yahoo returns any recent data."""
    try:
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=30)
        df = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True, threads=False)
        return not df.empty
    except Exception:
        return False

def keep_top15_valid(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first 15 valid symbols by mentions (validated via yfinance)."""
    if sent_df.empty:
        return sent_df
    picked = []
    for t in sent_df.sort_values(["mentions","compound"], ascending=[False, False])["ticker"]:
        if t in picked:
            continue
        if is_valid_symbol_yf(t):
            picked.append(t)
        if len(picked) >= 15:
            break
    return sent_df[sent_df["ticker"].isin(picked)].copy()

# -------------------- Market enrichment --------------------
def fetch_market_block(ticker, lookback_days=220, retries=1):
    """RSI/SMA/volume snapshot with small retry."""
    data = None
    for i in range(retries + 1):
        try:
            end = dt.datetime.utcnow()
            start = end - dt.timedelta(days=lookback_days)
            data = yf.download(
                ticker, start=start.date(), end=end.date(),
                progress=False, auto_adjust=True, threads=False
            )
            if not data.empty:
                break
        except Exception:
            pass
        time.sleep(0.5 * (i + 1))

    if data is None or data.empty:
        return None

    close = data["Close"]
    vol   = data["Volume"]

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI(14)
    delta = close.diff()
    gain  = (delta.clip(lower=0)).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))

    vol20 = vol.rolling(20).mean()
    last = data.iloc[-1]
    last_close = float(last["Close"])
    last_vol   = float(last["Volume"])

    return {
        "price": last_close,
        "change_d1_pct": (last_close / float(data["Close"].iloc[-2]) - 1) * 100 if len(data) > 1 else np.nan,
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan,
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else np.nan,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else np.nan,
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else np.nan,
        "vol_spike": float(last_vol / vol20.iloc[-1]) if (vol20.iloc[-1] and vol20.iloc[-1] > 0) else np.nan,
        "data_points": len(data),
    }

@st.cache_data(ttl=900)
def enrich_with_market(sent_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Safely enrich sentiment rows with market data via yfinance."""
    if sent_df.empty:
        return sent_df

    tickers = list(sent_df["ticker"].head(top_n))
    market_rows = []

    prog = st.progress(0)
    total = max(1, len(tickers))
    for i, t in enumerate(tickers, start=1):
        block = fetch_market_block(t)
        if block:
            market_rows.append({"ticker": t, **block})
        prog.progress(i / total)
        time.sleep(0.05)

    if not market_rows:
        st.warning("No market data found for selected tickers (weekend/holiday/delisted?).")
        return sent_df

    mdf = pd.DataFrame(market_rows)
    if "ticker" not in sent_df.columns or "ticker" not in mdf.columns:
        st.error("Ticker column missing in one of the dataframes.")
        return sent_df

    merged = pd.merge(sent_df, mdf, on="ticker", how="left", validate="1:1")
    return merged

# -------------------- Scoring + classification --------------------
def composite_score(row):
    comp = 0.0
    # sentiment (-1..1) -> 0..1
    s = (row.get("compound", 0) + 1) / 2
    comp += 0.35 * s
    # momentum: RSI closeness to 50
    rsi = row.get("rsi", np.nan)
    if not np.isnan(rsi):
        comp += 0.30 * (1 - min(abs(rsi - 50), 50) / 50.0)
    # trend: price near SMA50
    price, sma50 = row.get("price", np.nan), row.get("sma50", np.nan)
    if not (np.isnan(price) or np.isnan(sma50) or sma50 == 0):
        dist = abs(price - sma50) / sma50
        comp += 0.20 * (1 - min(dist, 0.5) / 0.5)
    # volume spike (cap 3x)
    vs = row.get("vol_spike", np.nan)
    if not np.isnan(vs):
        comp += 0.15 * min(max(vs, 0), 3) / 3.0
    return round(float(comp), 4)

def classify_zone(row):
    rsi = row.get("rsi", np.nan)
    price, sma20 = row.get("price", np.nan), row.get("sma20", np.nan)
    near_20 = False
    if not (np.isnan(price) or np.isnan(sma20) or sma20 == 0):
        near_20 = abs(price - sma20) / sma20 <= 0.02  # within 2%
    if not np.isnan(rsi):
        if 45 <= rsi <= 60 and near_20:
            return "prime"
        elif rsi >= 70 or (not np.isnan(sma20) and (price - sma20) / sma20 >= 0.10):
            return "stretched"
    return "warm"

def ai_decision(zone, sentiment_compound):
    if zone == "prime" and sentiment_compound >= 0.05:
        return "BUY"
    if zone == "stretched" or sentiment_compound < -0.05:
        return "AVOID"
    return "HOLD"

# -------------------- UI --------------------
st.set_page_config(page_title="Hotlist AI — Mobile", page_icon="📈", layout="wide")
st.title("📈 Hotlist AI — Mobile")
st.caption("Auto-discovers top 15 tickers from Reddit with Pushshift (with fallback), then adds sentiment + technicals. No Reddit login needed.")

with st.sidebar:
    st.header("Settings")

    subs_raw = st.text_input(
        "Subreddits (comma or space separated)",
        value=", ".join(DEFAULT_SUBREDDITS),
        help="Example: stocks, pennystocks",
    )
    subreddits = [s.strip() for s in re.split(r"[,\s]+", subs_raw) if s.strip()]

    auto_discover = st.toggle("Auto-discover top 15 (recent window)", value=True)

    limit_per = st.slider("Items per subreddit (posts + comments)", 50, 500, 150, step=25)

    st.write("---")
    st.write("Data source: PullPush → fallback to Pushshift. Market data: Yahoo Finance.")

# -------- Step 1: Social scan with resilient window/fallback --------
st.subheader("Step 1 · Reddit-like scan (Pushshift w/ fallback)")
after_ts = last_two_market_days_start_ts() if auto_discover else None
sent_df = scan_pushshift(subreddits, limit_per, whitelist=None, after_ts=after_ts)

# If still empty, broaden to ~5 days and try again (once)
if sent_df.empty and auto_discover:
    st.info("No data found in last 2 market days. Broadening window to ~5 days…")
    after_ts_wide = five_day_window_start_ts()
    sent_df = scan_pushshift(subreddits, limit_per, whitelist=None, after_ts=after_ts_wide)

# Keep only top 15 valid tickers (Yahoo validated)
if auto_discover and not sent_df.empty:
    sent_df = keep_top15_valid(sent_df)

st.dataframe(sent_df, use_container_width=True)
if sent_df.empty:
    st.stop()

# -------- Step 2: Market enrichment --------
st.subheader("Step 2 · Market enrichment")
full = enrich_with_market(sent_df, top_n=25)  # enrich a bit more than 15
if full.empty:
    st.stop()

# -------- Step 3: Scoring + ranked output --------
full["Composite"] = full.apply(composite_score, axis=1)
full["BuyZone"] = full.apply(classify_zone, axis=1)
full["AI_Decision"] = [ai_decision(z, c) for z, c in zip(full["BuyZone"], full["compound"].fillna(0.0))]

cols = [
    "ticker","Composite","mentions","compound","rsi","vol_spike",
    "price","change_d1_pct","sma20","sma50","sma200","BuyZone","AI_Decision"
]
present = [c for c in cols if c in full.columns]
table = full[present].sort_values("Composite", ascending=False).reset_index(drop=True)

st.subheader("Step 3 · Ranked output")
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    csv,
    file_name=f"hotlist_{dt.datetime.utcnow().strftime('%Y-%m-%d_%H%M')}.csv",
    mime="text/csv"
)

st.caption("Composite = sentiment (35%) + RSI-to-50 (30%) + price≈SMA50 (20%) + volume spike (15%). "
           "Zones: Prime (RSI 45–60 & near SMA20), Stretched (RSI≥70 or ≥10% above SMA20), otherwise Warm.")
