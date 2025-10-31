# -----------------------------
# Hotlist AI — Mobile (Streamlit)
# Reddit mentions + sentiment + technicals (RSI, SMAs, volume spike)
# -----------------------------

import os, re, time, math, datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Market data (no API key)
import yfinance as yf

# Reddit API (needs keys in Streamlit secrets)
import praw

# ------- One-time NLTK bootstrap (cached on Streamlit Cloud) -------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ------- Defaults -------
DEFAULT_SUBREDDITS = ["stocks", "pennystocks", "themadinvestor", "TheRaceTo10Million"]
TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")

DEFAULT_TICKER_WHITELIST = """
AAPL, MSFT, AMZN, NVDA, TSLA, META, GOOGL, AMD, AVGO, NFLX,
BYND, DFLI, GME, AMC, PLTR, SOFI, INTC, MU, TSM, ARM, CAT, BA, SHOP, UBER, ABNB
"""

# -------------------- Helpers --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_reddit_client():
    secrets = st.secrets.get("reddit", {})
    required = {"client_id", "client_secret", "user_agent"}
    if not (required <= set(secrets.keys())):
        return None, "Reddit API keys missing in Secrets. Add [reddit] client_id, client_secret, user_agent."
    try:
        reddit = praw.Reddit(
            client_id=secrets["client_id"],
            client_secret=secrets["client_secret"],
            user_agent=secrets["user_agent"],
        )
        return reddit, None
    except Exception as e:
        return None, f"Failed to init Reddit client: {e}"

def clean_whitelist(raw: str):
    return sorted({t.strip().upper() for t in re.split(r"[,\s]+", raw) if t.strip()})

def extract_tickers(text: str, whitelist):
    if not text:
        return []
    found = set(re.findall(TICKER_PATTERN, text.upper()))
    return [t for t in found if t in whitelist]

@st.cache_data(ttl=900)
def scan_reddit(subreddits, limit_per_sub, whitelist):
    """Return dataframe of mentions & sentiment by ticker."""
    reddit, err = get_reddit_client()
    if err:
        st.warning(err)
        return pd.DataFrame(columns=["ticker","mentions","pos","neg","neu","compound"])

    sid = SentimentIntensityAnalyzer()
    rows = []
    for sub in subreddits:
        try:
            for post in reddit.subreddit(sub).hot(limit=limit_per_sub):
                body = f"{post.title}\n{post.selftext or ''}"
                tickers = extract_tickers(body, whitelist)
                if not tickers:
                    continue
                s = sid.polarity_scores(body)
                for t in tickers:
                    rows.append({"ticker": t, **s})
        except Exception as e:
            st.info(f"Skipping r/{sub} ({e})")

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
def fetch_market_block(ticker, lookback_days=220):
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)
    data = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
    if data.empty:
        return None

    close = data["Close"]
    vol = data["Volume"]

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI(14)
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    vol20 = vol.rolling(20).mean()
    last = data.iloc[-1]
    last_close = float(last["Close"])
    last_vol = float(last["Volume"])

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
def enrich_with_market(sent_df, top_n=25):
    if sent_df.empty:
        return sent_df
    tickers = list(sent_df["ticker"].head(top_n))
    market_rows = []
    for t in tickers:
        block = fetch_market_block(t)
        if block:
            market_rows.append({"ticker": t, **block})
        time.sleep(0.1)  # small politeness delay
    mdf = pd.DataFrame(market_rows)
    return sent_df.merge(mdf, on="ticker", how="left")

def composite_score(row):
    # Weighted blend into 0..1-ish
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
        dist = abs(price - sma50) / sma50  # smaller better
        comp += 0.20 * (1 - min(dist, 0.5) / 0.5)

    # volume spike (cap at 3x)
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
st.caption("Ranks tickers using Reddit mentions + sentiment + market technicals (RSI, SMAs, volume spike). Optimized for phones.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    subs_raw = st.text_input(
        "Subreddits (comma or space separated)",
        value=", ".join(DEFAULT_SUBREDDITS),
        help="Example: stocks, pennystocks"
    )
    subreddits = [s.strip() for s in re.split(r"[,\s]+", subs_raw) if s.strip()]

    limit_per = st.slider("Posts per subreddit", 50, 500, 150, step=25)

    wl_text = st.text_area(
        "Ticker whitelist (comma or space separated)",
        value=DEFAULT_TICKER_WHITELIST,
        height=140
    )
    whitelist = clean_whitelist(wl_text)

    top_n = st.slider("Top N by mentions to enrich", 5, 50, 20, step=5)

    st.write("---")
    st.write("**Secrets needed** in Streamlit → Settings → Secrets:\n`[reddit] client_id, client_secret, user_agent`")
    st.write("Market data via Yahoo Finance (no key).")

if not whitelist:
    st.warning("Please provide at least one ticker in the whitelist.")
    st.stop()

# Step 1: Reddit scan
st.subheader("Step 1 · Reddit scan")
sent_df = scan_reddit(subreddits, limit_per, whitelist)
st.dataframe(sent_df, use_container_width=True)
if sent_df.empty:
    st.stop()

# Step 2: Market enrichment
st.subheader("Step 2 · Market enrichment")
full = enrich_with_market(sent_df, top_n=top_n)
if full.empty:
    st.info("No market data resolved for the selected tickers.")
    st.stop()

# Step 3: Composite + classifications
full["Composite"] = full.apply(composite_score, axis=1)
full["BuyZone"] = full.apply(classify_zone, axis=1)
full["AI_Decision"] = [
    ai_decision(z, c) for z, c in zip(full["BuyZone"], full["compound"].fillna(0.0))
]

# Present
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

st.caption("Heuristics: Composite blends sentiment (35%), RSI-to-50 (30%), price≈SMA50 (20%), and volume spike (15%). Zones: Prime (RSI 45–60 & near SMA20), Stretched (RSI≥70 or ≥10% above SMA20), otherwise Warm.")
