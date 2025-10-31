# Hotlist AI — Mobile (Reliable Reddit API version)
# Official Reddit API (PRAW) -> auto-discover last-2-market-days top 15 tickers
# + VADER sentiment + RSI/SMA/Volume (Yahoo) + composite ranking

import os, re, time, datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    test_reddit = praw.Reddit(
        client_id=st.secrets["reddit"]["client_id"],
        client_secret=st.secrets["reddit"]["client_secret"],
        user_agent=st.secrets["reddit"]["user_agent"],
    )
    test_post = next(test_reddit.subreddit("stocks").hot(limit=1))
    st.sidebar.success(f"✅ Reddit connected: {test_post.title[:40]}...")
except Exception as e:
    st.sidebar.error(f"❌ Reddit auth failed: {e}")


# --- Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Reddit API
import praw
from prawcore import RequestException, ResponseException, PrawcoreException

# Bootstrap VADER lexicon once
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ------------------ Config ------------------
DEFAULT_SUBS = ["stocks", "pennystocks", "themadinvestor", "TheRaceTo10Million"]
TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
STOP_TICKER_WORDS = {
    "USD","CEO","DD","ATH","IPO","ETF","AI","YOLO","OTC","TOS","API",
    "GDP","CPI","FOMC","SEC","DTCC","RSI","SMA","EMA","EPS","PE","PEG"
}
REDDIT_HEADERS = {"User-Agent": st.secrets["reddit"]["user_agent"]}

# ------------- Time window helpers -------------
def start_of_earlier_of_last_two_bdays(tz="US/Eastern") -> int:
    now_local = pd.Timestamp.now(tz).normalize()
    found = 0; day = 0; start = now_local
    while found < 2:
        d = now_local - pd.Timedelta(days=day)
        if d.weekday() < 5:  # Mon–Fri
            start = d
            found += 1
        day += 1
    return int(start.tz_convert("UTC").timestamp())

def five_days_ago_ts() -> int:
    return int((pd.Timestamp.utcnow() - pd.Timedelta(days=5)).timestamp())

# ------------- Reddit client -------------
def reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=st.secrets["reddit"]["client_id"],
        client_secret=st.secrets["reddit"]["client_secret"],
        user_agent=st.secrets["reddit"]["user_agent"],
        ratelimit_seconds=5,
    )

# ------------- Extraction helpers -------------
def extract_tickers(text: str, whitelist: Optional[List[str]]) -> List[str]:
    if not text:
        return []
    found = set(re.findall(TICKER_RE, text.upper()))
    if whitelist is None:
        return [t for t in found if t not in STOP_TICKER_WORDS and 2 <= len(t) <= 5]
    return [t for t in found if t in whitelist]

# ------------- Source fallback (last resort) -------------
def reddit_public_json(subreddit: str, limit: int = 100) -> List[str]:
    texts = []
    for feed in ("new", "hot"):
        url = f"https://www.reddit.com/r/{subreddit}/{feed}.json"
        j = requests.get(url, params={"limit": min(100, max(1, limit))},
                         headers=REDDIT_HEADERS, timeout=15)
        if j.status_code != 200:
            continue
        data = j.json().get("data", {})
        for c in data.get("children", []):
            d = c.get("data", {})
            title = d.get("title") or ""
            body  = d.get("selftext") or ""
            if title or body:
                texts.append(f"{title}\n{body}")
        time.sleep(0.2)
    return texts

# ------------- Collect posts & comments (PRAW) -------------
@st.cache_data(ttl=900)
def fetch_texts_for_sub(
    sub: str,
    after_ts: int,
    per_sub_limit: int = 150
) -> List[str]:
    """
    Use Reddit API to pull submissions (new) + comments since after_ts.
    If nothing is returned (rare), fall back to public JSON for submissions.
    """
    rd = reddit_client()
    s = rd.subreddit(sub)
    texts: List[str] = []

    # Submissions: stream newest and filter by created_utc
    got = 0
    try:
        for post in s.new(limit=1000):
            if post.created_utc < after_ts:
                break
            title = post.title or ""
            body  = post.selftext or ""
            if title or body:
                texts.append(f"{title}\n{body}")
                got += 1
            if got >= per_sub_limit:
                break
    except (RequestException, ResponseException, PrawcoreException):
        pass

    # Comments: grab recent comments and filter by created_utc
    got_c = 0
    try:
        for c in s.comments(limit=per_sub_limit):
            if getattr(c, "created_utc", 0) < after_ts:
                continue
            body = getattr(c, "body", "") or ""
            if body:
                texts.append(body)
                got_c += 1
            if got_c >= per_sub_limit:
                break
    except (RequestException, ResponseException, PrawcoreException):
        pass

    # If still empty, last resort: public JSON (submissions only)
    if not texts:
        texts = reddit_public_json(sub, limit=min(100, per_sub_limit))

    return texts

# ------------- Scan + aggregate -------------
@st.cache_data(ttl=900)
def scan_reddit(subs: List[str], after_ts: int, per_sub: int, whitelist=None) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    rows = []
    prog = st.progress(0)
    total = max(1, len(subs))
    for i, sub in enumerate(subs, start=1):
        texts = fetch_texts_for_sub(sub, after_ts=after_ts, per_sub_limit=per_sub)
        if not texts:
            st.info(f"Skipping r/{sub} (no data from Reddit)")
        else:
            for t in texts:
                for tk in extract_tickers(t, whitelist):
                    s = sid.polarity_scores(t)
                    rows.append({"ticker": tk, **s})
        prog.progress(i/total)
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

# ------------- Validation + markets -------------
@st.cache_data(ttl=900)
def is_valid_symbol_yf(ticker: str) -> bool:
    try:
        end = dt.datetime.utcnow(); start = end - dt.timedelta(days=30)
        df = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True, threads=False)
        return not df.empty
    except Exception:
        return False

def keep_top15_valid(sent_df: pd.DataFrame) -> pd.DataFrame:
    if sent_df.empty:
        return sent_df
    picked = []
    for t in sent_df["ticker"]:
        if t in picked: 
            continue
        if is_valid_symbol_yf(t):
            picked.append(t)
        if len(picked) >= 15:
            break
    return sent_df[sent_df["ticker"].isin(picked)].copy()

def fetch_market_block(ticker, lookback_days=220, retries=1):
    data = None
    for i in range(retries + 1):
        try:
            end = dt.datetime.utcnow(); start = end - dt.timedelta(days=lookback_days)
            data = yf.download(ticker, start=start.date(), end=end.date(),
                               progress=False, auto_adjust=True, threads=False)
            if not data.empty: break
        except Exception: pass
        time.sleep(0.5*(i+1))
    if data is None or data.empty:
        return None
    close = data["Close"]; vol = data["Volume"]
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    delta = close.diff(); gain=(delta.clip(lower=0)).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss; rsi = 100 - (100/(1+rs))
    vol20 = vol.rolling(20).mean()
    last = data.iloc[-1]
    return {
        "price": float(last["Close"]),
        "change_d1_pct": (float(last["Close"])/float(data["Close"].iloc[-2]) - 1)*100 if len(data)>1 else np.nan,
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan,
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else np.nan,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else np.nan,
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else np.nan,
        "vol_spike": float(last["Volume"]/vol20.iloc[-1]) if (vol20.iloc[-1] and vol20.iloc[-1]>0) else np.nan,
        "data_points": len(data)
    }

@st.cache_data(ttl=900)
def enrich_with_market(sent_df: pd.DataFrame, top_n=25) -> pd.DataFrame:
    if sent_df.empty: return sent_df
    market_rows = []; prog = st.progress(0); tickers = list(sent_df["ticker"].head(top_n))
    for i,t in enumerate(tickers, start=1):
        blk = fetch_market_block(t)
        if blk: market_rows.append({"ticker": t, **blk})
        prog.progress(i/len(tickers)); time.sleep(0.05)
    if not market_rows:
        st.warning("No market data found for selected tickers (weekend/holiday/delisted?).")
        return sent_df
    mdf = pd.DataFrame(market_rows)
    return pd.merge(sent_df, mdf, on="ticker", how="left", validate="1:1")

# ------------- Scoring -------------
def composite_score(row):
    comp=0.0
    comp += 0.35*((row.get("compound",0)+1)/2)   # sentiment
    rsi=row.get("rsi", np.nan)
    if not np.isnan(rsi): comp += 0.30*(1 - min(abs(rsi-50),50)/50)
    price=row.get("price",np.nan); sma50=row.get("sma50",np.nan)
    if not (np.isnan(price) or np.isnan(sma50) or sma50==0):
        comp += 0.20*(1 - min(abs(price-sma50)/sma50, 0.5)/0.5)
    vs=row.get("vol_spike",np.nan)
    if not np.isnan(vs): comp += 0.15*min(max(vs,0),3)/3
    return round(float(comp),4)

def classify_zone(row):
    rsi=row.get("rsi",np.nan); price=row.get("price",np.nan); sma20=row.get("sma20",np.nan)
    near_20=False
    if not (np.isnan(price) or np.isnan(sma20) or sma20==0):
        near_20=abs(price-sma20)/sma20 <= 0.02
    if not np.isnan(rsi):
        if 45<=rsi<=60 and near_20: return "prime"
        elif rsi>=70 or (not np.isnan(sma20) and (price-sma20)/sma20>=0.10): return "stretched"
    return "warm"

def ai_decision(zone, comp_sent):
    if zone=="prime" and comp_sent>=0.05: return "BUY"
    if zone=="stretched" or comp_sent<-0.05: return "AVOID"
    return "HOLD"

# ------------------ UI ------------------
st.set_page_config(page_title="Hotlist AI — Mobile", page_icon="📈", layout="wide")
st.title("📈 Hotlist AI — Mobile")
st.caption("Official Reddit API (reliable) → last-2-market-days → auto-discover 15 tickers → sentiment + technicals.")

with st.sidebar:
    st.header("Settings")
    subs_raw = st.text_input("Subreddits (comma/space separated)", value=", ".join(DEFAULT_SUBS))
    subs = [s.strip() for s in re.split(r"[,\s]+", subs_raw) if s.strip()]
    limit_per = st.slider("Items per subreddit (posts + comments)", 50, 500, 150, step=25)
    st.write("---")
    st.write("Data: Reddit API (PRAW). Fallback: reddit JSON. Market: Yahoo Finance.")

# Step 1: Reddit scan (last 2 market days; widen to 5 days if truly empty)
st.subheader("Step 1 · Reddit scan")
after_ts = start_of_earlier_of_last_two_bdays()
sent_df = scan_reddit(subs, after_ts=after_ts, per_sub=limit_per, whitelist=None)
if sent_df.empty:
    st.info("No data in last 2 market days — widening to ~5 days…")
    sent_df = scan_reddit(subs, after_ts=five_days_ago_ts(), per_sub=limit_per, whitelist=None)

# Keep top 15 valid tickers
if not sent_df.empty:
    sent_df = keep_top15_valid(sent_df)

st.dataframe(sent_df, use_container_width=True)
if sent_df.empty: st.stop()

# Step 2: Market enrichment
st.subheader("Step 2 · Market enrichment")
full = enrich_with_market(sent_df, top_n=25)
if full.empty: st.stop()

# Step 3: Score + rank
full["Composite"] = full.apply(composite_score, axis=1)
full["BuyZone"]   = full.apply(classify_zone, axis=1)
full["AI_Decision"] = [ai_decision(z, c) for z,c in zip(full["BuyZone"], full["compound"].fillna(0.0))]

cols = ["ticker","Composite","mentions","compound","rsi","vol_spike","price",
        "change_d1_pct","sma20","sma50","sma200","BuyZone","AI_Decision"]
table = full[[c for c in cols if c in full.columns]].sort_values("Composite", ascending=False).reset_index(drop=True)

st.subheader("Step 3 · Ranked output")
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv,
    file_name=f"hotlist_{dt.datetime.utcnow().strftime('%Y-%m-%d_%H%M')}.csv",
    mime="text/csv")

st.caption("Composite = sentiment (35%) + RSI-to-50 (30%) + price≈SMA50 (20%) + volume spike (15%).")
