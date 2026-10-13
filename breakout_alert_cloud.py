"""
NSE breakout alert -> Telegram
Stocks: Gandhar Oil Refinery, N R Agarwal Industries, Uttam Sugar Mills

Checks your 6 confirmations on 5-minute data:
  1. Price above previous day's high
  2. Price above the first 5-min candle's high (09:15 candle)
  3. Daily EMA 10 > 20 > 50 (bullish alignment)
  4. Daily ADX(14) above 25
  5. Price above VWAP
  6. Latest 5-min bar volume above 1.5x the average 5-min bar volume (last 5 days)

When all 6 pass, it sends ONE Telegram message per stock per day with
entry, stop and targets calculated from live data.

Setup:  pip install yfinance pandas requests tzdata
Run:    python breakout_alert.py          (normal, only inside market hours)
Test:   python breakout_alert.py --test   (ignores market hours, sends a test ping,
                                           and prints the status of every check)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ----------------------------- CONFIG ---------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")      # set as a GitHub secret
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")      # set as a GitHub secret

# VERIFY these tickers on Yahoo Finance / TradingView before relying on them.
WATCHLIST = {
    "Gandhar Oil Refinery": "GANDHAR.NS",
    "N R Agarwal Industries": "NRAIL.NS",
    "Uttam Sugar Mills": "UTTAMSUGAR.NS",
}

VOLUME_MULT = 1.5
ADX_MIN = 25
RISK_REWARD = (2.0, 3.0)     # target 1 and target 2, in multiples of risk
STATE_FILE = Path(__file__).with_name("alert_state.json")
IST = ZoneInfo("Asia/Kolkata")
# -----------------------------------------------------------------------------


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=a, adjust=False).mean()


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    r.raise_for_status()


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state))


def in_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return datetime.strptime("09:20", "%H:%M").time() <= t <= datetime.strptime("15:30", "%H:%M").time()


def check_stock(name: str, symbol: str):
    """Returns (passed: bool, details: dict) or (False, {'error': ...})."""
    tk = yf.Ticker(symbol)
    daily = tk.history(period="1y", interval="1d")
    intra = tk.history(period="5d", interval="5m")
    if daily.empty or intra.empty:
        return False, {"error": "no data returned (check ticker / connection)"}

    intra.index = intra.index.tz_convert(IST)
    daily.index = daily.index.tz_localize(None) if daily.index.tz is None else daily.index.tz_convert(IST).tz_localize(None)

    today = intra.index[-1].date()
    today_df = intra[intra.index.date == today]
    past_intra = intra[intra.index.date < today]
    hist = daily[daily.index.date < today]          # completed daily candles only
    if len(hist) < 60 or past_intra.empty:
        return False, {"error": "not enough history"}
    if today_df.index[0].strftime("%H:%M") != "09:15":
        return False, {"error": "first 5-min candle missing (data delayed?)"}

    price = float(today_df["Close"].iloc[-1])
    prev_high = float(hist["High"].iloc[-1])
    first_high = float(today_df["High"].iloc[0])

    ema10 = hist["Close"].ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = hist["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
    adx_val = float(adx(hist).iloc[-1])

    typical = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
    vwap = float((typical * today_df["Volume"]).cumsum().iloc[-1] / max(today_df["Volume"].cumsum().iloc[-1], 1))

    last_vol = float(today_df["Volume"].iloc[-1])
    avg_vol = float(past_intra["Volume"].mean())

    checks = {
        "1. Above prev day high": price > prev_high,
        "2. Above first 5-min high": price > first_high,
        "3. EMA 10>20>50": ema10 > ema20 > ema50,
        f"4. ADX>{ADX_MIN}": adx_val > ADX_MIN,
        "5. Above VWAP": price > vwap,
        f"6. Volume>{VOLUME_MULT}x avg": last_vol > VOLUME_MULT * avg_vol,
    }

    # Entry / stop / targets
    low10 = float(hist["Low"].tail(10).min())
    stop = max(float(ema20), low10)                 # whichever is closer (higher)
    if stop >= price:
        stop = low10
    risk = price - stop
    t1 = price + RISK_REWARD[0] * risk
    t2 = price + RISK_REWARD[1] * risk

    details = {
        "price": price, "prev_high": prev_high, "first_high": first_high,
        "ema10": float(ema10), "ema20": float(ema20), "ema50": float(ema50),
        "adx": adx_val, "vwap": vwap, "last_vol": last_vol, "avg_vol": avg_vol,
        "stop": stop, "risk_pct": 100 * risk / price, "t1": t1, "t2": t2,
        "checks": checks,
    }
    return all(checks.values()), details


def main() -> None:
    test = "--test" in sys.argv
    now = datetime.now(IST)

    if not test and not in_market_hours(now):
        return

    if test:
        send_telegram("Breakout alert script: Telegram connection works.")

    state = load_state()
    today_key = now.strftime("%Y-%m-%d")
    state = {k: v for k, v in state.items() if v == today_key}   # drop old days

    for name, symbol in WATCHLIST.items():
        try:
            passed, d = check_stock(name, symbol)
        except Exception as e:
            print(f"[{name}] error: {e}")
            continue

        if "error" in d:
            print(f"[{name}] {d['error']}")
            continue

        if test:
            print(f"\n{name} ({symbol})  price {d['price']:.2f}")
            for label, ok in d["checks"].items():
                print(f"   {'PASS' if ok else 'fail'}  {label}")

        if passed and state.get(symbol) != today_key:
            msg = (
                f"BREAKOUT: {name} ({symbol})\n"
                f"Time: {now.strftime('%d %b %H:%M')} IST\n"
                f"Entry (approx): {d['price']:.2f}\n"
                f"Stop: {d['stop']:.2f}  (risk {d['risk_pct']:.1f}%)\n"
                f"Target 1 ({RISK_REWARD[0]:.0f}R): {d['t1']:.2f}\n"
                f"Target 2 ({RISK_REWARD[1]:.0f}R): {d['t2']:.2f}\n"
                f"ADX {d['adx']:.1f} | VWAP {d['vwap']:.2f} | "
                f"Vol {d['last_vol'] / d['avg_vol']:.1f}x avg\n"
                f"Rule-based signal, not advice. Confirm on the chart before acting."
            )
            send_telegram(msg)
            state[symbol] = today_key

    save_state(state)


if __name__ == "__main__":
    main()
