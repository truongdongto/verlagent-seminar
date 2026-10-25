"""Download US daily prices and compute technical features.

Usage (from the verl-agent root):
    python -m fintrade.data --out data/fin
    python -m fintrade.data --tickers AAPL NVDA --start 2014-06-01 --end 2026-10-01

Indicators follow FinRL-Meta's default list (meta/config.py: INDICATORS) and are
computed with stockstats, so TA-Lib is not required. Raw indicators are then
turned into scale-free features so that one prompt template fits every ticker.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

# FinRL-Meta default indicator list
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

ID_TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM"]
OOD_TICKERS = ["TSLA", "AMD", "KO", "UNH"]

# columns consumed by fintrade.sim
FEATURE_COLUMNS = ["ret", "sma30_gap", "sma60_gap", "rsi", "macd_pct", "cci", "dx", "bb_pos", "vol_ratio"]


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.dropna()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    from stockstats import wrap

    stock = wrap(df.copy())
    out = df.copy()
    for name in INDICATORS:
        out[name] = stock[name].values

    close = out["close"]
    band = (out["boll_ub"] - out["boll_lb"]).replace(0, np.nan)
    out["ret"] = close.pct_change()
    out["sma30_gap"] = close / out["close_30_sma"] - 1.0
    out["sma60_gap"] = close / out["close_60_sma"] - 1.0
    out["rsi"] = out["rsi_30"]
    out["macd_pct"] = out["macd"] / close * 100.0
    out["cci"] = out["cci_30"]
    out["dx"] = out["dx_30"]
    out["bb_pos"] = (close - out["boll_lb"]) / band
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(20).mean()

    # the first ~60 sessions have incomplete moving averages
    out = out.iloc[60:].replace([np.inf, -np.inf], np.nan)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].ffill().fillna(0.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=ID_TICKERS + OOD_TICKERS)
    parser.add_argument("--start", default="2014-06-01", help="includes warm-up for 60-day indicators")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--out", default="data/fin")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i, ticker in enumerate(args.tickers):
        df = add_features(download(ticker, args.start, args.end))
        path = os.path.join(args.out, f"{ticker}.parquet")
        df.to_parquet(path)
        print(f"{ticker}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()} -> {path}")
        if i < len(args.tickers) - 1:
            time.sleep(1.0)  # be gentle with the Yahoo endpoint


if __name__ == "__main__":
    main()
