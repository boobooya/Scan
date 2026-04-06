import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

TICKERS = [  # shortened example, keep your full list
"SPY","QQQ","IWM","AAPL","MSFT","NVDA","TSLA","AMD","META"
]

# -----------------------
# VOLATILITY
# -----------------------
def realized_vol(stock):
    hist = stock.history(period="1mo")["Close"]
    returns = hist.pct_change().dropna()
    return returns.std() * np.sqrt(252)

# -----------------------
# EXPECTED MOVE
# -----------------------
def expected_move(price, iv, dte):
    return price * iv * np.sqrt(dte/365)

# -----------------------
# PROBABILITY (IMPROVED)
# -----------------------
def probability(price, target, vol, dte):
    std = price * vol * np.sqrt(dte/365)
    z = (target - price) / (std + 1e-6)
    return max(0, min(1, np.exp(-0.5*z*z)))

# -----------------------
# PROCESS ONE TICKER
# -----------------------
def process_ticker(ticker):

    results = []

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty:
            return results

        price = hist["Close"].iloc[-1]
        rv = realized_vol(stock)

        for exp in stock.options:

            dte = (datetime.strptime(exp,"%Y-%m-%d") - datetime.today()).days
            if dte <= 0 or dte > 7:
                continue

            chain = stock.option_chain(exp)

            for df, typ in [(chain.calls,"CALL"), (chain.puts,"PUT")]:

                for _, row in df.iterrows():

                    iv = row["impliedVolatility"]
                    bid = row["bid"]
                    ask = row["ask"]
                    strike = row["strike"]
                    oi = row["openInterest"]

                    if iv < 0.05 or bid == 0 or ask == 0:
                        continue

                    spread = ask - bid
                    if spread <= 0:
                        continue

                    em = expected_move(price, iv, dte)

                    # 1 STD DEV FILTER
                    if abs(strike - price) > em:
                        continue

                    # IMPROVED FEATURES
                    vol_edge = (rv - iv)
                    liquidity = np.log(oi + 1) / (spread + 0.01)
                    distance = abs(strike - price) / price

                    # GAMMA PROXY IMPROVED
                    gamma_proxy = oi / (distance + 0.01)

                    # FINAL SCORE
                    score = (
                        vol_edge * 4 +
                        liquidity * 1.5 +
                        gamma_proxy / 200
                    )

                    proj = price + em if typ=="CALL" else price - em
                    prob = probability(price, proj, rv, dte)

                    results.append({
                        "ticker": ticker,
                        "type": typ,
                        "strike": round(strike,2),
                        "expiration": exp,
                        "price": round(price,2),
                        "expected_move": round(em,2),
                        "projected_price": round(proj,2),
                        "probability": round(prob,3),
                        "score": round(score,2),
                        "dte": dte
                    })

        print(f"✔ {ticker}")

    except Exception as e:
        print(f"❌ {ticker}", e)

    return results


# -----------------------
# PARALLEL EXECUTION
# -----------------------
all_results = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = executor.map(process_ticker, TICKERS)

    for res in futures:
        all_results.extend(res)

df = pd.DataFrame(all_results)

if not df.empty:
    df = df.sort_values("score", ascending=False)

df.to_json("signals.json", orient="records", indent=2)

print("\n🔥 DONE — TOP PICKS:\n")
print(df.head(20))
