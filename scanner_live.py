import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import time

# YOUR FULL UNIVERSE
TICKERS = [
"SPY","QQQ","IWM","DIA","EEM","EFA","VXX","AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX",
"AMD","INTC","CRM","ORCL","ADBE","CSCO","AVGO","QCOM","TXN","MU","AMAT","LRCX","SNOW","NOW","DDOG",
"NET","ZS","CRWD","WDAY","JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","USB","PNC","TFC","COF",
"SOFI","WMT","HD","TGT","COST","NKE","SBUX","MCD","DIS","CMCSA","BKNG","ABNB","UBER","ETSY","SHOP",
"EBAY","LOW","JNJ","UNH","PFE","ABBV","TMO","ABT","MRK","LLY","AMGN","GILD","BMY","CVS","REGN","VRTX",
"ISRG","OXY","CVX","XOM","COP","SLB","HAL","MPC","PSX","VLO","EOG","GLD","SLV","GDX","NEM","FCX",
"BA","CAT","GE","MMM","HON","UPS","RTX","LMT","NOC","GD","DE","FDX","DAL","UAL","LUV","AAL","TSM",
"MCHP","NXPI","F","GM","RIVN","NIO","PLUG","T","VZ","TMUS","SNAP","PINS","RBLX","PYPL","SQ","V","MA",
"COIN","PLTR","SPG","PLD","AMT","CCI","EQIX","PSA","BABA","JD","PDD","NEE","DUK","SO","PG","KO","PEP",
"CCL","RCL","MAR","HLT","SQQQ","TQQQ","SPXL","TLT","HYG","JNK","USO","MARA","RIOT","ZM","DKNG","ROKU","ARKK"
]

def realized_vol(stock):
    hist = stock.history(period="1mo")["Close"]
    returns = hist.pct_change().dropna()
    return returns.std() * np.sqrt(252)

def expected_move(price, iv, dte):
    return price * iv * np.sqrt(dte/365)

def probability(price, target, vol, dte):
    std = price * vol * np.sqrt(dte/365)
    z = (target - price)/std
    return np.exp(-0.5*z*z)

results = []

for ticker in TICKERS:

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty:
            continue

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

                    if iv == 0 or bid == 0 or ask == 0:
                        continue

                    spread = ask - bid
                    em = expected_move(price, iv, dte)

                    # 1 STD DEV FILTER
                    if abs(strike - price) > em:
                        continue

                    # GAMMA PROXY
                    gamma_proxy = oi / (abs(strike - price) + 1)

                    # SCORE
                    score = (
                        (rv - iv)*3 +
                        (1/(spread+0.01)) +
                        np.log(oi+1) +
                        gamma_proxy/100
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

        print(f"✔ {ticker} done")

        # Avoid rate limit
        time.sleep(1)

    except Exception as e:
        print(f"❌ {ticker} error:", e)

df = pd.DataFrame(results).sort_values("score", ascending=False)

df.to_json("signals.json", orient="records", indent=2)

print("\n🔥 TOP 20 SIGNALS:")
print(df.head(20))
