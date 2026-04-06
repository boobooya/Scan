import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

# Your full ticker list
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
    try:
        hist = stock.history(period="1mo")["Close"]
        returns = hist.pct_change().dropna()
        return returns.std() * np.sqrt(252)
    except:
        return 0.2  # fallback

def expected_move(price, iv, dte):
    return price * iv * np.sqrt(dte/365)

def probability(price, target, vol, dte):
    try:
        std = price * vol * np.sqrt(dte/365)
        z = (target - price) / (std + 1e-6)
        return max(0, min(1, np.exp(-0.5*z*z)))
    except:
        return 0.5

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
            try:
                dte = (datetime.strptime(exp,"%Y-%m-%d") - datetime.today()).days
                if dte <=0 or dte > 7:
                    continue
                try:
                    chain = stock.option_chain(exp)
                except:
                    continue
                for df, typ in [(chain.calls,"CALL"), (chain.puts,"PUT")]:
                    for _, row in df.iterrows():
                        try:
                            iv = row["impliedVolatility"]
                            bid = row["bid"]
                            ask = row["ask"]
                            strike = row["strike"]
                            oi = row["openInterest"]
                            if iv < 0.05 or bid==0 or ask==0:
                                continue
                            spread = ask-bid
                            if spread<=0: continue
                            em = expected_move(price, iv, dte)
                            if abs(strike-price) > em: continue
                            vol_edge = (rv - iv)
                            liquidity = np.log(oi+1)/(spread+0.01)
                            distance = abs(strike-price)/price
                            gamma_proxy = oi/(distance+0.01)
                            score = vol_edge*4 + liquidity*1.5 + gamma_proxy/200
                            proj = price+em if typ=="CALL" else price-em
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
                        except: continue
            except: continue
        print(f"✔ {ticker}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ {ticker} error:", e)
        traceback.print_exc()
    return results

all_results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    for res in executor.map(process_ticker, TICKERS):
        all_results.extend(res)

df = pd.DataFrame(all_results)
if not df.empty:
    df = df.sort_values("score", ascending=False)
df.to_json("signals.json", orient="records", indent=2)

print("\n🔥 DONE — TOP 20 SIGNALS:\n")
print(df.head(20))
