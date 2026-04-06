import requests
import numpy as np
import pandas as pd
from datetime import datetime

TOKEN = "YOUR_TRADIER_API_KEY"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}

TICKERS = ["SPY","QQQ","AAPL","TSLA","NVDA"]

# -----------------------
# GET OPTIONS CHAIN
# -----------------------
def get_chain(symbol):

    url = f"https://api.tradier.com/v1/markets/options/chains"

    params = {
        "symbol": symbol,
        "expiration": "",  # will loop later
        "greeks": "true"
    }

    r = requests.get(url, headers=HEADERS, params=params)
    return r.json()

# -----------------------
# EXPECTED MOVE
# -----------------------
def expected_move(price, iv, dte):
    return price * iv * np.sqrt(dte/365)

# -----------------------
# MAIN ENGINE
# -----------------------
def process(symbol):

    url_exp = f"https://api.tradier.com/v1/markets/options/expirations"
    r = requests.get(url_exp, headers=HEADERS, params={"symbol":symbol})
    expirations = r.json()["expirations"]["date"]

    results = []

    for exp in expirations:

        dte = (datetime.strptime(exp,"%Y-%m-%d") - datetime.today()).days
        if dte <= 0 or dte > 7:
            continue

        chain_url = "https://api.tradier.com/v1/markets/options/chains"

        r = requests.get(chain_url, headers=HEADERS, params={
            "symbol": symbol,
            "expiration": exp,
            "greeks": "true"
        })

        options = r.json()["options"]["option"]

        for opt in options:

            strike = opt["strike"]
            iv = opt["greeks"]["mid_iv"] or 0
            gamma = opt["greeks"]["gamma"] or 0
            bid = opt["bid"]
            ask = opt["ask"]
            oi = opt["open_interest"]
            typ = opt["option_type"]

            if iv == 0 or bid == 0 or ask == 0:
                continue

            spread = ask - bid
            price = opt["underlying_price"]

            em = expected_move(price, iv, dte)

            if abs(strike - price) > em:
                continue

            # REAL gamma exposure
            gex = gamma * oi * 100

            # scoring
            score = (
                (1/spread) +
                np.log(oi+1) +
                abs(gex)/10000
            )

            proj = price + em if typ=="call" else price - em

            prob = np.exp(-0.5*((proj-price)/(price*iv))**2)

            results.append({
                "ticker": symbol,
                "type": typ,
                "strike": strike,
                "exp": exp,
                "price": price,
                "expected_move": em,
                "projected": proj,
                "probability": prob,
                "gex": gex,
                "score": score
            })

    return results


# RUN
all_data = []

for t in TICKERS:
    try:
        all_data += process(t)
    except:
        pass

df = pd.DataFrame(all_data).sort_values("score", ascending=False)

df.to_json("signals.json", orient="records", indent=2)

print(df.head(10))
