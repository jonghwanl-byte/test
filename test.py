import yfinance as yf, pandas as pd

px = yf.download("TLT", period="max", auto_adjust=True, progress=False)["Close"]
if hasattr(px, 'columns'): px = px.iloc[:, 0]

ma = px.rolling(20).mean()
st = pd.Series(float('nan'), index=px.index)
st[px > ma * 1.015] = 1.0
st[px < ma * 0.975] = 0.0
st = st.ffill().fillna(0.0)

df = pd.DataFrame({
    "price": px, "ma20": ma,
    "이격도": px/ma - 1, "state": st,
}).dropna().tail(40)
df["전환"] = df.state.diff().map({1.0: "→ON", -1.0: "→OFF"}).fillna("")
print(df.to_string(formatters={"이격도": "{:+.2%}".format}))
