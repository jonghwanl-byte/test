import sys
import pandas as pd
import yfinance as yf

TICKER = sys.argv[1] if len(sys.argv) > 1 else "TLT"
BAND_UP, BAND_DN = 1.015, 0.975

px = yf.download(TICKER, period="max", auto_adjust=True,
                 progress=False, threads=False)["Close"]
if isinstance(px, pd.DataFrame):
    px = px.iloc[:, 0]
px = px.dropna()

print(f"=== {TICKER} | {px.index[0].date()} ~ {px.index[-1].date()} "
      f"({len(px)}행) ===\n")

for window in (20, 120, 200):
    ma = px.rolling(window).mean()
    st = pd.Series(float("nan"), index=px.index)
    st[px > ma * BAND_UP] = 1.0
    st[px < ma * BAND_DN] = 0.0
    st = st.ffill().fillna(0.0)

    df = pd.DataFrame({"price": px, "ma": ma,
                       "gap": px / ma - 1, "state": st}).dropna()

    flips = df[df.state.diff() != 0].tail(5)
    cur = "ON" if df.state.iloc[-1] == 1 else "OFF"

    print(f"[{window}일선]  현재 {cur}  이격도 {df.gap.iloc[-1]:+.2%}")
    print("  최근 전환 이력:")
    for d, r in flips.iterrows():
        print(f"    {d.date()}  {'→ON ' if r.state == 1 else '→OFF'}"
              f"  이격도 {r.gap:+.2%}  종가 {r.price:.2f}")

    tail = df.tail(25)
    print("  최근 25거래일:")
    for d, r in tail.iterrows():
        mark = "ON " if r.state == 1 else "OFF"
        hit = ""
        if r.gap > BAND_UP - 1:
            hit = "  <<< 상단 돌파"
        elif r.gap < BAND_DN - 1:
            hit = "  <<< 하단 이탈"
        print(f"    {d.date()}  {r.price:8.2f}  {r.gap:+7.2%}  {mark}{hit}")
    print()
