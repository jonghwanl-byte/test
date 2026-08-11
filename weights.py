#!/usr/bin/env python3
"""
자산 비중 시나리오 비교

확정 설정(±1.5%/-2.5%, MA 20/120/200, 100/75/50/0)을 고정하고
기본 비중만 바꿔가며 성과를 비교한다.

사용:
  python weights.py                    # 기본 시나리오 모음
  python weights.py 60,20,20 70,15,15  # 임의 조합 지정
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 확정 설정 =======================================================
TICKERS = ["QQQ", "TLT", "GLD"]
MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

EXEC_LAG = 1
COST = 0.0010
CASH_RATE = 0.02
START = "2004-11-18"
TD = 252

DEFAULT = [
    (40, 30, 30), (50, 25, 25), (60, 20, 20), (65, 17.5, 17.5),
    (70, 15, 15), (75, 12.5, 12.5), (80, 10, 10), (90, 5, 5), (100, 0, 0),
]

PERIODS = {
    "2007-2009 금융위기": ("2007-10-09", "2009-06-01"),
    "2015-2016 횡보장": ("2015-07-17", "2017-02-17"),
    "2018 4분기 급락": ("2018-09-20", "2019-04-30"),
    "2020 코로나": ("2020-02-19", "2020-08-31"),
    "2021-2023 금리인상": ("2021-11-19", "2023-07-18"),
    "2023-2025 AI 랠리": ("2023-01-01", "2025-12-31"),
}
# =======================================================================


def scalar_frame(close: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for t in TICKERS:
        px = close[t]
        s = pd.Series(0.0, index=px.index)
        for n in MA_PERIODS:
            ma = px.rolling(n, min_periods=n).mean()
            st = pd.Series(np.nan, index=px.index)
            st[px > ma * BAND_UP] = 1.0
            st[px < ma * BAND_DN] = 0.0
            st = st.ffill().fillna(0.0)
            st[ma.isna()] = 0.0
            s += st
        out[t] = s.map(lambda x: SCALAR_MAP.get(int(x), 0.0)).astype(float)
    return pd.DataFrame(out)


def run(S: pd.DataFrame, rets: pd.DataFrame, w: dict, rf: float) -> pd.Series:
    r = pd.Series(rf, index=S.index)
    for t in TICKERS:
        if w[t] <= 0:
            continue
        ww = (w[t] * S[t]).shift(1 + EXEC_LAG).fillna(0.0)
        r += ww * (rets[t] - rf) - COST * ww.diff().abs().fillna(0.0)
    return r


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    ex = r - CASH_RATE / TD
    ann = (1 + r).groupby(r.index.year).prod() - 1
    down = np.where(ex < 0, ex, 0.0)
    dstd = np.sqrt((down ** 2).sum() / (len(r) - 1)) * np.sqrt(TD)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {
        "CAGR": cagr,
        "Vol": r.std() * np.sqrt(TD),
        "MDD": dd.min(),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(TD),
        "Sortino": ex.mean() * TD / dstd if dstd > 0 else 0.0,
        "Calmar": cagr / -dd.min(),
        "최악연도": ann.min(),
        "최고연도": ann.max(),
        "손실연수": int((ann < 0).sum()),
        "회복일": int(max((dd < -1e-9).astype(int)
                          .groupby((dd >= -1e-9).cumsum()).sum())),
    }


def main():
    if len(sys.argv) > 1:
        scen = []
        for a in sys.argv[1:]:
            p = [float(x) for x in a.split(",")]
            scen.append(tuple(p))
    else:
        scen = DEFAULT

    close = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                        progress=False, threads=False)["Close"]
    close = close[TICKERS].ffill().dropna()
    rets = close.pct_change().fillna(0.0)
    rf = CASH_RATE / TD
    S = scalar_frame(close)
    m = np.asarray(close.index >= pd.Timestamp(START))

    series, rows = {}, []
    for q, t, g in scen:
        w = {"QQQ": q / 100, "TLT": t / 100, "GLD": g / 100}
        r = run(S, rets, w, rf)[m]
        series[f"{q:.0f}/{t:.0f}/{g:.0f}"] = r
        rows.append({"비중": f"{q:.0f}/{t:.0f}/{g:.0f}", **stats(r)})

    df = pd.DataFrame(rows)
    idx = list(series.values())[0].index
    print(f"평가구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)/TD:.1f}년)")
    print(f"설정: 밴드 +1.5%/-2.5%  MA 20/120/200  스케일 100/75/50/0\n")

    print("■ 비중별 성과")
    print(f"  {'비중':<12}{'CAGR':>8}{'Vol':>8}{'MDD':>9}{'Sharpe':>9}"
          f"{'Sortino':>9}{'Calmar':>8}{'최악연도':>9}{'손실연수':>9}")
    print("  " + "─" * 81)
    for _, r in df.iterrows():
        print(f"  {r['비중']:<12}{r['CAGR']:>8.2%}{r['Vol']:>8.2%}"
              f"{r['MDD']:>9.2%}{r['Sharpe']:>9.3f}{r['Sortino']:>9.3f}"
              f"{r['Calmar']:>8.2f}{r['최악연도']:>9.1%}{r['손실연수']:>8}년")

    print("\n■ 최적 비중")
    for k in ("Sharpe", "Sortino", "Calmar", "CAGR"):
        b = df.loc[df[k].idxmax()]
        print(f"  {k:<8} 최대 : {b['비중']:<12}"
              f"(CAGR {b['CAGR']:.2%}, MDD {b['MDD']:.2%}, Sharpe {b['Sharpe']:.3f})")

    # ---- 60 vs 70 직접 비교 ----
    a, b = "60/20/20", "70/15/15"
    if a in series and b in series:
        ra, rb = df[df.비중 == a].iloc[0], df[df.비중 == b].iloc[0]
        print(f"\n■ {a} vs {b} 직접 비교\n")
        print(f"  {'지표':<12}{a:>12}{b:>12}{'차이':>12}")
        print("  " + "─" * 48)
        for k, f in [("CAGR", "{:.2%}"), ("Vol", "{:.2%}"), ("MDD", "{:.2%}"),
                     ("Sharpe", "{:.3f}"), ("Sortino", "{:.3f}"),
                     ("Calmar", "{:.2f}"), ("최악연도", "{:.1%}")]:
            d = rb[k] - ra[k]
            ds = f.format(d) if k != "Calmar" else f"{d:+.2f}"
            print(f"  {k:<12}{f.format(ra[k]):>12}{f.format(rb[k]):>12}{ds:>12}")

        # 21년 복리 차이
        yrs = len(idx) / TD
        mult = (1 + rb["CAGR"]) ** yrs / (1 + ra["CAGR"]) ** yrs - 1
        print(f"\n  {yrs:.0f}년 누적 자산 차이: {mult:+.1%}")

    # ---- 구간별 ----
    print("\n■ 구간별 수익률 (괄호는 구간 MDD)\n")
    keys = [k for k in ("50/25/25", "60/20/20", "70/15/15", "80/10/10")
            if k in series]
    print(f"  {'구간':<20}" + "".join(f"{k:>19}" for k in keys))
    print("  " + "─" * (20 + 19 * len(keys)))
    for name, (s0, e0) in PERIODS.items():
        mm = np.asarray((idx >= pd.Timestamp(s0)) & (idx <= pd.Timestamp(e0)))
        if mm.sum() < 30:
            continue
        cells = []
        for k in keys:
            seg = series[k][mm]
            tot = (1 + seg).prod() - 1
            eq = (1 + seg).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            cells.append(f"{tot:>9.1%} ({dd:>6.1%})")
        print(f"  {name:<20}" + "".join(f"{c:>19}" for c in cells))

    # ---- 연도별 ----
    print("\n■ 연도별 수익률\n")
    print(f"  {'연도':<7}" + "".join(f"{k:>12}" for k in keys))
    print("  " + "─" * (7 + 12 * len(keys)))
    years = sorted(set(idx.year))
    for y in years:
        cells = []
        for k in keys:
            r = series[k]
            cells.append((1 + r[r.index.year == y]).prod() - 1)
        print(f"  {y:<7}" + "".join(f"{c:>12.1%}" for c in cells))


if __name__ == "__main__":
    main()
