#!/usr/bin/env python3
"""
0/3 상태 잔여 비중(floor) 검증

현재 룰은 세 이동평균이 모두 OFF(0/3)일 때 비중을 0으로 만든다.
그 결과 반등 초입에 완전히 못 타는 문제가 있다.
0 대신 소량을 남겨두면 개선되는지 확인한다.

  현재      100 / 75 / 50 / 0
  후보      100 / 75 / 50 / f      (f = 10~30%)

판정 기준 (사전 확정)
--------------------
  ① 샤프가 현재 대비 하락하지 않을 것 (-0.005 이내 허용)
  ② 2015-2016 횡보장 수익이 개선될 것
  ③ 2008 금융위기 / 2022 금리인상 구간이 악화되지 않을 것 (-0.5%p 이내)
  세 조건을 모두 만족해야 채택.

사용:
  python floor.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 확정 설정 =======================================================
TICKERS = ["QQQ", "TLT", "GLD"]
BASE_WEIGHTS = {"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20}
MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975

EXEC_LAG = 1
COST = 0.0010
CASH_RATE = 0.02
START = "2004-11-18"
TD = 252

FLOORS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]

CRISIS = {"2007-2009 금융위기": ("2007-10-09", "2009-06-01"),
          "2021-2023 금리인상": ("2021-11-19", "2023-07-18")}
TARGET = {"2015-2016 횡보장": ("2015-07-17", "2017-02-17")}
OTHER = {"2018 4분기 급락": ("2018-09-20", "2019-04-30"),
         "2020 코로나": ("2020-02-19", "2020-08-31"),
         "2023-2025 AI 랠리": ("2023-01-01", "2025-12-31")}
ALL_P = {**CRISIS, **TARGET, **OTHER}
# =======================================================================


def score_frame(close):
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
        out[t] = s
    return pd.DataFrame(out)


def run(SC, rets, floor, rf):
    lut = np.array([floor, 0.50, 0.75, 1.00])
    r = pd.Series(rf, index=SC.index)
    W = {}
    for t in TICKERS:
        s = pd.Series(lut[SC[t].to_numpy().astype(np.int64)], index=SC.index)
        w = (BASE_WEIGHTS[t] * s).shift(1 + EXEC_LAG).fillna(0.0)
        W[t] = w
        r += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)
    return r, pd.DataFrame(W)


def stats(r):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    ex = r - CASH_RATE / TD
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1,
            "Vol": r.std() * np.sqrt(TD),
            "MDD": dd.min(),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(TD),
            "최악연도": ((1 + r).groupby(r.index.year).prod() - 1).min()}


def seg_ret(r, s0, e0):
    m = np.asarray((r.index >= pd.Timestamp(s0)) & (r.index <= pd.Timestamp(e0)))
    if m.sum() < 30:
        return None, None
    seg = r[m]
    eq = (1 + seg).cumprod()
    return float((1 + seg).prod() - 1), float((eq / eq.cummax() - 1).min())


def main():
    close = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                        progress=False, threads=False)["Close"]
    close = close[TICKERS].ffill().dropna()
    rets = close.pct_change().fillna(0.0)
    rf = CASH_RATE / TD
    SC = score_frame(close)
    m = np.asarray(close.index >= pd.Timestamp(START))

    res, series, wts = {}, {}, {}
    for f in FLOORS:
        r, W = run(SC, rets, f, rf)
        series[f] = r[m]
        wts[f] = W[m]
        res[f] = stats(r[m])

    b = res[0.00]
    idx = series[0.00].index
    print(f"평가구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)/TD:.1f}년)")
    print(f"기준: 100/75/50/0   |   비중 60/20/20   밴드 +1.5%/-2.5%\n")

    print("■ 잔여 비중별 성과\n")
    print(f"  {'룰':<20}{'CAGR':>8}{'Vol':>8}{'MDD':>9}{'Sharpe':>9}"
          f"{'ΔSharpe':>9}{'최악연도':>9}{'0%일수':>8}")
    print("  " + "─" * 72)
    for f in FLOORS:
        s = res[f]
        z = (wts[f][TICKERS].sum(axis=1) < 0.01).mean()
        label = f"100/75/50/{f*100:.0f}"
        print(f"  {label:<20}{s['CAGR']:>8.2%}{s['Vol']:>8.2%}{s['MDD']:>9.2%}"
              f"{s['Sharpe']:>9.3f}{s['Sharpe']-b['Sharpe']:>+9.3f}"
              f"{s['최악연도']:>9.1%}{z:>8.1%}")

    # ---------- 구간별 ----------
    print("\n" + "=" * 76)
    print("■ 구간별 수익률\n")
    print(f"  {'구간':<22}" + "".join(f"{f'{f*100:.0f}%':>10}" for f in FLOORS))
    print("  " + "─" * (22 + 10 * len(FLOORS)))
    seg = {}
    for name, (s0, e0) in ALL_P.items():
        row = []
        for f in FLOORS:
            v, _ = seg_ret(series[f], s0, e0)
            row.append(v)
            seg[(name, f)] = v
        tag = " ★" if name in TARGET else (" ⚠" if name in CRISIS else "")
        print(f"  {name + tag:<22}" + "".join(
            f"{v:>10.1%}" if v is not None else f"{'-':>10}" for v in row))

    print("\n  ★ 개선 목표 구간   ⚠ 훼손되면 안 되는 구간")

    # ---------- 판정 ----------
    print("\n" + "=" * 76)
    print("■ 판정 (① 샤프 유지  ② 2015-16 개선  ③ 위기구간 유지)\n")
    print(f"  {'룰':<20}{'①샤프':>10}{'②2015-16':>12}{'③위기':>10}{'종합':>8}")
    print("  " + "─" * 60)

    passed = []
    for f in FLOORS:
        if f == 0:
            continue
        c1 = res[f]["Sharpe"] - b["Sharpe"] >= -0.005
        t_name = list(TARGET)[0]
        d2 = seg[(t_name, f)] - seg[(t_name, 0.0)]
        c2 = d2 > 0
        d3 = min(seg[(n, f)] - seg[(n, 0.0)] for n in CRISIS)
        c3 = d3 >= -0.005
        ok = c1 and c2 and c3
        if ok:
            passed.append(f)
        print(f"  {f'100/75/50/{f*100:.0f}':<20}"
              f"{('통과' if c1 else '실패'):>10}"
              f"{f'{d2:+.1%} ' + ('통과' if c2 else '실패'):>12}"
              f"{f'{d3:+.1%}':>10}"
              f"{('채택' if ok else '기각'):>8}")

    print("\n" + "=" * 76)
    if not passed:
        print("■ 결론: 채택 가능한 잔여 비중 없음 → 현재 룰(0%) 유지\n")
        best = max((f for f in FLOORS if f > 0),
                   key=lambda x: res[x]["Sharpe"])
        print(f"  참고) 샤프 최대 후보: {best*100:.0f}%  "
              f"({res[best]['Sharpe']:.3f} vs 기준 {b['Sharpe']:.3f})")
        t_name = list(TARGET)[0]
        print(f"        해당 후보의 2015-16: "
              f"{seg[(t_name, best)]:+.1%} (기준 {seg[(t_name, 0.0)]:+.1%})")
    else:
        best = max(passed, key=lambda x: res[x]["Sharpe"])
        print(f"■ 결론: {len(passed)}개 통과 → 최적 잔여 비중 {best*100:.0f}%\n")
        print(f"  CAGR {res[best]['CAGR']:.2%} (기준 {b['CAGR']:.2%})   "
              f"MDD {res[best]['MDD']:.2%} (기준 {b['MDD']:.2%})   "
              f"Sharpe {res[best]['Sharpe']:.3f} (기준 {b['Sharpe']:.3f})")

    # ---------- 노출 ----------
    print("\n" + "=" * 76)
    print("■ 노출 및 회전율\n")
    print(f"  {'룰':<20}{'평균주식':>10}{'평균현금':>10}{'연회전율':>10}")
    print("  " + "─" * 50)
    for f in FLOORS:
        W = wts[f]
        tot = W[TICKERS].sum(axis=1)
        turn = W[TICKERS].diff().abs().sum(axis=1).sum() / (len(W) / TD)
        print(f"  {f'100/75/50/{f*100:.0f}':<20}{tot.mean():>10.1%}"
              f"{1-tot.mean():>10.1%}{turn:>10.2f}")


if __name__ == "__main__":
    main()
