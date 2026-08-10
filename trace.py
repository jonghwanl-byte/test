#!/usr/bin/env python3
"""
특정 구간 비중 변화 추적

전략이 언제 무엇을 사고 팔았는지, 그 결과 자산곡선이 어떻게 움직였는지
날짜별로 재현한다. 2015~2016 횡보장 같은 취약 구간 분석용.

사용:
  python trace.py                          # 2015-07-17 ~ 2017-02-17
  python trace.py 2021-11-19 2023-07-18    # 임의 구간
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 확정 설정 =======================================================
TICKERS = ["QQQ", "TLT", "GLD"]
BASE_WEIGHTS = {"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20}
MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

EXEC_LAG = 1
COST = 0.0010
CASH_RATE = 0.02

S0 = sys.argv[1] if len(sys.argv) > 2 else "2015-07-17"
E0 = sys.argv[2] if len(sys.argv) > 2 else "2017-02-17"
# =======================================================================


def score_series(px: pd.Series) -> pd.DataFrame:
    """MA별 ON/OFF + 합계 스코어."""
    out = {}
    for n in MA_PERIODS:
        ma = px.rolling(n, min_periods=n).mean()
        st = pd.Series(np.nan, index=px.index)
        st[px > ma * BAND_UP] = 1.0
        st[px < ma * BAND_DN] = 0.0
        st = st.ffill().fillna(0.0)
        st[ma.isna()] = 0.0
        out[n] = st
    df = pd.DataFrame(out)
    df["score"] = df.sum(axis=1)
    return df


def main():
    px = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                     progress=False, threads=False)["Close"]
    px = px[TICKERS].ffill().dropna()
    rets = px.pct_change().fillna(0.0)
    rf = CASH_RATE / 252

    scores, weights = {}, {}
    port = pd.Series(rf, index=px.index)
    for t in TICKERS:
        sc = score_series(px[t])
        scores[t] = sc["score"]
        s = sc["score"].map(lambda x: SCALAR_MAP.get(int(x), 0.0)).astype(float)
        w = (BASE_WEIGHTS[t] * s).shift(1 + EXEC_LAG).fillna(0.0)
        weights[t] = w
        port += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)

    W = pd.DataFrame(weights)
    SC = pd.DataFrame(scores)
    W["현금"] = 1.0 - W[TICKERS].sum(axis=1)

    s, e = pd.Timestamp(S0), pd.Timestamp(E0)
    m = (px.index >= s) & (px.index <= e)
    seg_w, seg_p, seg_sc = W[m], port[m], SC[m]

    eq = (1 + seg_p).cumprod()
    dd = eq / eq.cummax() - 1

    qqq_eq = (1 + rets["QQQ"][m]).cumprod()
    qqq_dd = qqq_eq / qqq_eq.cummax() - 1

    print(f"■ 구간: {seg_p.index[0].date()} ~ {seg_p.index[-1].date()}"
          f"  ({len(seg_p)}거래일, {len(seg_p)/252:.1f}년)\n")
    print(f"  전략      수익 {eq.iloc[-1]-1:>+7.2%}   MDD {dd.min():>7.2%}")
    print(f"  QQQ       수익 {qqq_eq.iloc[-1]-1:>+7.2%}   MDD {qqq_dd.min():>7.2%}")
    print(f"  차이      {(eq.iloc[-1]-1)-(qqq_eq.iloc[-1]-1):>+12.2%}\n")

    # ---- 비중 변경 이벤트 ----
    chg = seg_w[TICKERS].diff().abs().sum(axis=1) > 1e-9
    events = seg_w[chg]
    print(f"■ 비중 변경 이력 ({len(events)}회)\n")
    print(f"  {'날짜':<12}{'QQQ':>7}{'TLT':>7}{'GLD':>7}{'현금':>8}"
          f"{'  스코어':>12}{'  누적':>9}{'낙폭':>8}")
    print("  " + "─" * 74)

    prev = None
    for d, row in events.iterrows():
        sc = seg_sc.loc[d]
        mark = ""
        if prev is not None:
            dq = row["QQQ"] - prev["QQQ"]
            if abs(dq) > 1e-9:
                mark = " ←QQQ" + ("↑" if dq > 0 else "↓")
        print(f"  {d.date()}  {row['QQQ']:>6.0%}{row['TLT']:>7.0%}"
              f"{row['GLD']:>7.0%}{row['현금']:>8.0%}"
              f"   {int(sc['QQQ'])}/{int(sc['TLT'])}/{int(sc['GLD'])}"
              f"    {eq.loc[d]-1:>+7.1%}{dd.loc[d]:>8.1%}{mark}")
        prev = row

    # ---- 월별 요약 ----
    print(f"\n■ 월별 평균 비중 및 성과\n")
    print(f"  {'월':<10}{'QQQ':>7}{'TLT':>7}{'GLD':>7}{'현금':>8}"
          f"{'  전략':>9}{'  QQQ':>9}")
    print("  " + "─" * 60)
    per = seg_p.index.to_period("M")
    for p in sorted(set(per)):
        mm = per == p
        wm = seg_w[mm].mean()
        r_s = (1 + seg_p[mm]).prod() - 1
        r_q = (1 + rets["QQQ"][m][mm]).prod() - 1
        flag = "  ←" if r_s < 0 and r_q > 0 else ""
        print(f"  {str(p):<10}{wm['QQQ']:>7.0%}{wm['TLT']:>7.0%}"
              f"{wm['GLD']:>7.0%}{wm['현금']:>8.0%}"
              f"{r_s:>9.1%}{r_q:>9.1%}{flag}")

    print("\n  ← 표시: 시장은 올랐는데 전략은 손실난 달")

    # ---- 노출 통계 ----
    print(f"\n■ 구간 요약")
    print(f"  평균 주식(QQQ) 노출   {seg_w['QQQ'].mean():.1%}"
          f"   (최대 {seg_w['QQQ'].max():.0%} / 최소 {seg_w['QQQ'].min():.0%})")
    print(f"  평균 현금 비중        {seg_w['현금'].mean():.1%}")
    print(f"  QQQ 0% 였던 기간      {(seg_w['QQQ'] < 1e-9).mean():.1%}"
          f"  ({int((seg_w['QQQ'] < 1e-9).sum())}거래일)")
    print(f"  현금 50% 이상 기간    {(seg_w['현금'] >= 0.5).mean():.1%}")
    print(f"  비중 변경 횟수        {len(events)}회"
          f"  (연 {len(events)/(len(seg_p)/252):.0f}회)")

    lo = seg_w["QQQ"].idxmin()
    print(f"\n  QQQ 최저 시점: {lo.date()}  "
          f"QQQ가격 ${px.loc[lo,'QQQ']:.2f}")
    fwd = px["QQQ"][px.index > lo].head(126)
    if len(fwd):
        print(f"    이후 6개월 QQQ 수익률: {fwd.iloc[-1]/px.loc[lo,'QQQ']-1:+.1%}"
              f"   ← 이 구간을 놓쳤는지 확인")


if __name__ == "__main__":
    main()
