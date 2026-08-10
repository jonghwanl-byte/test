#!/usr/bin/env python3
"""
ADX 판별력 진단 (1단계)

전략을 바꾸기 전에, ADX가 정말 국면을 구분하는지부터 확인한다.

핵심 질문
---------
Q1. 횡보장(2015-16)에서 ADX가 낮게 나오는가?
Q2. 완만한 약세장(2008, 2022)에서는 높게 나오는가?
    -> 두 국면을 구분 못 하면 이 아이디어는 폐기.
Q3. ADX가 낮을 때 실제로 동적 전략이 정적보다 못했는가?
    -> 이게 성립해야 스위치로 쓸 수 있다.
Q4. ADX는 후행 지표다. 손실이 난 시점에 이미 신호가 떴는가?

사용:
  python adx_check.py           # ADX 14일
  python adx_check.py 20        # ADX 20일
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

# ===== 설정 ============================================================
TICKERS = ["QQQ", "TLT", "GLD"]
BASE_WEIGHTS = {"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20}
MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

EXEC_LAG = 1
COST = 0.0010
CASH_RATE = 0.02
START = "2004-11-18"

ADX_N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
THRESHOLDS = [15, 20, 25, 30]

# 국면 정의 (앞선 검증에서 사용한 것과 동일)
REGIMES = {
    "급락장 (V자 회복)": [("2020-02-20", "2020-08-31"),
                          ("2018-09-20", "2019-04-30")],
    "완만한 약세장": [("2007-10-09", "2009-03-09"),
                      ("2022-01-01", "2022-12-31")],
    "횡보장": [("2011-04-30", "2012-06-01"),
               ("2015-07-17", "2017-02-17")],
    "추세 상승장": [("2013-01-01", "2014-12-31"),
                    ("2023-01-01", "2024-12-31")],
}
# =======================================================================


def adx(high, low, close, n=14):
    """Wilder ADX. 미래 정보 없음(과거 n일만 사용)."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)

    # Wilder 평활 (alpha = 1/n)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False).mean() / atr

    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def scalar_series(px):
    score = pd.Series(0.0, index=px.index)
    for n in MA_PERIODS:
        ma = px.rolling(n, min_periods=n).mean()
        st = pd.Series(np.nan, index=px.index)
        st[px > ma * BAND_UP] = 1.0
        st[px < ma * BAND_DN] = 0.0
        st = st.ffill().fillna(0.0)
        st[ma.isna()] = 0.0
        score += st
    return score.map(lambda s: SCALAR_MAP.get(int(s), 0.0)).astype(float)


def mask_of(index, spans):
    m = np.zeros(len(index), dtype=bool)
    for s, e in spans:
        m |= np.asarray((index >= pd.Timestamp(s)) & (index <= pd.Timestamp(e)))
    return m


def main():
    raw = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                      progress=False, threads=False)
    close = raw["Close"][TICKERS].ffill().dropna()
    high = raw["High"][TICKERS].reindex(close.index).ffill()
    low = raw["Low"][TICKERS].reindex(close.index).ffill()

    a = adx(high["QQQ"], low["QQQ"], close["QQQ"], ADX_N)

    rets = close.pct_change().fillna(0.0)
    rf = CASH_RATE / 252

    # 동적 전략
    dyn = pd.Series(rf, index=close.index)
    for t in TICKERS:
        w = (BASE_WEIGHTS[t] * scalar_series(close[t])).shift(1 + EXEC_LAG).fillna(0.0)
        dyn += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)

    # 정적 배분 (매월 리밸런싱)
    sw = pd.Series(BASE_WEIGHTS)
    sta = pd.Series(0.0, index=close.index)
    for _, idx in rets.groupby(rets.index.to_period("M")).groups.items():
        seg = rets.loc[idx]
        eq = (sw * (1 + seg).cumprod()).sum(axis=1)
        sta.loc[idx] = eq.pct_change().fillna(eq.iloc[0] - 1)

    m0 = np.asarray(close.index >= pd.Timestamp(START))
    a, dyn, sta = a[m0], dyn[m0], sta[m0]
    idx = dyn.index

    print(f"ADX({ADX_N}) 진단  |  {idx[0].date()} ~ {idx[-1].date()} "
          f"({len(idx)/252:.1f}년)")
    print(f"전체 ADX 분포: 중앙값 {a.median():.1f}  "
          f"25%={a.quantile(.25):.1f}  75%={a.quantile(.75):.1f}\n")

    # ---------- Q1 & Q2 : 국면별 ADX ----------
    print("=" * 72)
    print("■ Q1/Q2. ADX가 국면을 구분하는가\n")
    print(f"  {'국면':<20}{'ADX중앙값':>10}{'평균':>8}"
          f"{'<20비율':>10}{'<25비율':>10}{'기간':>8}")
    print("  " + "─" * 66)
    rows = {}
    for name, spans in REGIMES.items():
        m = mask_of(idx, spans)
        av = a[m].dropna()
        if len(av) < 30:
            continue
        rows[name] = av
        print(f"  {name:<20}{av.median():>10.1f}{av.mean():>8.1f}"
              f"{(av < 20).mean():>10.1%}{(av < 25).mean():>10.1%}"
              f"{len(av)/252:>7.1f}년")

    if "횡보장" in rows and "완만한 약세장" in rows:
        s_med = rows["횡보장"].median()
        b_med = rows["완만한 약세장"].median()
        gap = b_med - s_med
        print(f"\n  판정: 횡보장 {s_med:.1f}  vs  완만한 약세장 {b_med:.1f}"
              f"   격차 {gap:+.1f}")
        if gap > 4:
            print("        → 구분 가능. 다음 단계 진행 가치 있음.")
        elif gap > 1:
            print("        → 구분 미약. 오작동 위험 있음.")
        else:
            print("        → 구분 불가. 이 아이디어는 폐기 권장.")
            print("           (ADX가 낮을 때 정적으로 바꾸면 약세장에서 낙폭이 커진다)")

    # ---------- Q3 : ADX 구간별 동적 vs 정적 ----------
    print("\n" + "=" * 72)
    print("■ Q3. ADX가 낮을 때 실제로 동적이 정적보다 못했는가\n")
    print(f"  {'ADX 구간':<14}{'거래일':>8}{'동적CAGR':>11}{'정적CAGR':>11}"
          f"{'차이':>9}{'동적MDD':>10}{'정적MDD':>10}")
    print("  " + "─" * 73)

    bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
    for lo, hi in bins:
        m = np.asarray((a >= lo) & (a < hi))
        if m.sum() < 60:
            continue
        d, s = dyn[m], sta[m]
        yrs = m.sum() / 252
        cd = (1 + d).prod() ** (1 / yrs) - 1
        cs = (1 + s).prod() ** (1 / yrs) - 1
        ed, es = (1 + d).cumprod(), (1 + s).cumprod()
        dd = (ed / ed.cummax() - 1).min()
        ds = (es / es.cummax() - 1).min()
        flag = "  ← 정적 우세" if cs > cd else ""
        print(f"  {lo:>3}~{hi:<10}{m.sum():>8}{cd:>11.2%}{cs:>11.2%}"
              f"{cd - cs:>9.2%}{dd:>10.1%}{ds:>10.1%}{flag}")

    print("\n  * ADX 구간별 수익률은 연속 구간이 아니므로 참고용")

    # ---------- Q4 : 후행성 ----------
    print("\n" + "=" * 72)
    print("■ Q4. ADX는 후행 지표다 — 손실 시점에 신호가 떴는가\n")
    print("  2015-2016 횡보장 월별 ADX 및 성과")
    print(f"  {'월':<10}{'ADX평균':>9}{'<20':>7}{'동적':>9}{'정적':>9}{'차이':>9}")
    print("  " + "─" * 53)
    m = mask_of(idx, [("2015-07-17", "2017-02-17")])
    sub_a, sub_d, sub_s = a[m], dyn[m], sta[m]
    per = sub_d.index.to_period("M")
    for p in sorted(set(per)):
        mm = np.asarray(per == p)
        av = sub_a[mm].mean()
        rd = (1 + sub_d[mm]).prod() - 1
        rs = (1 + sub_s[mm]).prod() - 1
        tag = "○" if av < 20 else ""
        print(f"  {str(p):<10}{av:>9.1f}{tag:>7}{rd:>9.1%}{rs:>9.1%}"
              f"{rd - rs:>9.1%}")

    print("\n  ○ = ADX 20 미만 (횡보 신호)")
    print("  차이가 크게 마이너스인 달에 ○ 가 붙어 있어야 필터가 작동한다.")

    # ---------- 요약 ----------
    print("\n" + "=" * 72)
    print("■ 임계치별 커버리지\n")
    print(f"  {'임계치':<10}{'전체비중':>10}{'횡보장포착':>12}"
          f"{'약세장오탐':>12}{'상승장오탐':>12}")
    print("  " + "─" * 56)
    for th in THRESHOLDS:
        row = [f"  {'ADX<' + str(th):<10}{(a < th).mean():>10.1%}"]
        for key in ("횡보장", "완만한 약세장", "추세 상승장"):
            if key in rows:
                row.append(f"{(rows[key] < th).mean():>12.1%}")
            else:
                row.append(f"{'-':>12}")
        print("".join(row))

    print("\n  '횡보장포착'은 높고 '약세장오탐'은 낮아야 유용하다.")
    print("  둘이 비슷하면 필터가 국면을 구분하지 못한다는 뜻이다.")


if __name__ == "__main__":
    main()
