#!/usr/bin/env python3
"""
혼합 비율 테스트 — 동적 전략 + 정적 배분

동적 전략(60/20/20 히스테리시스)과 정적 배분(60/20/20 매수보유)을
0:100 ~ 100:0 으로 섞어가며 성과를 비교한다.

두 전략의 약점이 서로 다른 시기에 나타나므로(동적=횡보장, 정적=폭락장),
혼합하면 변동성이 각각보다 낮아질 수 있다.

새 파라미터를 추가하지 않으므로 과최적화 위험이 낮다.

사용:
  python mix.py            # 매월 리밸런싱
  python mix.py 0          # 리밸런싱 없음 (비중 드리프트 허용)
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
START = "2004-11-18"
CASH_RATE = 0.02
REBAL_MONTHLY = (sys.argv[1] != "0") if len(sys.argv) > 1 else True
# =======================================================================

TD = 252


def scalar_series(px: pd.Series) -> pd.Series:
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


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    ex = r - CASH_RATE / TD
    vol = r.std() * np.sqrt(TD)
    return {
        "CAGR": eq.iloc[-1] ** (1 / yrs) - 1,
        "Vol": vol,
        "Sharpe": ex.mean() / ex.std() * np.sqrt(TD) if ex.std() > 0 else 0.0,
        "MDD": dd.min(),
        "Calmar": (eq.iloc[-1] ** (1 / yrs) - 1) / -dd.min() if dd.min() < 0 else 0.0,
        "최악연도": ((1 + r).groupby(r.index.year).prod() - 1).min(),
    }


def blend(a: pd.Series, b: pd.Series, w: float, monthly: bool) -> pd.Series:
    """a 비중 w, b 비중 (1-w).

    monthly=True  : 매월 말 w 로 되돌림 (현실적)
    monthly=False : 최초 배분 후 방치 (드리프트 허용)
    """
    if not monthly:
        eq = w * (1 + a).cumprod() + (1 - w) * (1 + b).cumprod()
        return eq.pct_change().fillna(0.0)

    out = pd.Series(0.0, index=a.index)
    period = a.index.to_period("M")
    for _, idx in a.groupby(period).groups.items():
        seg_a, seg_b = a.loc[idx], b.loc[idx]
        eq = w * (1 + seg_a).cumprod() + (1 - w) * (1 + seg_b).cumprod()
        out.loc[idx] = eq.pct_change().fillna(eq.iloc[0] - 1)
    return out


def main():
    px = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                     progress=False, threads=False)["Close"]
    px = px[TICKERS].ffill().dropna()
    rets = px.pct_change().fillna(0.0)
    rf = CASH_RATE / TD

    # --- 동적 전략 ---
    dyn = pd.Series(rf, index=px.index)
    for t in TICKERS:
        s = scalar_series(px[t]).shift(1 + EXEC_LAG).fillna(0.0)
        w = BASE_WEIGHTS[t] * s
        dyn += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)

    # --- 정적 배분 (매월 리밸런싱) ---
    sw = pd.Series(BASE_WEIGHTS)
    sta = pd.Series(0.0, index=px.index)
    for _, idx in rets.groupby(rets.index.to_period("M")).groups.items():
        seg = rets.loc[idx]
        eq = (sw * (1 + seg).cumprod()).sum(axis=1)
        sta.loc[idx] = eq.pct_change().fillna(eq.iloc[0] - 1)

    mask = px.index >= pd.Timestamp(START)
    dyn, sta = dyn[mask], sta[mask]

    mode = "매월 리밸런싱" if REBAL_MONTHLY else "리밸런싱 없음"
    print(f"평가구간: {dyn.index[0].date()} ~ {dyn.index[-1].date()} "
          f"({len(dyn) / TD:.1f}년)   |   혼합 방식: {mode}\n")

    # --- 비율별 성과 ---
    print("■ 혼합 비율별 성과")
    print(f"  {'동적':>5}{'정적':>6} │{'CAGR':>8}{'Vol':>8}{'MDD':>8}"
          f"{'Sharpe':>9}{'Calmar':>8}{'최악연도':>9}")
    print("  " + "─" * 62)

    rows = []
    for i in range(0, 11):
        w = i / 10
        r = blend(dyn, sta, w, REBAL_MONTHLY)
        s = stats(r)
        s["w"] = w
        rows.append(s)
        star = ""
        print(f"  {w:>4.0%}{1 - w:>6.0%} │{s['CAGR']:>8.2%}{s['Vol']:>8.2%}"
              f"{s['MDD']:>8.2%}{s['Sharpe']:>9.3f}{s['Calmar']:>8.2f}"
              f"{s['최악연도']:>9.1%}{star}")

    df = pd.DataFrame(rows)
    best_sh = df.loc[df.Sharpe.idxmax()]
    best_ca = df.loc[df.Calmar.idxmax()]

    print("\n■ 최적 비율")
    print(f"  Sharpe 최대 : 동적 {best_sh.w:.0%} / 정적 {1 - best_sh.w:.0%}"
          f"  →  {best_sh.Sharpe:.3f}  (CAGR {best_sh.CAGR:.2%}, MDD {best_sh.MDD:.2%})")
    print(f"  Calmar 최대 : 동적 {best_ca.w:.0%} / 정적 {1 - best_ca.w:.0%}"
          f"  →  {best_ca.Calmar:.2f}  (CAGR {best_ca.CAGR:.2%}, MDD {best_ca.MDD:.2%})")

    pure_d = df[df.w == 1.0].iloc[0]
    pure_s = df[df.w == 0.0].iloc[0]
    print(f"\n  순수 동적 Sharpe {pure_d.Sharpe:.3f} / 순수 정적 {pure_s.Sharpe:.3f}")
    gain = best_sh.Sharpe - max(pure_d.Sharpe, pure_s.Sharpe)
    print(f"  혼합 이득: {gain:+.3f}"
          f"  →  {'유의미' if gain > 0.03 else '미미 (혼합 실익 적음)'}")

    print(f"\n  두 전략 상관계수: {dyn.corr(sta):.3f}"
          f"   (낮을수록 혼합 효과 큼)")

    # --- 취약 구간 검증 ---
    print("\n■ 구간별 비교 (동적 100% / 50:50 / 정적 100%)")
    periods = {
        "2007-2009 금융위기": ("2007-10-09", "2009-06-01"),
        "2015-2016 횡보장": ("2015-07-17", "2017-02-17"),
        "2018 4분기 급락": ("2018-09-20", "2019-04-30"),
        "2020 코로나": ("2020-02-19", "2020-08-31"),
        "2021-2023 금리인상": ("2021-11-19", "2023-07-18"),
        "2023-2025 AI 랠리": ("2023-01-01", "2025-12-31"),
    }
    mix = blend(dyn, sta, 0.5, REBAL_MONTHLY)
    print(f"  {'구간':<22}{'동적':>18}{'50:50':>18}{'정적':>18}")
    print("  " + "─" * 74)
    for name, (s0, e0) in periods.items():
        m = (dyn.index >= pd.Timestamp(s0)) & (dyn.index <= pd.Timestamp(e0))
        if m.sum() < 30:
            continue
        cells = []
        for series in (dyn, mix, sta):
            seg = series[m]
            tot = (1 + seg).prod() - 1
            eq = (1 + seg).cumprod()
            d = (eq / eq.cummax() - 1).min()
            cells.append(f"{tot:>8.1%} ({d:>6.1%})")
        print(f"  {name:<22}" + "".join(f"{c:>18}" for c in cells))

    print("\n  * 각 칸: 구간수익률 (구간MDD)")


if __name__ == "__main__":
    main()
