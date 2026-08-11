#!/usr/bin/env python3
"""
자산 상관관계 분석

지정한 티커들의 상관구조를 분석하고, 현재 조합(QQQ/TLT/GLD)과 비교한다.

핵심 질문
---------
Q1. 평시 상관은?
Q2. 위기 때 상관이 어떻게 변하나? (분산이 가장 필요할 때 작동하는가)
Q3. 롤링 상관이 안정적인가?
Q4. 분산 효과를 수치로 얼마나 얻는가?

사용:
  python corr.py                    # QQQ XLE XLF  vs  QQQ TLT GLD
  python corr.py QQQ XLE XLF XLV    # 임의 티커
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf

BASE = ["QQQ", "TLT", "GLD"]
TARGET = sys.argv[1:] if len(sys.argv) > 1 else ["QQQ", "XLE", "XLF"]
START = "2004-11-18"
TD = 252

CRISIS = {
    "2008 금융위기": ("2007-10-09", "2009-03-09"),
    "2011 유럽위기": ("2011-04-30", "2011-10-03"),
    "2015-16 차이나쇼크": ("2015-07-17", "2016-02-11"),
    "2018 4분기 급락": ("2018-09-20", "2018-12-24"),
    "2020 코로나": ("2020-02-19", "2020-03-23"),
    "2022 금리인상": ("2022-01-01", "2022-10-12"),
}
CALM = {
    "2013-2014 저변동": ("2013-01-01", "2014-12-31"),
    "2017 저변동": ("2017-01-01", "2017-12-31"),
    "2023-2024 AI랠리": ("2023-01-01", "2024-12-31"),
}


def load(tickers):
    px = yf.download(sorted(set(tickers)), start="1999-01-01", auto_adjust=True,
                     progress=False, threads=False)["Close"]
    return px.ffill().dropna()


def corr_table(r: pd.DataFrame, title: str):
    c = r.corr()
    print(f"\n  {title}")
    cols = list(c.columns)
    print("       " + "".join(f"{x:>8}" for x in cols))
    for i in cols:
        print(f"  {i:<5}" + "".join(
            f"{c.loc[i, j]:>8.2f}" if i != j else f"{'—':>8}" for j in cols))
    off = [c.loc[a, b] for a, b in combinations(cols, 2)]
    print(f"  평균 상관: {np.mean(off):.3f}   최대 {max(off):.2f}   최소 {min(off):.2f}")
    return np.mean(off)


def div_ratio(r: pd.DataFrame, w=None) -> float:
    """분산비율 = (가중평균 개별변동성) / (포트폴리오 변동성). 클수록 분산 효과 큼."""
    n = r.shape[1]
    w = np.full(n, 1 / n) if w is None else np.asarray(w)
    vol = r.std().to_numpy() * np.sqrt(TD)
    pv = (r @ w).std() * np.sqrt(TD)
    return float((w @ vol) / pv)


def main():
    all_t = sorted(set(TARGET) | set(BASE))
    px = load(all_t)
    px = px[px.index >= pd.Timestamp(START)]
    r = px.pct_change().dropna()

    print(f"평가구간: {r.index[0].date()} ~ {r.index[-1].date()} "
          f"({len(r)/TD:.1f}년)")
    print(f"대상: {' / '.join(TARGET)}    비교군: {' / '.join(BASE)}")

    # ---------- Q1 ----------
    print("\n" + "=" * 70)
    print("■ Q1. 전체 기간 상관")
    m_t = corr_table(r[TARGET], f"[대상] {' / '.join(TARGET)}")
    m_b = corr_table(r[BASE], f"[비교] {' / '.join(BASE)}")
    print(f"\n  평균 상관 차이: {m_t:.3f} vs {m_b:.3f}  ({m_t - m_b:+.3f})")
    if m_t > m_b + 0.15:
        print("  → 대상 조합의 상관이 뚜렷하게 높다. 분산 효과 열위.")

    # ---------- 개별 성과 ----------
    print("\n" + "=" * 70)
    print("■ 개별 자산 성과 (매수보유)\n")
    print(f"  {'티커':<7}{'CAGR':>9}{'Vol':>9}{'MDD':>10}{'Sharpe':>9}")
    print("  " + "─" * 44)
    for t in all_t:
        s = r[t]
        eq = (1 + s).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        cagr = eq.iloc[-1] ** (TD / len(s)) - 1
        vol = s.std() * np.sqrt(TD)
        mark = " *" if t in TARGET else ""
        print(f"  {t:<7}{cagr:>9.2%}{vol:>9.2%}{dd:>10.2%}"
              f"{(cagr - 0.02) / vol:>9.3f}{mark}")

    # ---------- Q2 ----------
    print("\n" + "=" * 70)
    print("■ Q2. 위기 구간 상관 (분산이 가장 필요한 때)\n")
    print(f"  {'구간':<22}{'대상':>10}{'비교':>10}{'대상수익':>11}{'비교수익':>11}")
    print("  " + "─" * 64)
    for name, (s0, e0) in {**CRISIS}.items():
        m = (r.index >= pd.Timestamp(s0)) & (r.index <= pd.Timestamp(e0))
        if m.sum() < 20:
            continue
        seg = r[m]
        ct = np.mean([seg[TARGET].corr().loc[a, b]
                      for a, b in combinations(TARGET, 2)])
        cb = np.mean([seg[BASE].corr().loc[a, b]
                      for a, b in combinations(BASE, 2)])
        rt = (1 + seg[TARGET].mean(axis=1)).prod() - 1
        rb = (1 + seg[BASE].mean(axis=1)).prod() - 1
        print(f"  {name:<22}{ct:>10.2f}{cb:>10.2f}{rt:>11.1%}{rb:>11.1%}")

    print("\n  * 수익은 동일가중 매수보유 기준")
    print("  * 위기 때 상관이 1에 가까워지면 분산이 무너진 것이다")

    print("\n  [평온기 대조]")
    print(f"  {'구간':<22}{'대상':>10}{'비교':>10}")
    print("  " + "─" * 42)
    for name, (s0, e0) in CALM.items():
        m = (r.index >= pd.Timestamp(s0)) & (r.index <= pd.Timestamp(e0))
        if m.sum() < 20:
            continue
        seg = r[m]
        ct = np.mean([seg[TARGET].corr().loc[a, b]
                      for a, b in combinations(TARGET, 2)])
        cb = np.mean([seg[BASE].corr().loc[a, b]
                      for a, b in combinations(BASE, 2)])
        print(f"  {name:<22}{ct:>10.2f}{cb:>10.2f}")

    # ---------- Q3 ----------
    print("\n" + "=" * 70)
    print("■ Q3. 롤링 250일 평균 상관 (연도별)\n")

    def roll_mean_corr(sub):
        pairs = list(combinations(sub.columns, 2))
        acc = None
        for a, b in pairs:
            c = sub[a].rolling(250).corr(sub[b])
            acc = c if acc is None else acc + c
        return acc / len(pairs)

    rt_c = roll_mean_corr(r[TARGET]).dropna()
    rb_c = roll_mean_corr(r[BASE]).dropna()
    print(f"  {'연도':<8}{'대상':>9}{'비교':>9}      {'대상 분포':<20}")
    print("  " + "─" * 46)
    for y in sorted(set(rt_c.index.year)):
        a = rt_c[rt_c.index.year == y].mean()
        b = rb_c[rb_c.index.year == y].mean()
        bar = "█" * max(int(a * 20), 0)
        print(f"  {y:<8}{a:>9.2f}{b:>9.2f}      {bar}")
    print(f"\n  대상 상관 범위: {rt_c.min():.2f} ~ {rt_c.max():.2f}"
          f"   (변동폭 {rt_c.max() - rt_c.min():.2f})")
    print(f"  비교 상관 범위: {rb_c.min():.2f} ~ {rb_c.max():.2f}"
          f"   (변동폭 {rb_c.max() - rb_c.min():.2f})")

    # ---------- Q4 ----------
    print("\n" + "=" * 70)
    print("■ Q4. 분산 효과 정량화\n")
    for label, cols in (("대상", TARGET), ("비교", BASE)):
        sub = r[cols]
        dr = div_ratio(sub)
        eq = (1 + sub.mean(axis=1)).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        vol = sub.mean(axis=1).std() * np.sqrt(TD)
        avg_vol = (sub.std() * np.sqrt(TD)).mean()
        print(f"  [{label}] {' / '.join(cols)}")
        print(f"    개별 평균 변동성  {avg_vol:.2%}")
        print(f"    동일가중 변동성   {vol:.2%}   "
              f"(감소폭 {1 - vol / avg_vol:.1%})")
        print(f"    분산비율          {dr:.3f}   (1.0 = 분산효과 없음)")
        print(f"    동일가중 MDD      {dd:.2%}\n")

    print("=" * 70)
    print("■ 해석 가이드")
    print("  · 평균 상관 0.7 이상이면 사실상 같은 자산 3개를 보유하는 것에 가깝다")
    print("  · 위기 상관이 평시보다 크게 오르면, 정작 필요할 때 분산이 사라진다")
    print("  · 분산비율이 1.2 미만이면 자산을 늘린 효익이 크지 않다")


if __name__ == "__main__":
    main()
