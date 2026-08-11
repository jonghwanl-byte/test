#!/usr/bin/env python3
"""
강세 자산 현금 재배분 검증

아이디어
--------
특정 자산이 3/3 ON이면 강한 추세일 가능성이 높다.
이때 놀고 있는 현금을 그 자산으로 몰아준다.

규칙
----
1. 기본 비중 x 스케일로 각 자산 목표비중 산출 (기존과 동일)
2. 3/3 ON 자산이 있으면, 남은 현금을 그 자산들에 배분
   - 여러 개면 기본 비중 비율대로 나눔
   - 자산별 상한(CAP)까지만
3. 약한 자산의 비중은 건드리지 않음 (현금만 재배분)
4. 3/3 ON 자산이 없으면 재배분 없음 (현금 유지)

판정 기준 (사전 확정)
--------------------
  CAGR 이 오르면서 MDD 가 유지(악화 1%p 이내)될 때만 채택.

사용:
  python realloc.py            # 3/3 조건
  python realloc.py 2          # 2/3 이상으로 조건 완화
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

STRONG = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # 강세 판정 스코어
# 상한 방식
#   ("abs", x)  : 모든 자산 공통 상한 x       (TLT 도 x 까지 커질 수 있음)
#   ("mult", k) : 기본 비중의 k 배까지        (QQQ 60->k*60, TLT 20->k*20)
CAPS = [("abs", 0.70), ("abs", 0.80), ("abs", 1.00),
        ("mult", 1.25), ("mult", 1.50), ("mult", 2.00)]
FILLS = [0.50, 1.00]                                    # 현금 충전 비율

PERIODS = {
    "2007-2009 금융위기": ("2007-10-09", "2009-06-01"),
    "2015-2016 횡보장": ("2015-07-17", "2017-02-17"),
    "2018 4분기 급락": ("2018-09-20", "2019-04-30"),
    "2020 코로나": ("2020-02-19", "2020-08-31"),
    "2021-2023 금리인상": ("2021-11-19", "2023-07-18"),
    "2023-2025 AI 랠리": ("2023-01-01", "2025-12-31"),
}
# =======================================================================
TD = 252


def score_frame(close: pd.DataFrame) -> pd.DataFrame:
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


def weights_from(scores: pd.DataFrame, cap, fill: float) -> pd.DataFrame:
    """스코어 -> 목표비중. cap/fill 로 재배분 강도 조절.

    cap  = ("abs", x) 공통 상한 x  |  ("mult", k) 기본비중의 k배  |  None 재배분 없음
    fill = 남은 현금 중 몇 %를 투입할지
    """
    base = np.array([BASE_WEIGHTS[t] for t in TICKERS])
    sc = scores[TICKERS].to_numpy()
    lut = np.array([SCALAR_MAP[i] for i in range(len(MA_PERIODS) + 1)])
    W = base * lut[sc.astype(np.int64)]              # (T, 3) 기본 목표비중

    if cap is not None and fill > 0:
        mode, v = cap
        cap_vec = np.full(3, v) if mode == "abs" else base * v
        strong = sc >= STRONG                        # 강세 자산 마스크
        cash = np.clip(1.0 - W.sum(axis=1), 0.0, 1.0)
        avail = cash * fill

        # 강세 자산들의 기본 비중 비율대로 배분
        wgt = strong * base                          # (T, 3)
        tot = wgt.sum(axis=1, keepdims=True)
        share = np.divide(wgt, tot, out=np.zeros_like(wgt), where=tot > 0)
        add = share * avail[:, None]

        # 상한 적용 후 남는 몫은 다른 강세 자산에 재분배 (1회)
        room = np.maximum(cap_vec - W, 0.0) * strong
        placed = np.minimum(add, room)
        left = (add - placed).sum(axis=1, keepdims=True)
        room2 = np.maximum(room - placed, 0.0)
        tot2 = room2.sum(axis=1, keepdims=True)
        share2 = np.divide(room2, tot2, out=np.zeros_like(room2), where=tot2 > 0)
        placed = placed + np.minimum(share2 * left, room2)
        W = W + placed

    return pd.DataFrame(W, index=scores.index, columns=TICKERS)


def run(W: pd.DataFrame, rets: pd.DataFrame, rf: float) -> pd.Series:
    r = pd.Series(rf, index=W.index)
    for t in TICKERS:
        w = W[t].shift(1 + EXEC_LAG).fillna(0.0)
        r += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)
    return r


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    ex = r - CASH_RATE / TD
    return {
        "CAGR": eq.iloc[-1] ** (1 / yrs) - 1,
        "Vol": r.std() * np.sqrt(TD),
        "MDD": dd.min(),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(TD),
        "최악연도": ((1 + r).groupby(r.index.year).prod() - 1).min(),
    }


def main():
    close = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                        progress=False, threads=False)["Close"]
    close = close[TICKERS].ffill().dropna()
    rets = close.pct_change().fillna(0.0)
    rf = CASH_RATE / TD

    SC = score_frame(close)
    m = np.asarray(close.index >= pd.Timestamp(START))

    # 기준 (재배분 없음)
    W0 = weights_from(SC, None, 0.0)
    base_r = run(W0, rets, rf)[m]
    b = stats(base_r)

    print(f"강세 판정: 스코어 {STRONG}/3 이상")
    print(f"평가구간: {base_r.index[0].date()} ~ {base_r.index[-1].date()} "
          f"({len(base_r)/TD:.1f}년)\n")
    print(f"■ 기준 (재배분 없음)")
    print(f"  CAGR {b['CAGR']:.2%}   MDD {b['MDD']:.2%}   "
          f"Sharpe {b['Sharpe']:.3f}   최악연도 {b['최악연도']:.1%}")
    print(f"  평균 현금 {1 - W0[m].sum(axis=1).mean():.1%}\n")

    print("=" * 78)
    print("■ 재배분 변형별 성과\n")
    print(f"  {'상한방식':<12}{'충전':>6} │{'CAGR':>8}{'ΔCAGR':>8}{'MDD':>9}"
          f"{'ΔMDD':>8}{'Sharpe':>9}{'평균현금':>9}{'판정':>8}")
    print("  " + "─" * 78)

    results = []
    for cap in CAPS:
        for fill in FILLS:
            W = weights_from(SC, cap, fill)
            r = run(W, rets, rf)[m]
            s = stats(r)
            d_cagr = s["CAGR"] - b["CAGR"]
            d_mdd = s["MDD"] - b["MDD"]          # 음수면 악화
            ok = (d_cagr > 0) and (d_mdd > -0.01)
            verdict = "채택" if ok else ("MDD악화" if d_cagr > 0 else "CAGR↓")
            cash = 1 - W[m].sum(axis=1).mean()
            label = (f"공통 {cap[1]:.0%}" if cap[0] == "abs"
                     else f"기본 x{cap[1]:.2f}")
            results.append({"label": label, "cap": cap, "fill": fill, **s,
                            "d_cagr": d_cagr, "d_mdd": d_mdd, "ok": ok})
            print(f"  {label:<12}{fill:>6.0%} │{s['CAGR']:>8.2%}"
                  f"{d_cagr:>+8.2%}{s['MDD']:>9.2%}{d_mdd:>+8.2%}"
                  f"{s['Sharpe']:>9.3f}{cash:>9.1%}{verdict:>8}")

    print("\n  판정 기준: CAGR 상승 AND MDD 악화 1%p 이내")

    df = pd.DataFrame(results)
    passed = df[df.ok]

    print("\n" + "=" * 78)
    if passed.empty:
        print("■ 결론: 기준을 통과한 변형 없음 → 기각\n")
        best = df.loc[df.d_cagr.idxmax()]
        print(f"  참고) CAGR 최대: {best.label} / 충전 {best.fill:.0%}"
              f"  →  CAGR {best.d_cagr:+.2%}, MDD {best.d_mdd:+.2%}")
        print(f"        수익을 늘린 만큼 낙폭도 커졌다면 단순 노출 증가일 뿐이다.")
    else:
        print(f"■ 결론: {len(passed)}개 변형이 기준 통과\n")
        best = passed.loc[passed.CAGR.idxmax()]
        print(f"  최적: {best.label} / 충전 {best.fill:.0%}")
        print(f"        CAGR {best.CAGR:.2%} ({best.d_cagr:+.2%})   "
              f"MDD {best.MDD:.2%} ({best.d_mdd:+.2%})   "
              f"Sharpe {best.Sharpe:.3f}")

    # ---- 구간별 확인 ----
    top = df.loc[df.CAGR.idxmax()]
    W_top = weights_from(SC, top.cap, top.fill)
    r_top = run(W_top, rets, rf)[m]

    print("\n" + "=" * 78)
    print(f"■ 구간별 비교 (기준 vs {top.label}/충전{top.fill:.0%})\n")
    print(f"  {'구간':<22}{'기준':>20}{'재배분':>20}{'차이':>9}")
    print("  " + "─" * 71)
    for name, (s0, e0) in PERIODS.items():
        mm = np.asarray((base_r.index >= pd.Timestamp(s0))
                        & (base_r.index <= pd.Timestamp(e0)))
        if mm.sum() < 30:
            continue
        cells = []
        for series in (base_r, r_top):
            seg = series[mm]
            tot = (1 + seg).prod() - 1
            eq = (1 + seg).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            cells.append((tot, dd))
        diff = cells[1][0] - cells[0][0]
        print(f"  {name:<22}"
              f"{cells[0][0]:>10.1%} ({cells[0][1]:>6.1%})"
              f"{cells[1][0]:>10.1%} ({cells[1][1]:>6.1%})"
              f"{diff:>9.1%}")

    print("\n  * 각 칸: 구간수익률 (구간MDD)")
    print("  * 2015-2016 이 개선되면서 2008/2022 가 악화되지 않아야 유효")

    # ---- 노출 변화 ----
    print("\n" + "=" * 78)
    print(f"■ 노출 변화 ({top.label}/충전{top.fill:.0%})\n")
    for t in TICKERS:
        print(f"  {t}  평균 {W0[m][t].mean():.1%} → {W_top[m][t].mean():.1%}"
              f"   최대 {W0[m][t].max():.0%} → {W_top[m][t].max():.0%}")
    print(f"  현금 평균 {1 - W0[m].sum(axis=1).mean():.1%}"
          f" → {1 - W_top[m].sum(axis=1).mean():.1%}")
    turn0 = W0[m].diff().abs().sum(axis=1).sum() / (m.sum() / TD)
    turn1 = W_top[m].diff().abs().sum(axis=1).sum() / (m.sum() / TD)
    print(f"  연 회전율 {turn0:.2f} → {turn1:.2f}")


if __name__ == "__main__":
    main()
