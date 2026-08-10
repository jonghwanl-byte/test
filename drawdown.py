#!/usr/bin/env python3
"""
TAA 전략 낙폭 이력 추출

확정 설정(60/20/20, ±1.5%/-2.5%, MA 20/120/200, 100/75/50/0)으로
전 기간을 재현한 뒤, 임계치를 넘는 낙폭 구간을 모두 뽑아낸다.

각 구간에 대해:
  - 시작(전고점) / 저점 / 회복 날짜
  - 낙폭 크기, 하락 소요일, 회복 소요일
  - 같은 구간 QQQ 매수보유 낙폭 (비교)
  - 저점 시점의 자산별 비중 (전략이 어떻게 대응했는지)

사용:
  python drawdown.py            # -10% 이상 낙폭
  python drawdown.py 5          # -5% 이상 낙폭
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

EXEC_LAG = 1          # 익일 종가 체결 (룩어헤드 차단)
COST = 0.0010         # 편도 거래비용 10bp
START = "2004-11-18"  # GLD 상장일 (3자산 동시 가능 시점)
CASH_RATE = 0.02      # 현금 수익률 근사 (연율)
THRESHOLD = float(sys.argv[1]) / 100 if len(sys.argv) > 1 else 0.10
# =======================================================================


def scalar_series(px: pd.Series) -> pd.Series:
    """히스테리시스 상태 -> 투입 스케일(0~1). 전체 히스토리로 워밍업."""
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


def build():
    px = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                     progress=False, threads=False)["Close"]
    px = px[TICKERS].ffill().dropna()

    rets = px.pct_change().fillna(0.0)
    rf = CASH_RATE / 252

    weights, port = {}, pd.Series(rf, index=px.index)
    for t in TICKERS:
        # shift(1+EXEC_LAG): 신호 T -> 체결 T+1 -> 수익 T+2 부터
        s = scalar_series(px[t]).shift(1 + EXEC_LAG).fillna(0.0)
        w = BASE_WEIGHTS[t] * s
        weights[t] = w
        port += w * (rets[t] - rf) - COST * w.diff().abs().fillna(0.0)

    W = pd.DataFrame(weights)
    mask = px.index >= pd.Timestamp(START)
    return port[mask], W[mask], rets["QQQ"][mask]


def episodes(r: pd.Series, threshold: float):
    """낙폭 구간 추출. (전고점, 저점, 회복) 삼중항."""
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1

    out, i, n = [], 0, len(dd)
    while i < n:
        if dd.iloc[i] >= -1e-12:
            i += 1
            continue
        start = i                       # 낙폭 시작(전고점 다음날)
        while i < n and dd.iloc[i] < -1e-12:
            i += 1
        end = i - 1                     # 회복 직전 (또는 데이터 끝)
        seg = dd.iloc[start:end + 1]
        if seg.min() <= -threshold:
            trough = seg.idxmin()
            out.append({
                "peak_date": dd.index[start - 1] if start > 0 else dd.index[0],
                "trough_date": trough,
                "recover_date": dd.index[i] if i < n else None,
                "depth": float(seg.min()),
            })
    return out, dd


def main():
    print(f"낙폭 임계치: -{THRESHOLD:.0%}\n")
    port, W, qqq = build()
    eps, dd = episodes(port, THRESHOLD)

    eq_q = (1 + qqq).cumprod()
    dd_q = eq_q / eq_q.cummax() - 1

    print(f"평가구간: {port.index[0].date()} ~ {port.index[-1].date()} "
          f"({len(port) / 252:.1f}년)")
    print(f"전체 MDD: {dd.min():.2%}  |  QQQ 매수보유 MDD: {dd_q.min():.2%}")
    print(f"-{THRESHOLD:.0%} 이상 낙폭 구간: {len(eps)}건\n")
    print("=" * 78)

    for k, e in enumerate(eps, 1):
        p, t, rec = e["peak_date"], e["trough_date"], e["recover_date"]
        fall = len(port.loc[p:t]) - 1
        heal = (len(port.loc[t:rec]) - 1) if rec is not None else None

        seg_q = dd_q.loc[p:t]
        qdepth = float(seg_q.min()) if len(seg_q) else float("nan")

        w_tr = W.loc[t]
        w_pk = W.loc[p]

        print(f"\n[{k}] {p.date()} → {t.date()}"
              + (f" → {rec.date()}" if rec is not None else " → 미회복"))
        print(f"    낙폭        {e['depth']:>8.2%}   (QQQ 동기간 {qdepth:.2%})")
        print(f"    하락 기간   {fall:>5}거래일 ({fall / 21:.1f}개월)")
        if heal is not None:
            print(f"    회복 기간   {heal:>5}거래일 ({heal / 21:.1f}개월)")
            print(f"    총 소요     {fall + heal:>5}거래일 "
                  f"({(fall + heal) / 252:.1f}년)")
        else:
            print(f"    회복 기간   진행 중")
        print(f"    비중 변화   전고점 "
              f"{'/'.join(f'{w_pk[c]:.0%}' for c in TICKERS)}"
              f" (현금 {1 - w_pk.sum():.0%})"
              f"  →  저점 "
              f"{'/'.join(f'{w_tr[c]:.0%}' for c in TICKERS)}"
              f" (현금 {1 - w_tr.sum():.0%})")

    print("\n" + "=" * 78)
    print("\n■ 요약")
    if eps:
        d = pd.DataFrame(eps)
        falls = [len(port.loc[e['peak_date']:e['trough_date']]) - 1 for e in eps]
        heals = [len(port.loc[e['trough_date']:e['recover_date']]) - 1
                 for e in eps if e['recover_date'] is not None]
        print(f"  평균 낙폭        {d.depth.mean():.2%}")
        print(f"  최악 낙폭        {d.depth.min():.2%}  "
              f"({d.loc[d.depth.idxmin(), 'trough_date'].date()})")
        print(f"  평균 하락 기간   {np.mean(falls):.0f}거래일 "
              f"({np.mean(falls) / 21:.1f}개월)")
        if heals:
            print(f"  평균 회복 기간   {np.mean(heals):.0f}거래일 "
                  f"({np.mean(heals) / 21:.1f}개월)")
            print(f"  최장 회복 기간   {max(heals)}거래일 "
                  f"({max(heals) / 252:.1f}년)")
        print(f"  발생 빈도        {len(port) / 252 / len(eps):.1f}년에 1회")

    print("\n■ 연도별 최대 낙폭 (연중 고점 기준)")

    def within_year(r: pd.Series) -> pd.Series:
        def f(x):
            e = (1 + x).cumprod()
            return (e / e.cummax() - 1).min()
        return r.groupby(r.index.year).apply(f)

    ann, ann_q = within_year(port), within_year(qqq)
    ret = (1 + port).groupby(port.index.year).prod() - 1
    ret_q = (1 + qqq).groupby(qqq.index.year).prod() - 1

    print(f"  {'연도':<6}{'수익률':>8}{'MDD':>8}  |{'QQQ수익':>9}{'QQQ MDD':>9}")
    for y in ann.index:
        bar = "█" * int(abs(ann[y]) * 100)
        print(f"  {y:<6}{ret[y]:>8.1%}{ann[y]:>8.1%}  |"
              f"{ret_q[y]:>9.1%}{ann_q[y]:>9.1%}  {bar}")


if __name__ == "__main__":
    main()
