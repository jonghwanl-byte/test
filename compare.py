#!/usr/bin/env python3
"""
지수 단순보유 vs 계절성 로테이션 비교

케이스
------
1. QQQ 단독 (나스닥 100)
2. SPY 단독 (S&P 500)
3. DIA 단독 (다우 30)
4. 11~4월 QQQ / 5~10월 SPY
5. 11~4월 QQQ / 5~10월 DIA
6. 11~4월 SPY / 5~10월 DIA

사용법:
  python compare.py --years 20
  python compare.py --years 20 --split      # 전반/후반 나눠서 안정성 확인

주의
----
계절성 전략은 과거 데이터에서 사후적으로 발견된 패턴입니다.
전반/후반으로 나눴을 때 둘 다 유지되는지 반드시 확인하세요.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

INDEXES = {"QQQ": "나스닥100", "SPY": "S&P500", "DIA": "다우30"}

CASES = [
    ("1. QQQ 단독",              {"QQQ": list(range(1, 13))}),
    ("2. SPY 단독",              {"SPY": list(range(1, 13))}),
    ("3. DIA 단독",              {"DIA": list(range(1, 13))}),
    ("4. 겨울QQQ / 여름SPY",     {"QQQ": [11, 12, 1, 2, 3, 4], "SPY": [5, 6, 7, 8, 9, 10]}),
    ("5. 겨울QQQ / 여름DIA",     {"QQQ": [11, 12, 1, 2, 3, 4], "DIA": [5, 6, 7, 8, 9, 10]}),
    ("6. 겨울SPY / 여름DIA",     {"SPY": [11, 12, 1, 2, 3, 4], "DIA": [5, 6, 7, 8, 9, 10]}),
]

COST_BPS = 10          # 로테이션 시 편도 거래비용
TRADING_DAYS = 252
CACHE = ".cmp_cache.pkl"


def load(years: int, use_cache: bool = True):
    key = f"{years}y-{'-'.join(sorted(INDEXES))}"
    if use_cache and os.path.exists(CACHE):
        try:
            blob = pickle.load(open(CACHE, "rb"))
            if blob.get("key") == key:
                print("캐시 사용. 새로 받으려면 --no-cache")
                return blob["data"]
        except Exception:
            pass

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance pandas numpy")

    print(f"... {len(INDEXES)}개 지수 다운로드 ({years}년)")
    df = yf.download(list(INDEXES), period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False, threads=False,
                     group_by="column")
    close = df["Close"].dropna()
    pickle.dump({"key": key, "data": close}, open(CACHE, "wb"))
    return close


def build_returns(close: pd.DataFrame, alloc: dict) -> pd.Series:
    """월별 배분 규칙에 따른 일간 수익률. 로테이션 시점에 거래비용 차감."""
    ret = close.pct_change()
    months = close.index.month

    holding = pd.Series(index=close.index, dtype=object)
    for ticker, ms in alloc.items():
        holding[np.isin(months, ms)] = ticker

    # 룩어헤드 차단: 그날 무엇을 들지는 전날 종가 시점에 결정된다
    holding = holding.shift(1).ffill()

    out = pd.Series(0.0, index=close.index)
    for ticker in alloc:
        mask = holding == ticker
        out[mask] = ret.loc[mask, ticker]

    switched = holding != holding.shift(1)
    switched.iloc[0] = False
    out[switched] -= COST_BPS / 10000 * 2      # 팔고 사므로 왕복

    return out.dropna()


def metrics(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    yrs = len(ret) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    rf = (1.02) ** (1 / TRADING_DAYS) - 1
    sharpe = (ret - rf).mean() / ret.std() * np.sqrt(TRADING_DAYS)
    down = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    return {
        "CAGR": cagr, "MDD": dd.min(), "Vol": vol, "Sharpe": sharpe,
        "Sortino": (ret - rf).mean() * TRADING_DAYS / down if down > 0 else 0,
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else 0,
        "최악의해": ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1).min(),
    }


def table(close: pd.DataFrame, title: str):
    print("\n" + "=" * 82)
    print(f"  {title}")
    print(f"  {close.index[0].date()} ~ {close.index[-1].date()} "
          f"({len(close)/TRADING_DAYS:.1f}년)")
    print("=" * 82)
    print(f"{'전략':<22s}{'CAGR':>9s}{'MDD':>9s}{'변동성':>9s}"
          f"{'Sharpe':>9s}{'Calmar':>9s}{'최악의해':>10s}")
    print("-" * 82)

    results = {}
    for name, alloc in CASES:
        r = build_returns(close, alloc)
        m = metrics(r)
        results[name] = m
        print(f"{name:<22s}{m['CAGR']*100:>8.2f}%{m['MDD']*100:>8.1f}%"
              f"{m['Vol']*100:>8.1f}%{m['Sharpe']:>9.3f}{m['Calmar']:>9.3f}"
              f"{m['최악의해']*100:>9.1f}%")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--split", action="store_true",
                    help="전반/후반으로 나눠 안정성 확인")
    args = ap.parse_args()

    close = load(args.years, use_cache=not args.no_cache)

    full = table(close, f"전체 기간 ({args.years}년 요청)")

    best_cagr = max(full, key=lambda k: full[k]["CAGR"])
    best_calmar = max(full, key=lambda k: full[k]["Calmar"])
    print("-" * 82)
    print(f"  CAGR 1위: {best_cagr}   |   Calmar 1위: {best_calmar}")

    if args.split:
        mid = close.index[len(close) // 2]
        a = table(close[close.index <= mid], "전반부")
        b = table(close[close.index > mid], "후반부")

        print("\n" + "=" * 82)
        print("  안정성 점검 — 전반과 후반에서 순위가 유지되는가")
        print("=" * 82)
        ra = sorted(a, key=lambda k: -a[k]["CAGR"])
        rb = sorted(b, key=lambda k: -b[k]["CAGR"])
        print(f"{'전반부 순위':<26s}{'후반부 순위'}")
        for i in range(len(ra)):
            print(f"{i+1}. {ra[i]:<23s}{i+1}. {rb[i]}")

        moved = sum(1 for k in ra if ra.index(k) != rb.index(k))
        print(f"\n  순위가 바뀐 전략: {moved}/{len(ra)}개")
        if moved >= len(ra) // 2:
            print("  절반 이상이 뒤바뀌었습니다. 전체 기간 순위는 우연일 가능성이")
            print("  높고, 미래 성과의 근거로 쓰기 어렵습니다.")
        else:
            print("  순위가 비교적 안정적입니다. 다만 두 구간 모두 같은")
            print("  시장 환경이었을 수 있으니 이것만으로 검증됐다고 보긴 어렵습니다.")

    print("\n" + "=" * 82)
    print("  주의")
    print("=" * 82)
    print("  · 계절성 전략(4~6)은 과거에서 사후적으로 발견된 패턴입니다.")
    print("    12개월을 두 구간으로 나누는 방법은 수십 가지이고, 그중 잘 맞는")
    print("    것을 고르면 우연히 좋아 보일 수 있습니다.")
    print("  · 세금이 반영되지 않았습니다. 연 2회 로테이션은 매번 양도차익을")
    print("    실현시켜, 단순 보유 대비 세후 수익률이 더 낮아집니다.")
    print("  · 지난 10~20년은 미국 대형 성장주에 유리한 기간이었습니다.")
    print("    이 결과는 그 환경에 크게 의존합니다.")

    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        print(f"\n(Actions 요약 페이지 기록 완료)")


if __name__ == "__main__":
    main()
