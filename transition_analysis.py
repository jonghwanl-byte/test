#!/usr/bin/env python3
"""
신호 상태 전환(예: 4->3, 3->2, ...) 이후 N일 수익률 분석

backtest.py 와 같은 폴더에 두고 실행하세요. backtest.py 의 캐시(.bt_cache.pkl)를
그대로 재사용하므로, backtest.py 를 한 번이라도 실행해서 캐시를 만들어둔 뒤
돌리면 재다운로드 없이 바로 됩니다.

사용법:
  python transition_analysis.py                 # 4개 MA(20/60/120/200) 기준
  python transition_analysis.py --drop60         # 3개 MA(20/120/200) 기준, 60일선 제외
  python transition_analysis.py --years 10 --horizons 5,10,20,60
"""

import argparse
import sys

import numpy as np
import pandas as pd

import backtest as bt  # 같은 폴더의 backtest.py 를 그대로 재사용


def on_count_series_generic(close: pd.Series, ma_periods: list,
                             band_up: float, band_dn: float,
                             confirm: bool = True) -> pd.Series:
    """backtest.py의 on_count_series를 MA 구성 가변으로 일반화한 버전."""
    mas = {n: close.rolling(n).mean() for n in ma_periods}
    vals = close.values
    ma_vals = {n: mas[n].values for n in ma_periods}

    state = {n: 0 for n in ma_periods}
    counts = np.full(len(close), np.nan)
    start = max(ma_periods)

    for i in range(start, len(close)):
        p, pp = vals[i], vals[i - 1]
        rising, falling = p > pp, p < pp
        for n in ma_periods:
            ma = ma_vals[n][i]
            if np.isnan(ma):
                state[n] = 0
                continue
            if state[n] == 1:
                if p < ma * band_dn and (falling or not confirm):
                    state[n] = 0
            else:
                if p > ma * band_up and (rising or not confirm):
                    state[n] = 1
        counts[i] = sum(state.values())

    return pd.Series(counts, index=close.index)


def find_transitions(count: pd.Series):
    """(전환일 인덱스 위치, from상태, to상태) 리스트."""
    vals = count.values
    out = []
    for i in range(1, len(vals)):
        a, b = vals[i - 1], vals[i]
        if np.isnan(a) or np.isnan(b):
            continue
        a, b = int(a), int(b)
        if a != b:
            out.append((i, a, b))
    return out


def forward_returns_for_transitions(close: pd.Series, count: pd.Series,
                                     horizons: list, exec_lag: int):
    """전환마다 EXEC_LAG 반영 후 각 horizon의 미래 수익률을 계산."""
    vals = close.values
    n = len(vals)
    rows = []
    for i, a, b in find_transitions(count):
        exec_i = i + exec_lag
        if exec_i >= n:
            continue
        base = vals[exec_i]
        if np.isnan(base) or base == 0:
            continue
        row = {"from": a, "to": b, "direction": "up" if b > a else "down"}
        ok = True
        for h in horizons:
            j = exec_i + h
            if j >= n or np.isnan(vals[j]):
                ok = False
                break
            row[f"fwd_{h}d"] = vals[j] / base - 1
        if ok:
            rows.append(row)
    return rows


def summarize(rows: list, horizons: list, label: str):
    if not rows:
        print(f"\n[{label}] 전환 이벤트가 없습니다.")
        return None
    df = pd.DataFrame(rows)
    df["transition"] = df["from"].astype(str) + "->" + df["to"].astype(str)

    print(f"\n{'=' * 78}")
    print(f"  {label}  (전환 이벤트 {len(df)}건)")
    print("=" * 78)

    agg = {}
    for h in horizons:
        col = f"fwd_{h}d"
        agg[f"평균_{h}d"] = (col, "mean")
    grouped = df.groupby("transition").agg(
        건수=("transition", "count"),
        **{f"평균_{h}일수익률": (f"fwd_{h}d", "mean") for h in horizons},
        **{f"승률_{h}일": (f"fwd_{h}d", lambda s: (s > 0).mean()) for h in horizons},
    )
    # 컬럼 순서 정렬: 건수, (평균, 승률) x horizons
    ordered_cols = ["건수"]
    for h in horizons:
        ordered_cols += [f"평균_{h}일수익률", f"승률_{h}일"]
    grouped = grouped[ordered_cols]

    # 전환 방향(하락/상승) 기준으로 상태값 정렬: 4->3, 3->2, 2->1, 1->0, 0->1, 1->2, ...
    def sort_key(t):
        a, b = t.split("->")
        a, b = int(a), int(b)
        direction = 0 if a > b else 1  # 하락전환 먼저
        return (direction, max(a, b), a)

    grouped = grouped.reindex(sorted(grouped.index, key=sort_key))

    disp = grouped.copy()
    for h in horizons:
        disp[f"평균_{h}일수익률"] = (disp[f"평균_{h}일수익률"] * 100).round(2)
        disp[f"승률_{h}일"] = (disp[f"승률_{h}일"] * 100).round(1)
    print(disp.to_string())

    # 하락전환 중 가장 손실 방어 효과가 크게 기대되는 구간 / 상승전환 중 가장 상승폭 큰 구간 하이라이트
    if len(horizons):
        h0 = horizons[0]
        down = grouped[grouped.index.str.contains("->") &
                        grouped.index.to_series().apply(lambda t: int(t.split("->")[0]) > int(t.split("->")[1]))]
        up = grouped[grouped.index.to_series().apply(lambda t: int(t.split("->")[0]) < int(t.split("->")[1]))]
        if len(down):
            worst = down[f"평균_{h0}일수익률"].idxmin()
            print(f"\n  하락전환 중 {h0}일 평균수익률이 가장 낮은 구간: {worst} "
                  f"({down.loc[worst, f'평균_{h0}일수익률']*100:.2f}%)")
        if len(up):
            best = up[f"평균_{h0}일수익률"].idxmax()
            print(f"  상승전환 중 {h0}일 평균수익률이 가장 높은 구간: {best} "
                  f"({up.loc[best, f'평균_{h0}일수익률']*100:.2f}%)")

    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--drop60", action="store_true",
                     help="60일선을 제외한 3개 MA(20/120/200)로 분석")
    ap.add_argument("--horizons", type=str, default="5,10,20,60",
                     help="콤마로 구분된 forward horizon(거래일)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--up", type=float, default=bt.BAND_UP)
    ap.add_argument("--dn", type=float, default=bt.BAND_DN)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    confirm = not args.no_confirm
    ma_periods = [20, 120, 200] if args.drop60 else [20, 60, 120, 200]
    label = "3개 MA (20/120/200, 60일선 제외)" if args.drop60 else "4개 MA (20/60/120/200)"

    prices = bt.load_prices(args.years, use_cache=not args.no_cache)
    print(f"수신 완료: {len(prices)}종목  |  기간 {args.years}년  |  구성: {label}")

    all_rows = []
    per_ticker_counts = {}
    for t, name in bt.TICKERS.items():
        close = prices.get(t)
        if close is None or len(close) < max(ma_periods) + 60:
            continue
        count = on_count_series_generic(close, ma_periods, args.up, args.dn, confirm)
        rows = forward_returns_for_transitions(close, count, horizons, bt.EXEC_LAG)
        for r in rows:
            r["ticker"] = t
            r["name"] = name
        all_rows.extend(rows)
        per_ticker_counts[t] = len(rows)

    summarize(all_rows, horizons, f"전체 종목 통합 — {label}")

    print(f"\n종목별 전환 이벤트 수 (참고):")
    for t, c in sorted(per_ticker_counts.items(), key=lambda x: -x[1]):
        print(f"  {bt.TICKERS[t]:20s} {c:4d}건")

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        fname = "transition_events_drop60.csv" if args.drop60 else "transition_events_full.csv"
        out_df.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"\n원본 이벤트 데이터 -> {fname}")


if __name__ == "__main__":
    main()
