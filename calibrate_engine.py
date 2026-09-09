#!/usr/bin/env python3
"""
calibrate_engine.py
-------------------
eval_freq_study.py 의 백테스트 엔진을 기존 run_study.py 결과에 맞춘다.

[문제]
  평가주기 실험에서 일간 기준선이 기존 리포트와 어긋났다.
    전체 MDD      -16.41%  →  -15.18%
    -8% 에피소드     8건   →     9건
    2015 구간 전략  +0.87%  →   +5.10%
    2015 구간 QQQ  +17.90%  →  +16.19%
  QQQ 매수보유 MDD 는 -53.40% 로 정확히 일치하므로 데이터가 아니라
  엔진 사양의 차이다.

[유력 원인 — 리밸런싱 방식]
  현재 엔진은 매일 목표 비중으로 되돌린다(고정비중). 실제 운용은 신호가
  바뀔 때만 조정하고 그 사이 비중은 가격에 따라 흘러간다(드리프트).
  고정비중은 횡보장에서 자동 저가매수 효과가 생겨 성과가 과대평가된다.
  2015 구간에서 격차가 가장 크게 벌어진 것과 일치한다.

[방법]
  사양 후보를 격자로 돌려 기존 리포트 목표값과의 오차를 점수화한다.
  이것은 전략 최적화가 아니라 '이미 검증된 엔진의 사양 복원'이므로
  격자 탐색이 정당하다. 재현 후에는 사양을 고정한다.

환경변수:
  TARGET_MDD / TARGET_EPISODES / TARGET_2015_STRAT / TARGET_2015_QQQ
  TARGET_2015_MDD / TARGET_2015_SWITCH  (선택) 목표값 덮어쓰기
"""

import io
import os
import sys
import itertools
import contextlib
from datetime import datetime

import numpy as np
import pandas as pd

# ==============================================================
# CONFIG
# ==============================================================
def env(key, default):
    v = os.environ.get(key, "")
    if v is None or str(v).strip() == "":
        return default
    return type(default)(v)


BASE_WEIGHTS = {"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20}
MA_PERIODS = [20, 120, 200]
BAND_UP, BAND_DN = 1.015, 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

FETCH_START = "1999-01-01"
EVAL_START = "2004-11-18"
EVAL_END = env("EVAL_END", datetime.today().strftime("%Y-%m-%d"))
EP15 = ("2015-07-17", "2017-02-17")
TRADING_DAYS = 252
DD_THRESHOLD = -8.0

# ---------- 재현 목표값 (기존 리포트) --------------------------
TARGETS = {
    "mdd": env("TARGET_MDD", -16.41),
    "episodes": env("TARGET_EPISODES", 8),
    "s15": env("TARGET_2015_STRAT", 0.87),
    "q15": env("TARGET_2015_QQQ", 17.90),
    "m15": env("TARGET_2015_MDD", -11.60),
    "w15": env("TARGET_2015_SWITCH", 40),
}

# ---------- 탐색 격자 -----------------------------------------
GRID = {
    "drift": [True, False],          # True=신호변경 시만 조정, False=매일 고정비중
    "cost_bps": [0.0, 5.0, 10.0, 20.0],
    "exec_lag": [1, 2],
    "cash_yield": [0.0, 2.0],
    "total_return": [True, False],   # 배당 재투자 여부
}
TOP_N = 8
REPORT_PATH = "calibration_report.txt"
# =======================================================================


# ==============================================================
# FETCH
# ==============================================================
def fetch(total_return: bool) -> pd.DataFrame:
    import yfinance as yf

    tk = list(BASE_WEIGHTS)
    df = yf.download(tk, start=FETCH_START, end=EVAL_END,
                     auto_adjust=total_return, progress=False)
    px = df["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tk[0])
    idx = pd.to_datetime(px.index)
    px.index = idx.tz_localize(None) if idx.tz is not None else idx
    return px[tk].sort_index()


# ==============================================================
# SIGNAL
# ==============================================================
def score_series(close: pd.Series) -> pd.Series:
    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    state = {n: 0 for n in MA_PERIODS}
    vals = []
    for i in range(len(close)):
        p = float(close.iloc[i])
        for n in MA_PERIODS:
            ma = mas[n].iloc[i]
            if pd.isna(ma):
                state[n] = 0
                continue
            ma = float(ma)
            if p > ma * BAND_UP:
                state[n] = 1
            elif p < ma * BAND_DN:
                state[n] = 0
        vals.append(sum(state.values()))
    return pd.Series(vals, index=close.index)


def targets_frame(px: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        t[a] = score_series(px[a]).map(SCALAR_MAP) * BASE_WEIGHTS[a]
    return t


# ==============================================================
# BACKTEST
# ==============================================================
def backtest(px, tgt, drift, cost_bps, exec_lag, cash_yield):
    """drift=True 면 신호 변경일에만 목표비중으로 조정하고,
    그 사이에는 보유 비중이 가격에 따라 흘러간다."""
    w_t = tgt.shift(exec_lag)
    rets = px.pct_change().fillna(0.0)

    sl = slice(EVAL_START, EVAL_END)
    w_t, rets = w_t.loc[sl].fillna(0.0), rets.loc[sl]
    cash_r = (1 + cash_yield / 100) ** (1 / TRADING_DAYS) - 1

    n, k = len(w_t), w_t.shape[1]
    tv = w_t.to_numpy()
    rv = rets.to_numpy()

    port = np.empty(n)
    turn = np.zeros(n)
    held = tv[0].copy()

    # 목표가 바뀐 날 = 리밸런싱일
    changed = np.r_[True, (np.abs(np.diff(tv, axis=0)).sum(1) > 1e-9)]

    for i in range(n):
        if not drift or changed[i]:
            turn[i] = np.abs(tv[i] - held).sum()
            held = tv[i].copy()
        cash_w = max(0.0, 1.0 - held.sum())
        port[i] = float(held @ rv[i]) + cash_w * cash_r - \
            turn[i] * (cost_bps / 10000)
        if drift:                                   # 비중이 수익률만큼 흘러감
            grown = held * (1 + rv[i])
            total = grown.sum() + cash_w * (1 + cash_r)
            held = grown / total if total > 0 else grown

    out = pd.DataFrame({"ret": port, "turnover": turn}, index=w_t.index)
    out["equity"] = (1 + out["ret"]).cumprod()
    out["dd"] = out["equity"] / out["equity"].cummax() - 1
    out.attrs["changed"] = pd.Series(changed, index=w_t.index)
    return out


def count_episodes(res, thr):
    eq = res["equity"].to_numpy()
    n, i, peak = len(eq), 0, 0
    cnt = 0
    while i < n:
        if eq[i] >= eq[peak]:
            peak = i
            i += 1
            continue
        j, trough = i, i
        while j < n and eq[j] < eq[peak]:
            if eq[j] < eq[trough]:
                trough = j
            j += 1
        if (eq[trough] / eq[peak] - 1) * 100 <= thr:
            cnt += 1
        peak = j if j < n else n - 1
        i = j + 1
    return cnt


def evaluate(px, tgt, bench_eq, **spec):
    res = backtest(px, tgt, **spec)
    yrs = (res.index[-1] - res.index[0]).days / 365.25
    seg = res.loc[EP15[0]:EP15[1]]
    bseg = bench_eq.loc[EP15[0]:EP15[1]]

    if seg.empty or bseg.empty:
        return None

    eq15 = (1 + seg["ret"]).cumprod()
    ch = res.attrs["changed"].loc[EP15[0]:EP15[1]]

    return {
        "cagr": (res["equity"].iloc[-1] ** (1 / yrs) - 1) * 100,
        "mdd": res["dd"].min() * 100,
        "episodes": count_episodes(res, DD_THRESHOLD),
        "s15": (eq15.iloc[-1] - 1) * 100,
        "q15": (bseg.iloc[-1] / bseg.iloc[0] - 1) * 100,
        "m15": (eq15 / eq15.cummax() - 1).min() * 100,
        "w15": int(ch.sum()),
        "sharpe": (res["ret"].mean() * TRADING_DAYS) /
                  (res["ret"].std() * np.sqrt(TRADING_DAYS)),
    }


def score(m):
    """목표값과의 정규화 오차 합. 낮을수록 잘 맞음."""
    scale = {"mdd": 2.0, "episodes": 1.0, "s15": 2.0,
             "q15": 1.0, "m15": 2.0, "w15": 5.0}
    s = 0.0
    for k, target in TARGETS.items():
        s += abs(m[k] - target) / scale[k]
    return s


# ==============================================================
# MAIN
# ==============================================================
def main():
    line = "=" * 76
    print(line)
    print(" 엔진 캘리브레이션 — 기존 run_study.py 결과 재현")
    print(line)
    print(" 목표값 (기존 리포트):")
    print(f"   전체 MDD {TARGETS['mdd']:.2f}%  |  "
          f"-8% 에피소드 {TARGETS['episodes']:.0f}건")
    print(f"   2015 구간: 전략 {TARGETS['s15']:+.2f}%  "
          f"QQQ {TARGETS['q15']:+.2f}%  MDD {TARGETS['m15']:.2f}%  "
          f"변경 {TARGETS['w15']:.0f}회")
    print(line)
    print("\n※ 이것은 전략 최적화가 아니라 검증된 엔진의 사양 복원이다.")
    print("  재현 조합을 찾은 뒤에는 사양을 고정하고 실험에 재사용한다.\n")

    cache = {}
    rows = []
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"조합 {len(combos)}개 평가 중...")

    for combo in combos:
        spec = dict(zip(keys, combo))
        tr = spec.pop("total_return")
        if tr not in cache:
            px = fetch(tr)
            cache[tr] = (px, targets_frame(px),
                         (1 + px["QQQ"].pct_change()
                          .loc[EVAL_START:EVAL_END].fillna(0)).cumprod())
        px, tgt, bench_eq = cache[tr]
        m = evaluate(px, tgt, bench_eq, **spec)
        if m is None:
            continue
        m.update(spec)
        m["total_return"] = tr
        m["score"] = score(m)
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("score")

    print("\n" + line)
    print(f" 상위 {TOP_N}개 조합 (오차 점수 낮은 순)")
    print(line)
    hdr = (f"  {'드리프트':<8}{'비용':>6}{'지연':>5}{'현금':>6}{'TR':>5}"
           f"{'MDD':>9}{'에피':>5}{'2015전략':>10}{'2015QQQ':>9}"
           f"{'2015MDD':>9}{'변경':>6}{'점수':>8}")
    print(hdr)
    for _, r in df.head(TOP_N).iterrows():
        print(f"  {'유지' if r['drift'] else '매일':<8}"
              f"{r['cost_bps']:>5.0f}b{r['exec_lag']:>5.0f}"
              f"{r['cash_yield']:>5.1f}%{'TR' if r['total_return'] else 'PR':>5}"
              f"{r['mdd']:>8.2f}%{r['episodes']:>5.0f}"
              f"{r['s15']:>9.2f}%{r['q15']:>8.2f}%"
              f"{r['m15']:>8.2f}%{r['w15']:>6.0f}{r['score']:>8.2f}")

    best = df.iloc[0]
    print("\n" + line)
    print(" 최적 조합 상세")
    print(line)
    print(f"  리밸런싱   : {'신호 변경일만 (드리프트)' if best['drift'] else '매일 고정비중'}")
    print(f"  거래비용   : {best['cost_bps']:.0f}bp (편도)")
    print(f"  집행지연   : {best['exec_lag']:.0f}일")
    print(f"  현금수익률 : {best['cash_yield']:.1f}%")
    print(f"  가격기준   : {'배당재투자(TR)' if best['total_return'] else '가격(PR)'}")
    print()
    print(f"  {'항목':<14}{'재현값':>12}{'목표값':>12}{'차이':>12}")
    for k, lab in (("mdd", "전체 MDD"), ("episodes", "에피소드"),
                   ("s15", "2015 전략"), ("q15", "2015 QQQ"),
                   ("m15", "2015 MDD"), ("w15", "2015 변경")):
        t = TARGETS[k]
        print(f"  {lab:<14}{best[k]:>12.2f}{t:>12.2f}{best[k]-t:>+12.2f}")
    print(f"\n  참고: CAGR {best['cagr']:.2f}%  Sharpe {best['sharpe']:.3f}")

    print("\n" + line)
    print(" 해석")
    print(line)
    drift_gap = (df[df["drift"]]["score"].min() -
                 df[~df["drift"]]["score"].min())
    if drift_gap < 0:
        print("  → 드리프트 방식이 기존 결과를 더 잘 재현한다.")
        print("    eval_freq_study.py 의 run() 을 드리프트 방식으로 교체할 것.")
    else:
        print("  → 매일 고정비중이 더 잘 맞는다. 차이의 원인은 다른 곳에 있다.")
    if best["score"] > 8:
        print("\n  ⚠ 최적 조합도 오차가 크다. 격자에 없는 사양 차이가 있다.")
        print("    확인 후보: 스코어 산출 시 워밍업 처리, GLD 상장 전 구간 처리,")
        print("    현금을 KRW 단기채로 잡았는지, 세금·배당 처리 방식")
    print(line)


if __name__ == "__main__":
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            main()
        text = buf.getvalue()
    except Exception as e:                              # noqa: BLE001
        text = buf.getvalue() + f"\n[ERROR] {type(e).__name__}: {e}"
        print(text)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        sys.exit(1)

    print(text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"저장: {REPORT_PATH}")
