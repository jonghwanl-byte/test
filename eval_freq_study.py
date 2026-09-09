#!/usr/bin/env python3
"""
eval_freq_study.py
------------------
코어 TAA 평가주기 변형 실험 — 일간 / 주간 / 월간

[동기]
  -8% 이상 낙폭 8건 중 7건은 하락 대비 회복 기간이 정상 범위였다.
  2015-07 ~ 2017-02 한 건만 하락 122일 / 회복 279일로 이질적이고,
  QQQ 대비 방어폭도 2.3pp에 그쳤다. 방어 이득 없이 재진입 비용만
  치른 구간이다.

[가설]
  신호 평가 주기를 늦추면(주간/월간) 얕은 등락에서의 불필요한 이탈이
  줄어 2015년형 에피소드의 회복 기간이 단축된다. 대신 깊고 느린 하락
  (2008/2022)에서 대응이 늦어져 방어폭이 줄어든다.
  이 맞교환이 순이익인지 검정한다.

[왜 이 변형인가]
  평가주기는 파라미터 '추가'가 아니라 '대체'다. MA 기간·밴드·스칼라 맵을
  전혀 건드리지 않으므로 자유도가 늘지 않는다. 시행 횟수는 3(일/주/월).

[한계 — 미리 인지할 것]
  개선 대상 에피소드가 사실상 1건이다. 통과하더라도 '2015년에 맞춘 것'
  일 가능성을 배제할 수 없으므로 전후반 분할 재현을 필수로 본다.

[사전 합격기준]
  결과 출력 전에 인쇄한다. 실행 후 수정하지 않는다.
  미충족 시 현행(일간 평가)을 유지한다.

환경변수:
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (선택)
  COST_BPS      (선택) 편도 거래비용 bp. 기본 5
  CASH_YIELD    (선택) 현금 연수익률 %. 기본 0
  DD_THRESHOLD  (선택) 에피소드 판정 낙폭. 기본 -8
"""

import io
import os
import sys
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
BAND_UP = 1.015
BAND_DN = 0.975
SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

FETCH_START = "1999-01-01"          # 히스테리시스 워밍업 확보용
EVAL_START = env("EVAL_START", "2004-11-18")   # GLD 상장일
EVAL_END = env("EVAL_END", datetime.today().strftime("%Y-%m-%d"))

EXEC_LAG = 1                        # 당일 종가 산출 → 익영업일 집행
COST_BPS = env("COST_BPS", 5.0)     # 편도, 회전율 1단위당
CASH_YIELD = env("CASH_YIELD", 0.0) # 연 %
DD_THRESHOLD = env("DD_THRESHOLD", -8.0)
TRADING_DAYS = 252

VARIANTS = ["daily", "weekly", "monthly"]
N_TRIALS = len(VARIANTS)            # DSR 시행 횟수
REPORT_PATH = "eval_freq_report.txt"

# ---------- 사전 합격기준 (실행 전 확정) ----------------------
ACCEPTANCE = [
    "E1. 전체 MDD 가 일간 기준선보다 악화되지 않을 것",
    "E2. 2008·2022 방어폭이 기준선 대비 5pp 이내 열화",
    "E3. 최장 회복 기간 279일 → 200일 이하로 단축",
    "E4. Sharpe 개선 + DSR > 0.95 (시행 3회 반영)",
    "E5. 전후반 분할 표본 모두에서 E1·E3 방향 일치",
    "",
    "→ E1~E5 전부 충족한 변형만 채택 후보로 본다.",
    "→ 하나라도 미충족 시 현행 일간 평가를 유지한다.",
    "→ E3 이 이 실험의 존재 이유, E2 가 안전장치다.",
]
# =======================================================================


# ==============================================================
# FETCH
# ==============================================================
def fetch_prices():
    import yfinance as yf

    tk = list(BASE_WEIGHTS)
    df = yf.download(tk, start=FETCH_START, end=EVAL_END,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance 데이터 없음")

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
    """히스테리시스 상태 머신. 운영 봇과 동일한 규칙."""
    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    state = {n: 0 for n in MA_PERIODS}
    idx, vals = [], []
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
        idx.append(close.index[i])
        vals.append(sum(state.values()))
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def eval_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """평가일 여부. 주간=그 주 마지막 거래일, 월간=그 달 마지막 거래일."""
    s = pd.Series(True, index=index)
    if freq == "daily":
        return s
    key = index.to_period("W") if freq == "weekly" else index.to_period("M")
    last = pd.Series(index, index=index).groupby(key).transform("max")
    return pd.Series(index == last.values, index=index)


def build_weights(px: pd.DataFrame, freq: str) -> pd.DataFrame:
    """평가주기를 반영한 목표 비중. 평가일이 아니면 직전 비중 유지."""
    target = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        sc = score_series(px[a])
        target[a] = sc.map(SCALAR_MAP) * BASE_WEIGHTS[a]

    m = eval_mask(px.index, freq).to_numpy()
    mask2d = np.repeat(m[:, None], target.shape[1], axis=1)
    held = target.where(mask2d).ffill()
    return held.shift(EXEC_LAG)          # 익영업일 집행


# ==============================================================
# BACKTEST
# ==============================================================
def run(px: pd.DataFrame, freq: str) -> pd.DataFrame:
    w = build_weights(px, freq)
    rets = px.pct_change()

    sub = slice(EVAL_START, EVAL_END)
    w, rets = w.loc[sub].fillna(0.0), rets.loc[sub].fillna(0.0)

    cash_w = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    cash_r = (1 + CASH_YIELD / 100) ** (1 / TRADING_DAYS) - 1

    gross = (w * rets).sum(axis=1) + cash_w * cash_r
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * (COST_BPS / 10000)
    net = gross - cost

    out = pd.DataFrame({"ret": net, "turnover": turn})
    out["equity"] = (1 + out["ret"]).cumprod()
    out["dd"] = out["equity"] / out["equity"].cummax() - 1
    return out


def metrics(res: pd.DataFrame, bench: pd.Series) -> dict:
    r = res["ret"]
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = res["equity"].iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol else np.nan
    return {
        "CAGR": cagr * 100,
        "MDD": res["dd"].min() * 100,
        "Vol": vol * 100,
        "Sharpe": sharpe,
        "회전율": res["turnover"].sum() / yrs * 100,
        "n": len(r),
        "skew": r.skew(),
        "kurt": r.kurtosis() + 3,
    }


def dsr(sharpe, n, skew, kurt, n_trials):
    """Deflated Sharpe Ratio. Bailey & López de Prado."""
    from scipy.stats import norm

    if n < 100 or not np.isfinite(sharpe):
        return np.nan
    g = 0.5772156649
    z1 = norm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0.0
    z2 = norm.ppf(1 - 1 / (n_trials * np.e)) if n_trials > 1 else 0.0
    sr0 = np.sqrt(1 / TRADING_DAYS) * ((1 - g) * z1 + g * z2)   # 일간 단위

    sr_d = sharpe / np.sqrt(TRADING_DAYS)                        # 일간 SR
    denom = np.sqrt(max(1e-12,
                        1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return float(norm.cdf((sr_d - sr0) * np.sqrt(n - 1) / denom))


# ==============================================================
# EPISODES
# ==============================================================
def episodes(res: pd.DataFrame, thr: float):
    """낙폭 thr(%) 이상 구간을 (고점, 저점, 회복일, 낙폭%, 하락일, 회복일수)로."""
    dd = res["dd"]
    eq = res["equity"]
    out, i, n = [], 0, len(dd)
    peak_i = 0

    while i < n:
        if eq.iloc[i] >= eq.iloc[peak_i]:
            peak_i = i
            i += 1
            continue
        # 하락 시작 — 다음 신고가까지 추적
        j = i
        trough_i = i
        while j < n and eq.iloc[j] < eq.iloc[peak_i]:
            if eq.iloc[j] < eq.iloc[trough_i]:
                trough_i = j
            j += 1
        depth = (eq.iloc[trough_i] / eq.iloc[peak_i] - 1) * 100
        if depth <= thr:
            rec = (j - trough_i) if j < n else np.nan
            out.append({
                "peak": dd.index[peak_i], "trough": dd.index[trough_i],
                "recover": dd.index[j] if j < n else None,
                "depth": depth,
                "fall_days": trough_i - peak_i,
                "rec_days": rec,
            })
        peak_i = j if j < n else n - 1
        i = j + 1
    return out


def match_episode(eps, year):
    """지정 연도에 저점이 있는 에피소드를 찾는다."""
    for e in eps:
        if e["trough"].year == year:
            return e
    return None


# ==============================================================
# REPORT
# ==============================================================
def print_criteria():
    line = "=" * 74
    print(line)
    print(" 사전 합격기준 — 결과를 보기 전에 확정됨")
    print(line)
    for c in ACCEPTANCE:
        print(f" {c}")
    print(line)


def main():
    print_criteria()
    print(f"\n설정: 기본비중 {BASE_WEIGHTS} · MA {MA_PERIODS} · "
          f"밴드 +{(BAND_UP-1)*100:.1f}%/{(BAND_DN-1)*100:.1f}%")
    print(f"      집행지연 {EXEC_LAG}일 · 비용 {COST_BPS}bp(편도) · "
          f"현금 {CASH_YIELD}% · 낙폭기준 {DD_THRESHOLD}%")
    print(f"      구간 {EVAL_START} ~ {EVAL_END}\n")

    px = fetch_prices()
    print(f"데이터: {len(px):,}행 {px.index[0]:%Y-%m-%d} ~ {px.index[-1]:%Y-%m-%d}")

    bench = px["QQQ"].pct_change().loc[EVAL_START:EVAL_END].fillna(0)
    bench_eq = (1 + bench).cumprod()
    bench_mdd = (bench_eq / bench_eq.cummax() - 1).min() * 100

    results, eps_all, mets = {}, {}, {}
    for v in VARIANTS:
        r = run(px, v)
        results[v] = r
        eps_all[v] = episodes(r, DD_THRESHOLD)
        mets[v] = metrics(r, bench)

    line = "=" * 74
    # ---- 1. 기준선 재현 확인 ----
    print("\n" + line)
    print(" [1] 기준선(일간) 재현 확인 — 기존 낙폭 리포트와 대조")
    print(line)
    b = mets["daily"]
    print(f"  MDD {b['MDD']:.2f}%  (기존 리포트 -16.41%)")
    print(f"  QQQ 매수보유 MDD {bench_mdd:.2f}%  (기존 -53.40%)")
    print(f"  {DD_THRESHOLD}% 이상 낙폭 {len(eps_all['daily'])}건  (기존 8건)")
    print("  ⚠ 수치가 크게 다르면 비용·현금 가정을 맞춘 뒤 재실행할 것")

    # ---- 2. 변형별 요약 ----
    print("\n" + line)
    print(" [2] 변형별 성과")
    print(line)
    print(f"  {'변형':<10}{'CAGR':>8}{'MDD':>9}{'Vol':>8}"
          f"{'Sharpe':>9}{'회전율/년':>11}{'DSR':>8}")
    for v in VARIANTS:
        m = mets[v]
        d = dsr(m["Sharpe"], m["n"], m["skew"], m["kurt"], N_TRIALS)
        print(f"  {v:<10}{m['CAGR']:>7.2f}%{m['MDD']:>8.2f}%{m['Vol']:>7.2f}%"
              f"{m['Sharpe']:>9.3f}{m['회전율']:>10.0f}%{d:>8.3f}")

    # ---- 3. 에피소드 비교 ----
    print("\n" + line)
    print(f" [3] 에피소드 비교 (낙폭 {DD_THRESHOLD}% 이상)")
    print(line)
    for v in VARIANTS:
        print(f"\n  ── {v} — {len(eps_all[v])}건 " + "─" * 34)
        print(f"     {'고점':<12}{'저점':<12}{'낙폭':>8}"
              f"{'하락':>7}{'회복':>7}")
        for e in eps_all[v]:
            rec = f"{e['rec_days']:.0f}" if pd.notna(e["rec_days"]) else "미회복"
            print(f"     {e['peak']:%Y-%m-%d}  {e['trough']:%Y-%m-%d}  "
                  f"{e['depth']:>7.2f}%{e['fall_days']:>7d}{rec:>7}")

    # ---- 4. 핵심 지표 대조 ----
    print("\n" + line)
    print(" [4] 판정 근거 지표")
    print(line)
    print(f"  {'변형':<10}{'MDD':>9}{'2015회복':>10}{'2008방어':>10}"
          f"{'2022방어':>10}{'최장회복':>10}")
    rows = {}
    for v in VARIANTS:
        eps = eps_all[v]
        e15 = match_episode(eps, 2016) or match_episode(eps, 2015)
        e08 = match_episode(eps, 2008)
        e22 = match_episode(eps, 2022)

        def defend(e, qqq_dd):
            return (qqq_dd - e["depth"]) if e else np.nan

        # 동기간 QQQ 낙폭을 직접 계산
        def qqq_depth(e):
            if not e:
                return np.nan
            seg = bench_eq.loc[e["peak"]:e["trough"]]
            return (seg.iloc[-1] / seg.iloc[0] - 1) * 100

        d08 = defend(e08, qqq_depth(e08))
        d22 = defend(e22, qqq_depth(e22))
        r15 = e15["rec_days"] if e15 else np.nan
        longest = max((e["rec_days"] for e in eps
                       if pd.notna(e["rec_days"])), default=np.nan)
        rows[v] = dict(mdd=mets[v]["MDD"], r15=r15, d08=d08,
                       d22=d22, longest=longest)
        print(f"  {v:<10}{mets[v]['MDD']:>8.2f}%{r15:>10.0f}"
              f"{d08:>9.1f}p{d22:>9.1f}p{longest:>10.0f}")

    # ---- 5. 전후반 분할 ----
    print("\n" + line)
    print(" [5] 전후반 분할 재현")
    print(line)
    mid = results["daily"].index[len(results["daily"]) // 2]
    print(f"  분할점 {mid:%Y-%m-%d}")
    print(f"  {'변형':<10}{'전반 CAGR':>11}{'전반 MDD':>11}"
          f"{'후반 CAGR':>11}{'후반 MDD':>11}")
    halves = {}
    for v in VARIANTS:
        r = results[v]
        hs = []
        for seg in (r.loc[:mid], r.loc[mid:]):
            eq = (1 + seg["ret"]).cumprod()
            y = (seg.index[-1] - seg.index[0]).days / 365.25
            hs.append((eq.iloc[-1] ** (1 / y) - 1) * 100)      # CAGR
            hs.append((eq / eq.cummax() - 1).min() * 100)       # MDD
        halves[v] = hs
        print(f"  {v:<10}{hs[0]:>10.2f}%{hs[1]:>10.2f}%"
              f"{hs[2]:>10.2f}%{hs[3]:>10.2f}%")

    # ---- 6. 판정 ----
    print("\n" + line)
    print(" [6] 판정")
    print(line)
    base = rows["daily"]
    base_m = mets["daily"]
    any_pass = False
    for v in VARIANTS[1:]:
        rr, mm = rows[v], mets[v]
        d = dsr(mm["Sharpe"], mm["n"], mm["skew"], mm["kurt"], N_TRIALS)
        e1 = rr["mdd"] >= base["mdd"] - 1e-9
        e2 = (rr["d08"] >= base["d08"] - 5) and (rr["d22"] >= base["d22"] - 5)
        e3 = pd.notna(rr["longest"]) and rr["longest"] <= 200
        e4 = (mm["Sharpe"] > base_m["Sharpe"]) and (d > 0.95)
        h, hb = halves[v], halves["daily"]
        e5 = (h[1] >= hb[1] - 1e-9) and (h[3] >= hb[3] - 1e-9)
        ok = all([e1, e2, e3, e4, e5])
        any_pass |= ok
        print(f"\n  ── {v}")
        for k, val in (("E1 MDD 비악화", e1), ("E2 방어폭 유지", e2),
                       ("E3 최장회복 ≤200일", e3),
                       ("E4 Sharpe↑ & DSR>0.95", e4),
                       ("E5 전후반 일관", e5)):
            print(f"     {k:<24}{'✓' if val else '✗'}")
        print(f"     → {'채택 후보' if ok else '기각'}")

    print("\n" + line)
    if any_pass:
        print(" → 채택 후보 존재. 워크포워드 OOS 검증으로 진행할 것.")
        print("   (본 스크립트의 전후반 분할은 예비 점검이며 OOS 를 대체하지 않는다)")
    else:
        print(" → 전 변형 기각. 현행 일간 평가를 유지하고 기각로그에 기록한다.")
    print(line)


def send_telegram(text):
    token = env("TELEGRAM_BOT_TOKEN", "") or env("TELEGRAM_TOKEN", "")
    chat = env("TELEGRAM_CHAT_ID", "") or env("TELEGRAM_TO", "")
    if not token or not chat:
        print("[정보] 텔레그램 미설정 — 전송 생략", file=sys.stderr)
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for i in range(0, len(safe), 3500):
        try:
            requests.post(url, data={
                "chat_id": chat, "text": f"<pre>{safe[i:i+3500]}</pre>",
                "parse_mode": "HTML"}, timeout=20)
        except Exception as e:                          # noqa: BLE001
            print(f"[WARN] 텔레그램 실패: {e}", file=sys.stderr)


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
        send_telegram(text)
        sys.exit(1)

    print(text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"저장: {REPORT_PATH}")
    send_telegram(text)
