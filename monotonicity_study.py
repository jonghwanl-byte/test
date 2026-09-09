#!/usr/bin/env python3
"""
monotonicity_study.py
---------------------
새틀라이트 스칼라 맵 단조성 검정 — 스코어 2 vs 3

[배경]
  현행 스칼라 맵 {3:100%, 2:75%, 1:50%, 0:0%} 은 "ON 신호가 많을수록
  이후 수익이 좋다"는 단조증가 관계를 전제한다. 이전 진단에서 스코어 2가
  스코어 3보다 이후 수익이 높은 패턴이 관찰되어, 전제 자체를 검정한다.

[착시 가능성 — 반드시 분해할 것]
  스코어 2는 성격이 다른 두 상태를 한 칸에 담고 있다.
    2↑ : 1에서 올라온 2 (추세 회복 초기 — 이후 수익이 높은 게 자연스러움)
    2↓ : 3에서 내려온 2 (추세 약화)
  분해하지 않고 평균 내면 2↑ 효과가 전체를 끌어올려 역전처럼 보인다.
  진짜 이상 신호는 '2↓ 단독으로 3을 초과'하는 경우뿐이다.

[중복 관측]
  전방수익률 창이 겹치므로 Newey-West 보정(lag = 기간)을 적용한다.
  블록 부트스트랩은 '날짜 단위'로 샘플링해 자산 간 상관을 보존한다.

[사전 합격기준]
  결과 출력 전에 기준을 먼저 인쇄한다. 실행 후 수정하지 않는다.
  기준 미충족 시 현행 맵을 유지한다(변경하지 않는 것이 기본값).

환경변수:
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID   (선택) 리포트 전송
  HORIZONS      (선택) 전방수익률 기간, 콤마 구분. 기본 "20,60,120"
  BOOT_ITERS    (선택) 부트스트랩 반복. 기본 2000
  BLOCK_DAYS    (선택) 블록 길이. 기본 21
"""

import io
import os
import sys
import json
import re
import contextlib
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# ==============================================================
# CONFIG
# ==============================================================
def env(key, default):
    v = os.environ.get(key, "")
    if v is None or str(v).strip() == "":
        return default
    return type(default)(v)


TICKERS = {
    "102110": "한국 주식",
    "283580": "중국 주식",
    "241180": "일본 주식",
    "453810": "인도 주식",
    "385560": "채권 30년",
    "148070": "채권 10년",
}

# --- 신호 규칙: satellite_signal.py 와 완전히 동일해야 한다 ---
MA_PERIODS = [20, 120, 200]
BAND_UP = 1.015
BAND_DN = 0.975
WARMUP_EXTRA = 250
EXEC_LAG = 1          # 당일 종가로 산출 → 익영업일 집행. 룩어헤드 방지에 필수

KRX_START = "20050101"
KRX_CHUNK_YEARS = 6
TRADING_DAYS = 252
KST = timezone(timedelta(hours=9))

HORIZONS = [int(h) for h in env("HORIZONS", "20,60,120").split(",")]
BOOT_ITERS = env("BOOT_ITERS", 2000)
BLOCK_DAYS = env("BLOCK_DAYS", 21)
SEED = 20260909
REPORT_PATH = "monotonicity_report.txt"

# ---------- 사전 합격기준 (실행 전 확정) ----------------------
ACCEPTANCE = [
    "M1. 자산 과반에서 스코어2 > 스코어3 (동일 방향)",
    "M2. 20/60/120일 전 기간대에서 역전이 일관되게 나타남",
    "M3. 2↓(3→2) 단독으로도 스코어3을 초과  ← 착시 배제의 핵심",
    "M4. (스코어2 − 스코어3) 차이의 블록 부트스트랩 95% CI 하한 > 0",
    "M5. 차이의 Newey-West t > 2.0",
    "",
    "→ M1~M5 전부 충족 시에만 '역전 실재'로 판정한다.",
    "→ 하나라도 미충족 시 현행 스칼라 맵을 유지한다.",
    "→ '역전 실재' 판정이 나와도 맵 변경은 별도 단계다. 스코어3이 늦은",
    "   것인지(=MA 조합 문제) 맵이 틀린 것인지 먼저 구분해야 한다.",
]
# =======================================================================


# ==============================================================
# FETCH
# ==============================================================
def _krx_windows(start, end):
    """pykrx 3000행 응답 제한 회피용 기간 분할."""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out = []
    while s <= e:
        nxt = min(s.replace(year=s.year + KRX_CHUNK_YEARS), e)
        out.append((s.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        if nxt >= e:
            break
        s = nxt + timedelta(days=1)
    return out


def _via_pykrx(code):
    """KRX 회원제 전환 이후 인증 없이 동작하는 일반 시세 엔드포인트."""
    from pykrx import stock

    today = datetime.now(KST).strftime("%Y%m%d")
    frames = []
    for f, t in _krx_windows(KRX_START, today):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = stock.get_market_ohlcv_by_date(f, t, code)
        except Exception:                               # noqa: BLE001
            continue
        if df is not None and len(df) and "종가" in df.columns:
            frames.append(df)

    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    s = pd.to_numeric(df["종가"], errors="coerce").dropna()
    return s[s > 0]


def _via_naver(code):
    """네이버 금융 siseJson 폴백."""
    today = datetime.now(KST).strftime("%Y%m%d")
    url = ("https://api.finance.naver.com/siseJson.naver"
           f"?symbol={code}&requestType=1&startTime={KRX_START}"
           f"&endTime={today}&timeframe=day")
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"})
    r.raise_for_status()
    rows = json.loads(re.sub(r",\s*]", "]", r.text.strip().replace("'", '"')))
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.index = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
    s = pd.to_numeric(df["종가"], errors="coerce").dropna()
    return s[s > 0].sort_index()


def fetch_all():
    out = {}
    for code, name in TICKERS.items():
        s = None
        for label, fn in (("pykrx", _via_pykrx), ("naver", _via_naver)):
            try:
                s = fn(code)
            except Exception as e:                      # noqa: BLE001
                print(f"  {code} [{label}] 실패: {e}")
                s = None
            if s is not None and len(s):
                print(f"  {code} {name:8s} [{label}] {len(s):,}일 "
                      f"({s.index[0]:%Y-%m} ~ {s.index[-1]:%Y-%m})")
                break
        if s is None:
            print(f"  {code} {name:8s} 실패 — 제외", file=sys.stderr)
        else:
            out[code] = s
    return out


# ==============================================================
# COMPUTE
# ==============================================================
def score_series(close: pd.Series) -> pd.Series:
    """satellite_signal.py 와 동일한 히스테리시스로 일별 스코어를 만든다."""
    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    state = {n: 0 for n in MA_PERIODS}
    idx, vals = [], []

    for i in range(max(MA_PERIODS) - 1, len(close)):
        price = float(close.iloc[i])
        for n in MA_PERIODS:
            ma = mas[n].iloc[i]
            if pd.isna(ma):
                state[n] = 0
                continue
            ma = float(ma)
            if price > ma * BAND_UP:
                state[n] = 1
            elif price < ma * BAND_DN:
                state[n] = 0
        idx.append(close.index[i])
        vals.append(sum(state.values()))

    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="score")
    return s.iloc[WARMUP_EXTRA:]        # 히스테리시스 수렴 구간 제거


def label_path(score: pd.Series) -> pd.Series:
    """스코어 2를 진입 방향으로 분해한다. 2↑ = 상승 진입, 2↓ = 하락 진입."""
    lab = score.astype(str)
    prev_diff = score.diff()
    # 직전 변화가 없던 구간은 마지막 변화 방향을 이어받는다
    direction = prev_diff.replace(0, np.nan).ffill()
    is2 = score == 2
    lab[is2 & (direction > 0)] = "2↑"
    lab[is2 & (direction < 0)] = "2↓"
    lab[is2 & direction.isna()] = "2?"
    return lab


def build_panel(prices: dict) -> pd.DataFrame:
    """(date, asset, score, label, fwd_h...) 롱 포맷 패널."""
    frames = []
    for code, close in prices.items():
        sc = score_series(close)
        if len(sc) < max(HORIZONS) + 60:
            print(f"  {code} 표본 부족 — 제외", file=sys.stderr)
            continue
        lab = label_path(sc)
        d = pd.DataFrame({"score": sc, "label": lab})
        d["asset"] = code
        for h in HORIZONS:
            # t일 종가로 스코어를 산출하고 t+EXEC_LAG 에 집행하므로,
            # 전방수익률도 t+LAG 진입 → t+LAG+h 청산으로 잡는다.
            # close(t+h)/close(t) 로 계산하면 룩어헤드가 된다.
            entry = close.shift(-EXEC_LAG)
            exitp = close.shift(-(EXEC_LAG + h))
            fwd = exitp / entry - 1
            d[f"f{h}"] = fwd.reindex(sc.index)
        frames.append(d.reset_index().rename(columns={"index": "date"}))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={panel.columns[0]: "date"})
    return panel


def nw_t(x, lags):
    """평균 = 0 검정의 Newey-West t통계량."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * ((e[L:] @ e[:-L]) / n)
    se = np.sqrt(max(var, 1e-18) / n)
    return mu, mu / se, n


def nw_se(x, lags):
    """평균의 Newey-West 보정 표준오차."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    var = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - L / (lags + 1)) * ((e[L:] @ e[:-L]) / n)
    return np.sqrt(max(var, 1e-18) / n)


def boot_diff_ci(panel: pd.DataFrame, col: str, grp_a, grp_b,
                 iters: int, block: int, seed: int):
    """(grp_a 평균 − grp_b 평균) 의 블록 부트스트랩 CI.

    날짜 단위로 블록 샘플링하여 자산 간 동시 상관을 보존한다.
    """
    d = panel[["date", "label", col]].dropna()
    d = d[d["label"].isin(list(grp_a) + list(grp_b))]
    if d.empty:
        return np.nan, np.nan

    d = d.sort_values("date")
    dates = np.sort(d["date"].unique())
    pos = np.searchsorted(dates, d["date"].values)
    order = np.argsort(pos, kind="stable")
    pos, vals = pos[order], d[col].values[order]
    is_a = d["label"].isin(list(grp_a)).values[order]

    starts = np.searchsorted(pos, np.arange(len(dates)), side="left")
    ends = np.searchsorted(pos, np.arange(len(dates)), side="right")

    rng = np.random.default_rng(seed)
    n_blocks = max(1, len(dates) // block)
    out = np.empty(iters)

    for i in range(iters):
        bs = rng.integers(0, max(1, len(dates) - block), n_blocks)
        sel = np.concatenate([np.arange(starts[b], ends[min(b + block - 1,
                              len(dates) - 1)]) for b in bs])
        if sel.size == 0:
            out[i] = np.nan
            continue
        v, a = vals[sel], is_a[sel]
        out[i] = (v[a].mean() if a.any() else np.nan) - \
                 (v[~a].mean() if (~a).any() else np.nan)

    out = out[~np.isnan(out)]
    if out.size < 100:
        return np.nan, np.nan
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


# ==============================================================
# REPORT
# ==============================================================
def ann(x, h):
    return x * TRADING_DAYS / h * 100


def report(panel: pd.DataFrame):
    line = "=" * 70

    # ---- 1. 스코어별 단조성 (풀링) ----
    print("\n" + line)
    print(" [1] 스코어별 이후 수익률 — 전체 풀링 (연환산 %)")
    print(line)
    hdr = "  스코어 " + "".join(f"{h:>10d}일" for h in HORIZONS) + "      관측"
    print(hdr)
    for s in [0, 1, 2, 3]:
        sub = panel[panel["score"] == s]
        cells = ""
        for h in HORIZONS:
            v = sub[f"f{h}"].dropna()
            cells += f"{ann(v.mean(), h):>11.2f}" if len(v) else f"{'—':>11}"
        print(f"  {s:^6d}{cells}   {len(sub):>7,}")

    # 단조성 위반 여부
    print("\n  단조성 점검 (3 > 2 > 1 > 0 이어야 함):")
    for h in HORIZONS:
        m = [panel.loc[panel["score"] == s, f"f{h}"].mean() for s in range(4)]
        ok = all(m[i] >= m[i - 1] for i in range(1, 4) if pd.notna(m[i]))
        bad = [f"{i}<{i-1}" for i in range(1, 4)
               if pd.notna(m[i]) and m[i] < m[i - 1]]
        print(f"    {h:>3}일: {'✓ 단조' if ok else '✗ 위반 ' + ', '.join(bad)}")

    # ---- 2. 스코어 2 분해 ----
    print("\n" + line)
    print(" [2] 스코어 2 경로 분해 — 착시 배제")
    print(line)
    print("  2↑ = 1에서 상승 진입 / 2↓ = 3에서 하락 진입")
    print("  라벨 " + "".join(f"{h:>10d}일" for h in HORIZONS) + "      관측")
    for lab in ["3", "2↑", "2↓", "1"]:
        sub = panel[panel["label"] == lab]
        cells = ""
        for h in HORIZONS:
            v = sub[f"f{h}"].dropna()
            cells += f"{ann(v.mean(), h):>11.2f}" if len(v) else f"{'—':>11}"
        print(f"  {lab:^5s}{cells}   {len(sub):>7,}")

    # ---- 3. 차이 검정 ----
    print("\n" + line)
    print(" [3] (스코어2 − 스코어3) 차이 검정")
    print(line)
    results = {}
    for name, grp_a in (("2 전체", ("2↑", "2↓", "2?")), ("2↓ 단독", ("2↓",))):
        print(f"\n  ── {name} vs 3 " + "─" * 30)
        for h in HORIZONS:
            a = panel.loc[panel["label"].isin(grp_a), f"f{h}"].dropna()
            b = panel.loc[panel["label"] == "3", f"f{h}"].dropna()
            if len(a) < 30 or len(b) < 30:
                print(f"    {h:>3}일: 표본 부족")
                continue
            diff = a.mean() - b.mean()
            # 각 그룹 평균의 NW 표준오차를 구해 차이의 SE 로 합산한다.
            # 전방수익률 창이 겹치므로 lag = 기간(h) 을 쓴다.
            se_a = nw_se(a.values, h)
            se_b = nw_se(b.values, h)
            se = np.sqrt(se_a ** 2 + se_b ** 2)
            t = diff / se if se > 0 else np.nan

            lo, hi = boot_diff_ci(panel, f"f{h}", grp_a, ("3",),
                                  BOOT_ITERS, BLOCK_DAYS, SEED)
            print(f"    {h:>3}일: 차이 {ann(diff, h):+7.2f}%p/년  "
                  f"NW t={t:+5.2f}  "
                  f"CI[{ann(lo, h):+7.2f}, {ann(hi, h):+7.2f}]")
            results[(name, h)] = (diff, t, lo)

    # ---- 4. 자산별 ----
    print("\n" + line)
    print(" [4] 자산별 (스코어2 − 스코어3), 연환산 %p")
    print(line)
    print("  자산       " + "".join(f"{h:>10d}일" for h in HORIZONS))
    flips = {h: 0 for h in HORIZONS}
    n_assets = 0
    for code, name in TICKERS.items():
        sub = panel[panel["asset"] == code]
        if sub.empty:
            continue
        n_assets += 1
        cells = ""
        for h in HORIZONS:
            a = sub.loc[sub["score"] == 2, f"f{h}"].dropna()
            b = sub.loc[sub["score"] == 3, f"f{h}"].dropna()
            if len(a) < 20 or len(b) < 20:
                cells += f"{'—':>11}"
                continue
            d = a.mean() - b.mean()
            flips[h] += d > 0
            cells += f"{ann(d, h):>11.2f}"
        print(f"  {name:10s}{cells}")
    print(f"\n  역전(2>3) 자산 수: " +
          "  ".join(f"{h}일 {flips[h]}/{n_assets}" for h in HORIZONS))

    # ---- 5. 판정 ----
    print("\n" + line)
    print(" [5] 판정")
    print(line)
    m1 = all(flips[h] > n_assets / 2 for h in HORIZONS)
    m2 = all(results.get(("2 전체", h), (np.nan,))[0] > 0 for h in HORIZONS)
    m3 = all(results.get(("2↓ 단독", h), (np.nan,))[0] > 0 for h in HORIZONS)
    m4 = all(results.get(("2↓ 단독", h), (0, 0, np.nan))[2] > 0
             for h in HORIZONS)
    m5 = all(results.get(("2↓ 단독", h), (0, np.nan))[1] > 2.0
             for h in HORIZONS)
    for k, v in (("M1 자산 과반 역전", m1), ("M2 전 기간대 일관", m2),
                 ("M3 2↓ 단독 역전", m3), ("M4 부트스트랩 CI>0", m4),
                 ("M5 NW t>2", m5)):
        print(f"  {k:22s} {'✓' if v else '✗'}")
    print()
    if all([m1, m2, m3, m4, m5]):
        print("  → 역전 실재. 다음 단계: 스코어3이 늦은 것인지")
        print("    (MA 조합 문제) 맵이 틀린 것인지 구분하는 별도 실험.")
    else:
        print("  → 기준 미충족. 현행 스칼라 맵 유지.")
        print("    기각로그에 기록하고 이 건은 종결한다.")
    print(line)


def print_criteria():
    line = "=" * 70
    print(line)
    print(" 사전 합격기준 — 결과를 보기 전에 확정됨")
    print(line)
    for c in ACCEPTANCE:
        print(f" {c}")
    print(line)


def send_telegram(text):
    token = env("TELEGRAM_BOT_TOKEN", "") or env("TELEGRAM_TOKEN", "")
    chat = env("TELEGRAM_CHAT_ID", "") or env("TELEGRAM_TO", "")
    if not token or not chat:
        print("[정보] 텔레그램 미설정 — 전송 생략", file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for i in range(0, len(safe), 3500):
        try:
            requests.post(url, data={
                "chat_id": chat, "text": f"<pre>{safe[i:i + 3500]}</pre>",
                "parse_mode": "HTML"}, timeout=20)
        except Exception as e:                          # noqa: BLE001
            print(f"[WARN] 텔레그램 실패: {e}", file=sys.stderr)


def main():
    print_criteria()
    print(f"\n규칙: MA {MA_PERIODS} · 밴드 +{(BAND_UP-1)*100:.1f}%/"
          f"{(BAND_DN-1)*100:.1f}% · 워밍업 {WARMUP_EXTRA}일")
    print(f"기간대 {HORIZONS} · 부트스트랩 {BOOT_ITERS:,}회 "
          f"(블록 {BLOCK_DAYS}일, 날짜 단위 샘플링)\n")

    print("데이터 수집:")
    prices = fetch_all()
    if not prices:
        raise RuntimeError("전 종목 수집 실패")

    panel = build_panel(prices)
    print(f"\n패널: {len(panel):,}행  "
          f"{panel['date'].min():%Y-%m-%d} ~ {panel['date'].max():%Y-%m-%d}  "
          f"{panel['asset'].nunique()}자산")
    report(panel)


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
