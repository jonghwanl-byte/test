#!/usr/bin/env python3
"""
xle_residual_study.py
---------------------
XLE(에너지 섹터) 새틀라이트 편입 검토 — 잔차 회귀 검정

[가설]
  XLE는 QQQ에 대한 베타 노출을 제거한 뒤에도, QQQ와 장기채가 동시에
  실패하는 레짐에서 유의한 양(+)의 초과수익을 남긴다.

[왜 잔차인가]
  XLE 수익률의 상당 부분은 시장 베타다. 그 부분은 이미 QQQ로 보유
  중이므로 분산 기여가 0이다. 편입을 정당화하는 것은 오직 잔차뿐이다.
  잔차가 유의하지 않으면 "QQQ 비중을 늘린 것"과 동일하다.

[레짐 정의 — 결과를 보기 전에 확정한다]
  Primary   : QQQ와 TLT의 126거래일 수익률이 동시에 음(-)인 날
              (= 방어 대상으로 지목한 '동시 실패' 상태 그 자체)
  Secondary : 미국 CPI 전년비 > 4%  (FRED, API 키 있을 때만)

[사전 합격기준]
  본 스크립트는 결과 출력 전에 기준을 먼저 인쇄한다. 기준은 아래
  ACCEPTANCE 에 하드코딩되어 있으며, 실행 후 수정하지 않는다.

환경변수:
  FRED_API_KEY   (선택) 있으면 CPI 기반 보조 레짐도 검정
  START / END    (선택) 표본 구간
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ==============================================================
# CONFIG
# ==============================================================
def env(key: str, default):
    v = os.environ.get(key, "")
    if v is None or str(v).strip() == "":
        return default
    return type(default)(v)


TARGET = "XLE"                       # 검정 대상
FACTORS_1 = ["QQQ"]                  # 1요인 모형
FACTORS_2 = ["QQQ", "TLT"]           # 2요인 모형 (코어 구성 반영)
FX = "USDKRW=X"

START = env("START", "1999-03-10")   # QQQ 상장일
END = env("END", datetime.today().strftime("%Y-%m-%d"))

REGIME_WINDOW = env("REGIME_WINDOW", 126)   # 약 6개월
TRADING_DAYS = 252

BOOT_ITERS = env("BOOT_ITERS", 5000)
BLOCK_DAYS = env("BLOCK_DAYS", 21)          # 블록 부트스트랩 블록 길이
SEED = 20260909

COST_HURDLE = env("COST_HURDLE", 1.4)       # 실질 누수 추정치(%/년)
REPORT_PATH = "xle_residual_report.txt"

# ---------- 사전 합격기준 (실행 전 확정) ----------------------
ACCEPTANCE = [
    "C1. 레짐 내 잔차 평균의 Newey-West t > 2.0",
    "C2. 블록 부트스트랩 95% CI 하한 > 0",
    "C3. 독립 레짐 에피소드 3개 이상, 그중 과반에서 잔차 평균 > 0",
    "C4. 1요인(QQQ) / 2요인(QQQ+TLT) 모형 모두에서 C1~C3 충족",
    f"C5. 연환산 잔차 알파 > {COST_HURDLE:.1f}%p (실질 비용 임계값)",
    "",
    "→ 하나라도 미충족 시 기각. 사후에 기준을 완화하지 않는다.",
    "→ C1~C4 통과 후에만 상품(유동성/추적차이) 검토로 진행한다.",
]
# =======================================================================


# ==============================================================
# FETCH
# ==============================================================
def fetch(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)   # TR 기준
    if df is None or df.empty:
        raise RuntimeError("yfinance 데이터 없음")

    px = df["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers if isinstance(tickers, str) else tickers[0])

    idx = pd.to_datetime(px.index)
    px.index = idx.tz_localize(None) if idx.tz is not None else idx
    return px.sort_index()


def fetch_cpi_regime(index) -> pd.Series | None:
    """FRED CPI 전년비 > 4% 구간. 키가 없으면 None."""
    key = env("FRED_API_KEY", "")
    if not key:
        return None
    try:
        from fredapi import Fred
        cpi = Fred(api_key=key).get_series("CPIAUCSL")
    except Exception as e:                              # noqa: BLE001
        print(f"[WARN] FRED 조회 실패: {e}", file=sys.stderr)
        return None

    cpi.index = pd.to_datetime(cpi.index)
    yoy = cpi.pct_change(12) * 100
    # 발표 시차 1개월을 반영해 한 달 미룬다(룩어헤드 방지)
    yoy = yoy.shift(1).reindex(index.union(yoy.index)).ffill().reindex(index)
    return yoy > 4.0


# ==============================================================
# COMPUTE
# ==============================================================
def newey_west_t(x: np.ndarray, lags: int | None = None):
    """평균이 0인지에 대한 Newey-West 보정 t통계량."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n

    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))

    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        g = (e[L:] @ e[:-L]) / n
        var += 2 * (1 - L / (lags + 1)) * g

    se = np.sqrt(max(var, 1e-18) / n)
    return mu, mu / se, n


def block_bootstrap_ci(x: np.ndarray, iters: int, block: int, seed: int):
    """블록 부트스트랩으로 평균의 95% CI. 자기상관 구조를 보존한다."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block * 3:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(iters)

    for i in range(iters):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()

    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def find_episodes(mask: pd.Series, min_len: int = 21):
    """연속된 레짐 구간을 에피소드로 묶는다."""
    eps, start = [], None
    for dt, on in mask.items():
        if on and start is None:
            start = dt
        elif not on and start is not None:
            eps.append((start, prev))
            start = None
        prev = dt
    if start is not None:
        eps.append((start, mask.index[-1]))
    return [(a, b) for a, b in eps
            if len(mask.loc[a:b]) >= min_len]


def regress(rets: pd.DataFrame, target: str, factors: list) -> pd.Series:
    """OLS 잔차. 절편 포함."""
    sub = rets[[target] + factors].dropna()
    y = sub[target].values
    X = np.column_stack([np.ones(len(sub))] + [sub[f].values for f in factors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = pd.Series(resid, index=sub.index, name="resid")
    out.attrs["beta"] = dict(zip(["alpha"] + factors, beta))
    return out


# ==============================================================
# REPORT
# ==============================================================
def print_criteria():
    line = "=" * 66
    print(line)
    print(" 사전 합격기준 — 결과를 보기 전에 확정됨")
    print(line)
    for c in ACCEPTANCE:
        print(f" {c}")
    print(line)


def evaluate(resid: pd.Series, mask: pd.Series, label: str, factors: list):
    """레짐 내 잔차 검정 결과를 인쇄하고 (t, ci_low, ann_alpha, ok) 반환."""
    print(f"\n── {label} | 요인: {' + '.join(factors)} " + "─" * 20)

    b = resid.attrs["beta"]
    print("  적합 계수: " + "  ".join(f"{k} {v:+.4f}" for k, v in b.items()))

    m = mask.reindex(resid.index).fillna(False)
    inside = resid[m].values
    outside = resid[~m].values

    if len(inside) < 60:
        print(f"  ⚠ 레짐 표본 부족 ({len(inside)}일) — 검정 불가")
        return np.nan, np.nan, np.nan, False

    mu, t, n = newey_west_t(inside)
    lo, hi = block_bootstrap_ci(inside, BOOT_ITERS, BLOCK_DAYS, SEED)
    ann = mu * TRADING_DAYS * 100

    mu_o, t_o, _ = newey_west_t(outside)
    ann_o = mu_o * TRADING_DAYS * 100

    print(f"  레짐 내  : {n:,}일 ({n / len(resid):.1%})  "
          f"연환산 잔차 {ann:+.2f}%p  NW t={t:+.2f}")
    print(f"  레짐 외  : {len(outside):,}일  "
          f"연환산 잔차 {ann_o:+.2f}%p  NW t={t_o:+.2f}")
    print(f"  부트스트랩 95% CI (일평균): [{lo * TRADING_DAYS * 100:+.2f}, "
          f"{hi * TRADING_DAYS * 100:+.2f}] %p/년")

    eps = find_episodes(m)
    pos = 0
    print(f"  에피소드 {len(eps)}개:")
    for a, bb in eps:
        seg = resid.loc[a:bb][m.loc[a:bb]]
        v = seg.mean() * TRADING_DAYS * 100
        pos += v > 0
        print(f"    {a:%Y-%m} ~ {bb:%Y-%m}  ({len(seg):>4}일)  {v:+7.2f} %p/년")

    c1 = t > 2.0
    c2 = lo > 0
    c3 = len(eps) >= 3 and pos > len(eps) / 2
    c5 = ann > COST_HURDLE
    print(f"  판정: C1 {'✓' if c1 else '✗'}  C2 {'✓' if c2 else '✗'}  "
          f"C3 {'✓' if c3 else '✗'} ({pos}/{len(eps)})  C5 {'✓' if c5 else '✗'}")

    return t, lo, ann, bool(c1 and c2 and c3)


def main():
    print_criteria()

    tickers = sorted(set([TARGET] + FACTORS_2))
    print(f"\n데이터 수집: {tickers}  {START} ~ {END}")
    px = fetch(tickers, START, END)
    print(f"  {len(px):,}행  결측 제외 후 {len(px.dropna()):,}행")

    rets = px.pct_change().dropna()
    print(f"  공통 표본: {rets.index[0]:%Y-%m-%d} ~ {rets.index[-1]:%Y-%m-%d}")

    # ---- 레짐 정의 (룩어헤드 없음: 과거 126일 수익률만 사용) ----
    r_qqq = px["QQQ"] / px["QQQ"].shift(REGIME_WINDOW) - 1
    r_tlt = px["TLT"] / px["TLT"].shift(REGIME_WINDOW) - 1
    primary = ((r_qqq < 0) & (r_tlt < 0)).reindex(rets.index).fillna(False)

    print(f"\n레짐 정의: QQQ·TLT {REGIME_WINDOW}일 수익률 동시 음(-)")
    print(f"  해당 일수 {primary.sum():,} / {len(primary):,} "
          f"({primary.mean():.1%})")

    results = []
    for factors in (FACTORS_1, FACTORS_2):
        resid = regress(rets, TARGET, factors)
        results.append(evaluate(resid, primary, "Primary (동시 실패)", factors))

    # ---- 보조 레짐: CPI ----
    cpi_mask = fetch_cpi_regime(rets.index)
    if cpi_mask is not None:
        print("\n" + "─" * 66)
        print(" 보조 검정 — CPI 전년비 > 4% (참고용, 합격기준 아님)")
        for factors in (FACTORS_1, FACTORS_2):
            resid = regress(rets, TARGET, factors)
            evaluate(resid, cpi_mask, "Secondary (CPI>4%)", factors)
    else:
        print("\n[정보] FRED_API_KEY 미설정 — CPI 보조 검정 생략")

    # ---- 종합 판정 ----
    line = "=" * 66
    print("\n" + line)
    ok = all(r[3] for r in results)
    alpha_ok = all(r[2] > COST_HURDLE for r in results if not np.isnan(r[2]))
    print(f" C1~C3: {'통과' if ok else '미충족'}")
    print(f" C4 (두 모형 모두): {'통과' if ok else '미충족'}")
    print(f" C5 (알파 > {COST_HURDLE:.1f}%p): {'통과' if alpha_ok else '미충족'}")
    print()
    if ok and alpha_ok:
        print(" → 전 기준 통과. 상품 단계(유동성·추적차이) 검토로 진행.")
    else:
        print(" → 기각. 기각로그에 기록하고 상품 탐색을 중단한다.")
    print(line)


def send_telegram(text: str) -> None:
    token = env("TELEGRAM_BOT_TOKEN", "") or env("TELEGRAM_TOKEN", "")
    chat = env("TELEGRAM_CHAT_ID", "") or env("TELEGRAM_TO", "")
    if not token or not chat:
        print("[정보] 텔레그램 미설정 — 전송 생략", file=sys.stderr)
        return

    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    for i in range(0, len(safe), 3500):
        try:
            r = requests.post(url, data={
                "chat_id": chat,
                "text": f"<pre>{safe[i:i + 3500]}</pre>",
                "parse_mode": "HTML",
            }, timeout=20)
            if r.status_code != 200:
                print(f"[WARN] 텔레그램 {r.status_code}: {r.text}", file=sys.stderr)
        except Exception as e:                          # noqa: BLE001
            print(f"[WARN] 텔레그램 전송 실패: {e}", file=sys.stderr)


if __name__ == "__main__":
    import io
    import contextlib

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

    print(text)                                         # Actions 로그
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"저장: {REPORT_PATH}")
    send_telegram(text)
