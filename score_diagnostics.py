#!/usr/bin/env python3
"""
점수별 조건부 수익률 진단
=========================

질문: 히스테리시스 점수(0~3)가 실제로 미래 수익률을 구분하는가?
특히 **점수 1(50% 스칼라) 구간의 기대수익이 음수인가?**

점수 1은 20일선만 ON 이고 120/200일선은 OFF 인 상태가 대부분이다.
하락 추세에서 20일선을 타고 내려오는 구간이 여기 해당하며, 이때
기본비중의 50%를 계속 들고 있는 것이 합리적인지 검증한다.

[측정 방식]
  - 점수는 t일 종가로 산출, 포지션은 t+1일 종가에 진입 (lag=1)
  - 선행수익률 r(t) = P(t+1+h) / P(t+1) - 1
  - 점수별로 r 의 분포를 집계

[통계 처리]
  - 선행수익률 구간이 겹치므로 자기상관이 발생한다.
    -> Newey-West 보정 t값 사용 (lag = h)
  - 신뢰구간은 날짜 블록 부트스트랩으로 산출한다.
    자산 간 상관을 보존하기 위해 '날짜' 단위로 블록을 뽑는다.
  - 전 기간 결과만으로 판단하지 않는다. 3분할 하위기간에서
    부호가 일관되는지 함께 본다.

[반사실 검증]
  점수 1의 스칼라를 0.50 -> 0.00 으로 바꾼 포트폴리오를 실제로 돌려
  거래비용 차감 후 CAGR / MDD / Sharpe / 회전율을 비교한다.
  통계적 유의성과 실제 개선은 다른 문제이므로 둘 다 본다.

[주의]
  이 스크립트는 파라미터를 탐색하지 않는다. 사전에 정한 통과 기준에
  비추어 하나의 가설만 검증한다. 결과를 본 뒤 기준을 바꾸면 진단이
  아니라 사후 합리화가 된다.

환경변수:
  TICKERS             (선택) "102110.KS:한국주식,..." 형식으로 유니버스 재정의
  HORIZONS            (선택) 선행수익률 기간. 기본 "5,20,60"
  COST_BPS            (선택) 편도 거래비용(bp). 기본 "15"
  N_BOOT              (선택) 부트스트랩 횟수. 기본 "500"
  TELEGRAM_BOT_TOKEN  (선택) 설정 시 요약 전송
  TELEGRAM_CHAT_ID    (선택)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# ===== 설정 ============================================================
DEFAULT_UNIVERSE = {
    "102110.KS": "한국 주식",
    "283580.KS": "중국 주식",
    "241180.KS": "일본 주식",
    "453810.KS": "인도 주식",
    "385560.KS": "국고채 30년",
    "148070.KS": "국고채 10년",
    "426030.KS": "나스닥 주식",
}

MA_PERIODS = [20, 120, 200]
BAND_UP = 1.015
BAND_DN = 0.975

SCALAR_BASE = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}   # 현행
SCALAR_ALT = {3: 1.00, 2: 0.75, 1: 0.00, 0: 0.00}    # 점수1 제거안

BASE_WEIGHT = 0.10          # 자산별 상한 (10% 로 하향 조정됨)
WARMUP_EXTRA = 250
KRX_START = "20050101"
US_START = "2005-01-01"
BLOCK_DAYS = 40             # 부트스트랩 블록 길이(거래일)
SEED = 20260906

KST = timezone(timedelta(hours=9))
OUTDIR = "output"
# =======================================================================


def env(key: str, default: str = "") -> str:
    """빈 문자열로 주입된 환경변수도 미설정으로 취급한다."""
    return (os.environ.get(key) or default).strip()


def parse_universe() -> dict:
    raw = env("TICKERS")
    if not raw:
        return dict(DEFAULT_UNIVERSE)
    out = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ticker, _, name = item.partition(":")
        out[ticker.strip()] = name.strip() or ticker.strip()
    return out or dict(DEFAULT_UNIVERSE)


UNIVERSE = parse_universe()
HORIZONS = [int(x) for x in env("HORIZONS", "5,20,60").split(",") if x.strip()]
COST_BPS = float(env("COST_BPS", "15"))
N_BOOT = int(env("N_BOOT", "500"))


# ---------- 데이터 수집 -------------------------------------------------
def _clean(s) -> pd.Series | None:
    if s is None or len(s) == 0:
        return None
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def fetch_kr(ticker: str):
    from pykrx import stock

    code = ticker.split(".")[0]
    today = datetime.now(KST).strftime("%Y%m%d")
    for getter in (stock.get_etf_ohlcv_by_date, stock.get_market_ohlcv_by_date):
        try:
            df = getter(KRX_START, today, code)
        except Exception:                                # noqa: BLE001
            continue
        if df is not None and len(df) and "종가" in df.columns:
            return _clean(df["종가"])
    return None


def fetch_us(ticker: str):
    import yfinance as yf

    df = yf.download(ticker, start=US_START, interval="1d",
                     auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return None
        return _clean(df["Close"])
    return _clean(df["Close"]) if "Close" in df.columns else None


def fetch_prices() -> dict:
    out = {}
    for ticker, name in UNIVERSE.items():
        s = None
        try:
            s = fetch_kr(ticker) if ticker.endswith((".KS", ".KQ")) else fetch_us(ticker)
        except Exception as e:                           # noqa: BLE001
            print(f"  {ticker} 수신 오류: {e}", file=sys.stderr)

        need = max(MA_PERIODS) + WARMUP_EXTRA
        if s is None or len(s) < need:
            got = 0 if s is None else len(s)
            print(f"  {ticker} ({name}) 제외 — {got}일 / 최소 {need}일",
                  file=sys.stderr)
            continue

        out[ticker] = s
        print(f"  {ticker} ({name}): {len(s)}일, "
              f"{s.index[0].date()} ~ {s.index[-1].date()}")
        time.sleep(0.3)
    return out


# ---------- 점수 산출 ---------------------------------------------------
def score_series(close: pd.Series) -> pd.Series:
    """히스테리시스 상태 합계(0~3). 시그널 봇과 동일한 규칙."""
    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    px = close.to_numpy(dtype=float)
    state = {n: 0 for n in MA_PERIODS}
    scores = np.full(len(close), np.nan)

    start = max(MA_PERIODS) - 1
    for i in range(start, len(close)):
        total = 0
        for n in MA_PERIODS:
            ma = mas[n].iloc[i]
            if pd.isna(ma):
                state[n] = 0
            else:
                ma = float(ma)
                if px[i] > ma * BAND_UP:
                    state[n] = 1
                elif px[i] < ma * BAND_DN:
                    state[n] = 0
            total += state[n]
        scores[i] = total

    return pd.Series(scores, index=close.index)


def build_panel(prices: dict) -> pd.DataFrame:
    """[date, asset, score, fwd_h...] 롱 포맷 패널.

    fwd_h = P(t+1+h) / P(t+1) - 1  (t 종가 신호 -> t+1 진입, lag=1)
    """
    frames = []
    for ticker, close in prices.items():
        sc = score_series(close)
        entry = close.shift(-1)                          # t+1 진입가
        df = pd.DataFrame({"score": sc})
        for h in HORIZONS:
            df[f"fwd{h}"] = close.shift(-(1 + h)) / entry - 1.0
        df["asset"] = UNIVERSE.get(ticker, ticker)
        df["ticker"] = ticker
        df = df.dropna(subset=["score"])
        frames.append(df.reset_index().rename(columns={"index": "date",
                                                       df.index.name or "index": "date"}))

    panel = pd.concat(frames, ignore_index=True)
    panel.columns = ["date" if c in ("Date", "index", None) else c
                     for c in panel.columns]
    panel["score"] = panel["score"].astype(int)
    return panel


# ---------- 통계 --------------------------------------------------------
def newey_west_t(x: np.ndarray, lag: int) -> float:
    """평균이 0인지에 대한 Newey-West 보정 t값."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        g = (e[l:] @ e[:-l]) / n
        var += 2 * (1 - l / (lag + 1)) * g
    if var <= 0:
        return np.nan
    return mu / np.sqrt(var / n)


def block_bootstrap_means(panel: pd.DataFrame, col: str,
                          n_boot: int = N_BOOT) -> dict:
    """날짜 블록 부트스트랩으로 점수별 평균의 신뢰구간을 낸다.

    자산 간 동시적 상관을 보존하기 위해 자산이 아닌 '날짜'를 블록으로 뽑는다.
    """
    rng = np.random.default_rng(SEED)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)
    if n_dates < BLOCK_DAYS * 3:
        return {}

    by_date = {d: g for d, g in panel.groupby("date")}
    n_blocks = int(np.ceil(n_dates / BLOCK_DAYS))
    draws = {s: [] for s in sorted(panel["score"].unique())}

    for _ in range(n_boot):
        starts = rng.integers(0, n_dates, size=n_blocks)
        picked = []
        for st in starts:
            idx = [(st + k) % n_dates for k in range(BLOCK_DAYS)]
            picked.extend(dates[i] for i in idx)

        sample = pd.concat([by_date[d] for d in picked if d in by_date],
                           ignore_index=True)
        grp = sample.groupby("score")[col].mean()
        for s in draws:
            draws[s].append(grp.get(s, np.nan))

    out = {}
    for s, vals in draws.items():
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 50:
            continue
        out[s] = (float(np.percentile(arr, 2.5)),
                  float(np.percentile(arr, 97.5)))
    return out


def score_table(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    col = f"fwd{h}"
    sub = panel.dropna(subset=[col])
    rows = []
    ci = block_bootstrap_means(sub, col)

    for s, g in sub.groupby("score"):
        x = g[col].to_numpy(dtype=float)
        lo, hi = ci.get(s, (np.nan, np.nan))
        rows.append({
            "점수": s,
            "관측수": len(x),
            "비중(%)": 100 * len(x) / len(sub),
            "평균(%)": 100 * x.mean(),
            "중앙값(%)": 100 * float(np.median(x)),
            "연율(%)": 100 * ((1 + x.mean()) ** (252 / h) - 1),
            "승률(%)": 100 * float((x > 0).mean()),
            "NW_t": newey_west_t(x, h),
            "CI하단(%)": 100 * lo,
            "CI상단(%)": 100 * hi,
        })
    return pd.DataFrame(rows).sort_values("점수").reset_index(drop=True)


def subperiod_table(panel: pd.DataFrame, h: int, n_splits: int = 3) -> pd.DataFrame:
    col = f"fwd{h}"
    sub = panel.dropna(subset=[col]).copy()
    dates = np.sort(sub["date"].unique())
    edges = np.array_split(dates, n_splits)

    rows = []
    for i, seg in enumerate(edges, 1):
        part = sub[sub["date"].isin(seg)]
        label = (f"{pd.Timestamp(seg[0]).date()}~{pd.Timestamp(seg[-1]).date()}")
        rec = {"구간": f"P{i}", "기간": label}
        for s, g in part.groupby("score"):
            rec[f"점수{s}(%)"] = 100 * g[col].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def per_asset_table(panel: pd.DataFrame, h: int, score: int = 1) -> pd.DataFrame:
    col = f"fwd{h}"
    sub = panel[(panel["score"] == score)].dropna(subset=[col])
    rows = []
    for asset, g in sub.groupby("asset"):
        x = g[col].to_numpy(dtype=float)
        rows.append({
            "자산": asset,
            "관측수": len(x),
            "평균(%)": 100 * x.mean(),
            "승률(%)": 100 * float((x > 0).mean()),
            "NW_t": newey_west_t(x, h),
        })
    return pd.DataFrame(rows).sort_values("평균(%)").reset_index(drop=True)


# ---------- 반사실 백테스트 ---------------------------------------------
def sleeve_backtest(prices: dict, scalar_map: dict) -> dict:
    """점수 -> 비중 규칙을 적용한 슬리브 성과. lag=1, 비용 차감."""
    scores = pd.DataFrame({t: score_series(s) for t, s in prices.items()})
    rets = pd.DataFrame({t: s.pct_change() for t, s in prices.items()})

    common = scores.dropna(how="all").index
    scores = scores.reindex(common).ffill()
    rets = rets.reindex(common).fillna(0.0)

    weights = scores.map(lambda v: np.nan if pd.isna(v)
                         else BASE_WEIGHT * scalar_map[int(v)]).fillna(0.0)
    held = weights.shift(1).fillna(0.0)                  # lag=1

    gross = (held * rets).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * (COST_BPS / 10000.0)
    net = gross - cost

    eq = (1 + net).cumprod()
    valid = eq.dropna()
    if len(valid) < 252:
        return {}

    years = len(valid) / 252
    cagr = valid.iloc[-1] ** (1 / years) - 1
    dd = valid / valid.cummax() - 1
    ann_vol = net.std() * np.sqrt(252)

    return {
        "CAGR(%)": 100 * cagr,
        "MDD(%)": 100 * dd.min(),
        "Sharpe": (net.mean() * 252) / ann_vol if ann_vol > 0 else np.nan,
        "연변동성(%)": 100 * ann_vol,
        "연회전율(회)": turnover.sum() / years / 2,
        "평균투자비중(%)": 100 * held.sum(axis=1).mean(),
        "비용합계(%)": 100 * cost.sum(),
        "_equity": eq,
    }


# ---------- 리포트 ------------------------------------------------------
CRITERIA = """사전 통과 기준 (결과 확인 전 확정)
  1. 점수 1 구간 평균 선행수익률이 음수이고, NW t < -2.0
  2. 부트스트랩 95% 신뢰구간 상단이 0 미만
  3. 3분할 하위기간 중 2개 이상에서 부호가 음수로 일관
  4. 자산별로 봤을 때 과반이 음수 (특정 종목 한두 개가 끌고 가는 것이 아님)
  5. 반사실 백테스트에서 비용 차감 후 Sharpe 개선 +0.03 이상,
     MDD 악화 3%p 이내

  다섯 항목 모두 충족 -> SCALAR_MAP[1] 을 0.00 으로 변경
  하나라도 미충족    -> 현행 유지. 이번 손실은 추세추종의 정상 비용으로 간주"""


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df.empty:
        return "_데이터 없음_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body) + "\n"


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    print("=" * 60)
    print(CRITERIA)
    print("=" * 60)
    print(f"\n유니버스 {len(UNIVERSE)}종목 · 선행기간 {HORIZONS} · "
          f"비용 {COST_BPS:.0f}bp · 부트스트랩 {N_BOOT}회\n")

    prices = fetch_prices()
    if len(prices) < 2:
        print("사용 가능한 자산이 부족합니다.", file=sys.stderr)
        return 1

    panel = build_panel(prices)
    panel.to_csv(f"{OUTDIR}/panel.csv", index=False, encoding="utf-8-sig")

    md = ["# 점수별 조건부 수익률 진단", "",
          "```", CRITERIA, "```", "",
          f"- 자산 {len(prices)}종목 · 관측 {len(panel):,}행",
          f"- 기간 {pd.Timestamp(panel['date'].min()).date()} ~ "
          f"{pd.Timestamp(panel['date'].max()).date()}",
          f"- 거래비용 {COST_BPS:.0f}bp(편도) · 부트스트랩 {N_BOOT}회", ""]

    for h in HORIZONS:
        tbl = score_table(panel, h)
        tbl.to_csv(f"{OUTDIR}/score_h{h}.csv", index=False, encoding="utf-8-sig")
        md += [f"## 선행 {h}일 — 점수별 집계", "", md_table(tbl), ""]

        sp = subperiod_table(panel, h)
        sp.to_csv(f"{OUTDIR}/subperiod_h{h}.csv", index=False, encoding="utf-8-sig")
        md += [f"### 하위기간 3분할 (평균 %)", "", md_table(sp), ""]

    h_main = 20 if 20 in HORIZONS else HORIZONS[0]
    pa = per_asset_table(panel, h_main, score=1)
    pa.to_csv(f"{OUTDIR}/per_asset_score1.csv", index=False, encoding="utf-8-sig")
    md += [f"### 점수 1 · 선행 {h_main}일 · 자산별", "", md_table(pa), ""]

    base = sleeve_backtest(prices, SCALAR_BASE)
    alt = sleeve_backtest(prices, SCALAR_ALT)
    if base and alt:
        keys = [k for k in base if not k.startswith("_")]
        cmp_df = pd.DataFrame({
            "지표": keys,
            "현행(1→50%)": [base[k] for k in keys],
            "변경안(1→0%)": [alt[k] for k in keys],
            "차이": [alt[k] - base[k] for k in keys],
        })
        cmp_df.to_csv(f"{OUTDIR}/counterfactual.csv", index=False,
                      encoding="utf-8-sig")
        md += ["## 반사실 백테스트 (비용 차감)", "", md_table(cmp_df), ""]

        eq = pd.DataFrame({"현행": base["_equity"], "변경안": alt["_equity"]})
        eq.to_csv(f"{OUTDIR}/equity.csv", encoding="utf-8-sig")

    report = "\n".join(md)
    with open(f"{OUTDIR}/report.md", "w", encoding="utf-8") as f:
        f.write(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report)

    print(report)
    send_telegram(build_telegram_summary(panel, base, alt))
    return 0


def build_telegram_summary(panel, base, alt) -> str:
    h = 20 if 20 in HORIZONS else HORIZONS[0]
    tbl = score_table(panel, h)
    lines = [f"<b>🔬 점수별 조건부 수익률 (선행 {h}일)</b>",
             f"<i>{datetime.now(KST).strftime('%Y-%m-%d')} KST</i>", ""]
    for _, r in tbl.iterrows():
        t = r["NW_t"]
        flag = "" if pd.isna(t) else (" ⚠️" if t < -2 else (" ✅" if t > 2 else ""))
        lines.append(f"점수 {int(r['점수'])}: 평균 {r['평균(%)']:+.2f}% · "
                     f"승률 {r['승률(%)']:.0f}% · t={t:.2f}{flag}")
    if base and alt:
        lines += ["", "<b>반사실 (점수1 → 0%)</b>",
                  f"Sharpe {base['Sharpe']:.3f} → {alt['Sharpe']:.3f} "
                  f"({alt['Sharpe'] - base['Sharpe']:+.3f})",
                  f"MDD {base['MDD(%)']:.1f}% → {alt['MDD(%)']:.1f}%",
                  f"CAGR {base['CAGR(%)']:.2f}% → {alt['CAGR(%)']:.2f}%"]
    lines += ["", "<i>사전 기준 5개 전부 충족해야 변경</i>"]
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20)
        if r.status_code != 200:
            print(f"텔레그램 전송 실패 {r.status_code}: {r.text}", file=sys.stderr)
            return False
    except Exception as e:                               # noqa: BLE001
        print(f"텔레그램 전송 오류: {e}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
