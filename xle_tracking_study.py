"""
xle_tracking_study.py
---------------------
KODEX 미국S&P500에너지(합성) 218420 실측 진단

목적:
  1) 추적차이(Tracking Difference) 실측 - 총보수 0.25% 대비 실효비용 확인
     - 1차 지표: NAV vs 기초지수  (순수 운용 비용 + 스왑 스프레드)
     - 2차 지표: 종가 vs XLE 원화환산 TR 프록시 (백테스트 프록시 타당성)
  2) 괴리율(시장가 vs NAV) 분포 - LP 품질
  3) 유동성 - 거래대금 분포, 목표 주문금액 대비 참여율

출력: 콘솔 리포트 + CSV (선택)
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def env(key: str, default):
    """GitHub Actions vars.* 는 미설정 시 '빈 문자열'을 주입한다.
    os.environ.get()의 기본값 폴백이 동작하지 않으므로 명시적으로 처리."""
    v = os.environ.get(key, "")
    if v is None or str(v).strip() == "":
        return default
    return type(default)(v) if not isinstance(default, bool) else \
        str(v).strip().lower() in ("1", "true", "yes")


# ==============================================================
# CONFIG
# ==============================================================
TICKER_KRX = env("TICKER_KRX", "218420")   # KODEX 미국S&P500에너지(합성)
TICKER_US = env("TICKER_US", "XLE")        # 장기 프록시
TICKER_FX = "USDKRW=X"

START = env("START", "2015-05-04")         # 상장(2015-04-28) 직후부터
END = env("END", datetime.today().strftime("%Y-%m-%d"))

LAG_DAYS = env("LAG_DAYS", 1)              # 미국 종가 -> 익영업일 KRX 반영
TRADING_DAYS = 252

ORDER_KRW = env("ORDER_KRW", 10_000_000)   # 스코어3 진입 시 예상 주문금액
PARTICIPATION_WARN = 0.05                  # 일거래대금 대비 5% 초과 시 경고

SAVE_CSV = True
CSV_PATH = "xle_tracking_daily.csv"

TG_TOKEN = env("TELEGRAM_TOKEN", "")
TG_CHAT = env("TELEGRAM_CHAT_ID", "")

# ==============================================================
# FETCH
# ==============================================================
def fetch_naver(ticker: str, start: str, end: str) -> pd.DataFrame:
    """네이버 금융 siseJson — 로그인 불필요. OHLCV만 제공(NAV 없음).

    KRX 정보데이터시스템이 2025-12-27 회원제로 전환되어 pykrx는
    KRX_ID/KRX_PW 없이는 동작하지 않는다. 이쪽이 1차 소스.
    """
    import json
    import re
    import requests

    url = (
        "https://api.finance.naver.com/siseJson.naver"
        f"?symbol={ticker}&requestType=1"
        f"&startTime={start.replace('-', '')}"
        f"&endTime={end.replace('-', '')}&timeframe=day"
    )
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.naver.com/",
    })
    r.raise_for_status()

    # 응답이 JS 리터럴(작은따옴표, 트레일링 콤마) 형태라 정규화 필요
    txt = r.text.strip().replace("'", '"')
    txt = re.sub(r",\s*]", "]", txt)
    rows = json.loads(txt)
    if len(rows) < 2:
        raise RuntimeError(f"네이버 데이터 없음: {ticker}")

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={
        "날짜": "date", "시가": "시가", "고가": "고가",
        "저가": "저가", "종가": "종가", "거래량": "거래량",
    })
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for c in ["시가", "고가", "저가", "종가", "거래량"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["종가"] > 0]
    # 네이버는 거래대금을 주지 않으므로 종가×거래량으로 근사
    df["거래대금"] = df["종가"] * df["거래량"]
    return df


def fetch_pykrx(ticker: str, start: str, end: str):
    """pykrx 경로. satellite 와 동일하게 getter 를 두 개 시도한다.

    KRX 정보데이터시스템이 2025-12-27 회원제로 전환되면서
    get_etf_ohlcv_by_date 는 KRX_ID/KRX_PW 없이 'isin' 에러로 실패한다.
    반면 get_market_ohlcv_by_date 는 인증 없이 동작하므로 시세는 확보된다.
    NAV·기초지수는 ETF 엔드포인트에서만 나오므로 인증이 있을 때만 채워진다.

    반환: (시세 DataFrame, NAV DataFrame or None)
    """
    from pykrx import stock

    f, t = start.replace("-", ""), end.replace("-", "")
    px, nav = None, None

    for getter in (stock.get_etf_ohlcv_by_date, stock.get_market_ohlcv_by_date):
        try:
            df = getter(f, t, ticker)
        except Exception:                              # noqa: BLE001
            continue
        if df is None or len(df) == 0 or "종가" not in df.columns:
            continue

        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df["종가"] > 0]

        if "NAV" in df.columns:                        # ETF 엔드포인트 성공
            nav = df[[c for c in ("NAV", "기초지수") if c in df.columns]]
        if px is None or len(df) > len(px):
            px = df
        if nav is not None:
            break

    return px, nav


def fetch_krx(ticker: str, start: str, end: str) -> pd.DataFrame:
    """pykrx 우선, 실패 시 네이버 폴백."""
    px, nav = None, None
    try:
        px, nav = fetch_pykrx(ticker, start, end)
        if px is not None:
            print(f"  [pykrx] {len(px):,}행"
                  f"{' (NAV 포함)' if nav is not None else ' (NAV 없음 — KRX 인증 필요)'}")
    except Exception as e:                             # noqa: BLE001
        print(f"  [pykrx] 실패: {e}")

    if px is None:
        px = fetch_naver(ticker, start, end)
        print(f"  [naver] {len(px):,}행 (폴백)")

    if nav is not None:
        for c in nav.columns:
            px[c] = nav[c].reindex(px.index)

    if "거래대금" not in px.columns:
        px["거래대금"] = px["종가"] * px["거래량"]
    return px


def fetch_us_krw(ticker: str, fx: str, start: str, end: str) -> pd.DataFrame:
    """yfinance: XLE 총수익(TR) / 가격(PR) 종가 + USDKRW

    TR = auto_adjust=True (배당 재투자)
    PR = auto_adjust=False 의 Close (배당 제외)
    두 계열을 모두 받아 ETF가 어느 쪽을 추종하는지 판별한다.
    """
    import yfinance as yf

    tr = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    pr = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False)
    fxd = yf.download(fx, start=start, end=end,
                      auto_adjust=True, progress=False)

    if tr.empty or fxd.empty:
        raise RuntimeError("yfinance 데이터 없음")

    def close_col(d, field="Close"):
        c = d[field]
        return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c

    out = pd.DataFrame({
        "xle_tr_usd": close_col(tr),
        "xle_pr_usd": close_col(pr) if not pr.empty else close_col(tr),
        "usdkrw": close_col(fxd),
    })
    idx = pd.to_datetime(out.index)
    out.index = idx.tz_localize(None) if idx.tz is not None else idx
    return out.ffill().dropna()


def align_proxy(krx_index, us: pd.DataFrame, col: str,
                eq_lag: int, fx_lag: int) -> pd.Series:
    """KRX 거래일 축에 미국 시계열을 정렬한다.

    KRX 종가는 '전일 미국 종가 × 당일 한국시간 환율'을 반영하므로
    주가 시차(eq_lag)와 환율 시차(fx_lag)가 서로 다르다. 둘을 분리해
    각각 밀어 본 뒤 상관이 최대가 되는 조합을 채택한다.
    """
    full = krx_index.union(us.index)
    eq = us[col].reindex(full).ffill().reindex(krx_index).shift(eq_lag)
    fx = us["usdkrw"].reindex(full).ffill().reindex(krx_index).shift(fx_lag)
    return eq * fx


# ==============================================================
# COMPUTE
# ==============================================================
def ann_diff(a: pd.Series, b: pd.Series) -> float:
    """두 시계열의 연환산 수익률 차이 (a - b), %p"""
    a, b = a.dropna(), b.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < 30:
        return np.nan
    a, b = a.loc[idx], b.loc[idx]
    yrs = (idx[-1] - idx[0]).days / 365.25
    ca = (a.iloc[-1] / a.iloc[0]) ** (1 / yrs) - 1
    cb = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    return (ca - cb) * 100


def lag_grid(df: pd.DataFrame, us: pd.DataFrame, col: str, max_lag: int = 2):
    """(eq_lag, fx_lag) 조합별 일수익률 상관을 표로 만든다."""
    r_close = df["close"].pct_change()
    rows = []
    for eq in range(max_lag + 1):
        for fx in range(max_lag + 1):
            proxy = align_proxy(df.index, us, col, eq, fx)
            c = r_close.corr(proxy.pct_change())
            rows.append((eq, fx, c))
    return sorted(rows, key=lambda x: -(x[2] if pd.notna(x[2]) else -9))


def build(krx: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=krx.index)
    df["close"] = krx["종가"]
    df["volume"] = krx.get("거래량", np.nan)
    df["value"] = krx.get("거래대금", np.nan)

    if "NAV" in krx.columns:
        df["nav"] = krx["NAV"].replace(0, np.nan)
    if "기초지수" in krx.columns:
        df["bm_index"] = krx["기초지수"].replace(0, np.nan)

    # 괴리율 = (시장가 - NAV) / NAV
    if "nav" in df:
        df["disparity_pct"] = (df["close"] / df["nav"] - 1) * 100

    # 주가/환율 시차를 분리해 상관 최대 조합을 찾는다
    grid = lag_grid(df, us, "xle_tr_usd")
    eq_lag, fx_lag, best_corr = grid[0]
    df.attrs["grid"] = grid
    df.attrs["lags"] = (eq_lag, fx_lag, best_corr)

    df["xle_tr"] = align_proxy(df.index, us, "xle_tr_usd", eq_lag, fx_lag)
    df["xle_pr"] = align_proxy(df.index, us, "xle_pr_usd", eq_lag, fx_lag)

    for c in ["close", "nav", "bm_index", "xle_tr", "xle_pr"]:
        if c in df:
            df[f"r_{c}"] = df[c].pct_change()
    return df


def report(df: pd.DataFrame) -> None:
    line = "=" * 62
    eq_lag, fx_lag, best_corr = df.attrs["lags"]
    print(line)
    print(f" 218420 추적차이 진단  |  {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}")
    print(f" 관측치 {len(df):,}일  |  채택 시차: 주가 {eq_lag}일 / 환율 {fx_lag}일")
    print(line)

    print("\n[0] 시차 조합별 일수익률 상관 (상위 4개)")
    for eq, fx, c in df.attrs["grid"][:4]:
        star = " ←채택" if (eq, fx) == (eq_lag, fx_lag) else ""
        print(f"    주가 {eq}일 / 환율 {fx}일 : {c:.4f}{star}")
    if best_corr < 0.90:
        print("  ⚠ 최대 상관도 0.90 미만 — 프록시 자체가 부적합할 수 있음")

    # --- 1차: NAV vs 기초지수 -----------------------------------
    print("\n[1] 추적차이 — NAV vs 기초지수  (핵심 지표)")
    if "nav" in df and "bm_index" in df:
        td = ann_diff(df["nav"], df["bm_index"])
        te = (df["r_nav"] - df["r_bm_index"]).std() * np.sqrt(TRADING_DAYS) * 100
        print(f"  연환산 추적차이 : {td:+.3f} %p   (총보수 0.25% 기준선)")
        print(f"  추적오차(TE)    : {te:.3f} %")
        implied = -td
        print(f"  → 실효비용 추정 : {implied:.3f} %/년"
              f"  {'⚠ 보수 초과' if implied > 0.35 else '✓ 보수 수준'}")

        yr = pd.DataFrame({
            "nav": df["nav"], "bm": df["bm_index"]}).dropna()
        rows = []
        for y, g in yr.groupby(yr.index.year):
            if len(g) < 60:
                continue
            rows.append((y, (g["nav"].iloc[-1] / g["nav"].iloc[0] -
                             g["bm"].iloc[-1] / g["bm"].iloc[0]) * 100))
        print("\n  연도별 추적차이(%p):")
        for y, v in rows:
            print(f"    {y}  {v:+.3f}")
    else:
        print("  NAV/기초지수 컬럼 없음 — KRX 인증(KRX_ID/KRX_PW) 필요")


    # --- 2차: 종가 vs XLE 프록시 --------------------------------
    print("\n[2] 백테스트 프록시 타당성 — 종가 vs XLE 원화환산 TR")
    sub = df[["close", "xle_tr", "xle_pr", "r_close",
              "r_xle_tr", "r_xle_pr"]].dropna()
    if len(sub) > 60:
        td_tr = ann_diff(sub["close"], sub["xle_tr"])
        td_pr = ann_diff(sub["close"], sub["xle_pr"])
        corr = sub["r_close"].corr(sub["r_xle_tr"])
        te2 = (sub["r_close"] - sub["r_xle_tr"]).std() * np.sqrt(TRADING_DAYS) * 100

        print(f"  vs XLE 총수익(TR) : {td_tr:+.3f} %p/년")
        print(f"  vs XLE 가격(PR)   : {td_pr:+.3f} %p/년")
        print(f"  일수익률 상관     : {corr:.4f}")
        print(f"  추적오차(TE)      : {te2:.3f} %")

        # 배당 누락 판별: PR 기준 격차가 작으면 ETF는 가격지수만 추종
        div_yield = td_pr - td_tr          # TR-PR 격차 = 배당 기여
        print(f"\n  XLE 배당 기여 추정: {div_yield:+.2f} %p/년")
        if abs(td_pr) < 1.0 and td_tr < -2.0:
            print("  → 판정: ETF가 가격지수(PR)를 추종. 배당이 소멸하고 있음")
            print("     장기 백테스트는 XLE '가격수익률'로 돌려야 함")
            cost = max(0.0, -td_pr)
        elif td_tr < -2.0:
            print("  → 판정: PR 기준으로도 격차가 큼. 스왑 비용 과다 의심")
            cost = max(0.0, -td_tr)
        else:
            print("  → 판정: 총수익(TR) 정상 추종")
            cost = max(0.0, -td_tr)
        print(f"  → 백테스트 연비용 가정: {cost:.2f} %/년")

        print("\n  연도별 격차 (vs TR / vs PR, %p):")
        g = sub[["close", "xle_tr", "xle_pr"]]
        for y, blk in g.groupby(g.index.year):
            if len(blk) < 60:
                continue
            r_c = blk["close"].iloc[-1] / blk["close"].iloc[0]
            a = (r_c - blk["xle_tr"].iloc[-1] / blk["xle_tr"].iloc[0]) * 100
            b = (r_c - blk["xle_pr"].iloc[-1] / blk["xle_pr"].iloc[0]) * 100
            print(f"    {y}  {a:+7.2f}  /  {b:+7.2f}")

    # --- 3차: 괴리율 --------------------------------------------
    print("\n[3] 괴리율 분포 (시장가 vs NAV)")
    if "disparity_pct" in df:
        d = df["disparity_pct"].dropna()
        r = d.tail(TRADING_DAYS)
        print(f"  전체  평균 {d.mean():+.3f}%  표준편차 {d.std():.3f}%"
              f"  |{d.abs().quantile(0.95):.3f}%| (95p)")
        print(f"  최근1년 평균 {r.mean():+.3f}%  |{r.abs().quantile(0.95):.3f}%| (95p)")
        bad = (d.abs() > 0.5).mean() * 100
        print(f"  |괴리율|>0.5% 발생빈도: {bad:.2f}%"
              f"  {'✓ 양호' if bad < 2 else '⚠ LP 품질 점검'}")

    # --- 4차: 유동성 --------------------------------------------
    print("\n[4] 유동성 / 체결 가능성")
    v = df["value"].dropna()
    if len(v):
        r = v.tail(TRADING_DAYS)
        med = r.median()
        p10 = r.quantile(0.10)
        print(f"  최근1년 일거래대금  중앙값 {med/1e8:,.2f}억  "
              f"10분위 {p10/1e8:,.2f}억  최소 {r.min()/1e8:,.2f}억")
        print(f"  목표 주문금액 {ORDER_KRW/1e8:,.2f}억 기준 참여율:")
        print(f"    중앙일 {ORDER_KRW/med*100:5.2f}%   "
              f"한산한날(10분위) {ORDER_KRW/p10*100:5.2f}%   "
              f"최악일 {ORDER_KRW/r.min()*100:5.2f}%")
        worst = ORDER_KRW / p10
        print(f"  → {'✓ 슬리피지 무시 가능' if worst < PARTICIPATION_WARN else '⚠ 분할주문 필요'}")

    # --- 종합 ----------------------------------------------------
    print("\n" + line)
    print(" 백테스트 비용 가정 체크리스트")
    print("   - 연 보유비용 : [2]의 값 사용 (총보수 0.25%가 아님)")
    print("   - 왕복 거래비용: 괴리율 95p × 2 + 매매수수료")
    print("   - 프록시 상관 0.90 미만이면 LAG_DAYS=0/1/2 재실행")
    print(line)


# ==============================================================
# MAIN
# ==============================================================
def send_telegram(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    import requests
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):          # 4096자 제한 대비 청크 전송
        chunk = text[i:i + 3500]
        try:
            requests.post(url, data={
                "chat_id": TG_CHAT,
                "text": f"<pre>{chunk}</pre>",
                "parse_mode": "HTML",
            }, timeout=20)
        except Exception as e:
            print(f"[WARN] telegram 전송 실패: {e}", file=sys.stderr)


def main():
    import io
    import contextlib

    print("데이터 수집 중...")
    krx = fetch_krx(TICKER_KRX, START, END)
    print(f"  KRX  {len(krx):,}행  컬럼: {list(krx.columns)}")
    us = fetch_us_krw(TICKER_US, TICKER_FX, START, END)
    print(f"  US   {len(us):,}행")

    df = build(krx, us)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(df)
    text = buf.getvalue()
    print(text)                                   # Actions 로그
    send_telegram(text)                           # 텔레그램

    if SAVE_CSV:
        df.to_csv(CSV_PATH, encoding="utf-8-sig")
        print(f"\n저장: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
