#!/usr/bin/env python3
"""
이평선 히스테리시스 전략 백테스트 엔진 v2
 - 60일선 제외 (20/120/200 3개 MA)
 - 거래량 확인 필터 (평균거래량 대비 배수 미달 시 신규 진입 보류)
 - 레짐 필터 (장기 MA 기울기가 하락이면 신규 진입 차단, 청산은 그대로 허용)

기존 backtest.py와 별개로 동작하며, 캐시 파일도 분리되어 있어(.bt_cache_v2.pkl)
기존 캐시를 건드리지 않습니다. Volume 데이터가 추가로 필요해 재다운로드합니다.

사용법:
  python backtest_v2.py                    # 필터 전부 적용한 기본 실행
  python backtest_v2.py --compare          # 필터 on/off 4조합 비교 (핵심 기능)
  python backtest_v2.py --no-volume-filter # 거래량 필터만 끄기
  python backtest_v2.py --no-regime-filter # 레짐 필터만 끄기
  python backtest_v2.py --years 10 --no-cache

설계 원칙 (backtest.py와 동일)
---------------------------------
1) 룩어헤드 차단: T일 종가/거래량으로 계산한 상태는 T+1일부터 적용 (shift).
2) 거래비용 반영: 비중이 바뀐 만큼 편도 수수료+슬리피지 차감.
3) 현금 이자 반영: 미보유분은 연 CASH_RATE로 굴러간다.
4) 신규 진입만 필터링: 거래량/레짐 조건은 '켜지는' 전환에만 적용하고,
   '꺼지는'(청산) 전환은 그대로 둔다 — 손절/이익실현을 막지 않기 위함.
"""

import argparse
import os
import pickle
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ===== 설정 ============================================================
TICKERS = {
    "NVDA": "엔비디아", "AAPL": "애플", "GOOGL": "구글", "MSFT": "마이크로소프트",
    "MU": "마이크론", "AMZN": "아마존", "AMD": "AMD", "AVGO": "브로드컴",
    "META": "메타", "TSLA": "테슬라", "MRVL": "마벨테크놀로지",
    "PLTR": "팔란티어", "GEV": "GE버노바", "ETN": "이튼", "LEU": "센트러스에너지",
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "012330.KS": "현대모비스",
    "009150.KS": "삼성전기", "017670.KS": "SK텔레콤",
    "079550.KS": "LIG디펜스앤에어로스페이스", "012450.KS": "한화에어로스페이스",
    "016360.KS": "삼성증권", "003230.KS": "삼양식품",
}

MA_PERIODS = [20, 120, 200]     # 60일선 제외
SCALAR_MAP = {3: 1.00, 2: 0.66, 1: 0.33, 0: 0.00}

BAND_UP = 1.03
BAND_DN = 0.98
CONFIRM_DIRECTION = True

VOLUME_FILTER = True
VOLUME_MULT = 1.5               # 20일 평균거래량 대비 배수
VOLUME_LOOKBACK = 20

REGIME_FILTER = True
REGIME_LOOKBACK = 20            # N거래일 전 대비 장기MA 상승/하락 비교

CASH_RATE = 0.02
COST_BPS = 15
EXEC_LAG = 1

CACHE = ".bt_cache_v2.pkl"
TRADING_DAYS = 252
# =======================================================================


# ---------- 데이터 (Close + Volume) -------------------------------------
def load_prices(years: int, use_cache: bool = True):
    key = f"{years}y-v2"
    if use_cache and os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                print(f"캐시 사용 ({len(blob['close'])}종목). 새로 받으려면 --no-cache")
                return blob["close"], blob["volume"]
        except Exception:
            pass

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance pandas numpy")

    us = [t for t in TICKERS if not t.endswith((".KS", ".KQ"))]
    kr = [t for t in TICKERS if t.endswith((".KS", ".KQ"))]
    start = (date.today() - timedelta(days=int(years * 365.25) + 15)).isoformat()

    close_out, vol_out = {}, {}
    for group, label in ((us, "US"), (kr, "KR")):
        print(f"... {label} {len(group)}종목 다운로드 ({start} ~)")
        try:
            df = yf.download(group, start=start, interval="1d",
                             auto_adjust=True, progress=False, threads=False,
                             group_by="column")
        except Exception as e:
            print(f"    !! {label} 다운로드 실패: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            print(f"    !! {label} 응답이 비었습니다 (Yahoo 차단 가능성)", file=sys.stderr)
            continue

        close, vol = df["Close"], df["Volume"]
        if isinstance(close, pd.Series):
            close_out[group[0]] = close.dropna()
            vol_out[group[0]] = vol.dropna()
        else:
            for t in group:
                if t in close.columns:
                    s = close[t].dropna()
                    v = vol[t].reindex(s.index) if t in vol.columns else None
                    if len(s) > 0:
                        close_out[t] = s
                        vol_out[t] = v if v is not None else pd.Series(np.nan, index=s.index)
        print(f"    {label} 수신 {len([t for t in group if t in close_out])}/{len(group)}종목")

    if not close_out:
        sys.exit("가격 데이터를 하나도 받지 못했습니다.")

    with open(CACHE, "wb") as f:
        pickle.dump({"key": key, "close": close_out, "volume": vol_out}, f)
    return close_out, vol_out


# ---------- 상태 머신 (거래량/레짐 필터 포함) ----------------------------
def on_count_series(close: pd.Series, volume: pd.Series,
                     band_up: float, band_dn: float, confirm: bool = True,
                     volume_filter: bool = True, volume_mult: float = VOLUME_MULT,
                     volume_lookback: int = VOLUME_LOOKBACK,
                     regime_filter: bool = True, regime_lookback: int = REGIME_LOOKBACK
                     ) -> pd.Series:
    """각 날짜별 ON된 이평선 개수. 거래량/레짐 조건은 신규 진입(0->1)에만 적용."""
    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    vals = close.values
    ma_vals = {n: mas[n].values for n in MA_PERIODS}

    # 거래량 확인: 당일 거래량이 N일 평균거래량의 mult배 이상인가
    if volume is not None:
        vol_avg = volume.rolling(volume_lookback).mean()
        vol_vals = volume.reindex(close.index).values
        vol_avg_vals = vol_avg.reindex(close.index).values
    else:
        vol_vals = np.full(len(close), np.nan)
        vol_avg_vals = np.full(len(close), np.nan)

    # 레짐: 가장 긴 MA(200일)가 N거래일 전보다 높은가 (상승 레짐)
    regime_ma = close.rolling(max(MA_PERIODS)).mean()
    regime_vals = regime_ma.values
    regime_prev_vals = regime_ma.shift(regime_lookback).values

    state = {n: 0 for n in MA_PERIODS}
    counts = np.full(len(close), np.nan)
    start = max(MA_PERIODS)

    for i in range(start, len(close)):
        p, pp = vals[i], vals[i - 1]
        rising, falling = p > pp, p < pp

        vol_ok = True
        if volume_filter:
            vv, va = vol_vals[i], vol_avg_vals[i]
            if not (np.isnan(vv) or np.isnan(va)):
                vol_ok = vv >= va * volume_mult
            # 거래량 데이터가 없으면(np.nan) 필터를 적용하지 않음(보수적으로 막지 않음)

        regime_ok = True
        if regime_filter:
            rv, rp = regime_vals[i], regime_prev_vals[i]
            if not (np.isnan(rv) or np.isnan(rp)):
                regime_ok = rv > rp
            # 레짐 판단 불가 구간(워밍업)은 막지 않음

        for n in MA_PERIODS:
            ma = ma_vals[n][i]
            if np.isnan(ma):
                state[n] = 0
                continue
            if state[n] == 1:
                # 청산: 필터 없이 그대로. 손절/이익실현을 막지 않는다.
                if p < ma * band_dn and (falling or not confirm):
                    state[n] = 0
            else:
                # 신규 진입: 가격조건 + 거래량조건 + 레짐조건 모두 충족해야 함
                if (p > ma * band_up and (rising or not confirm)
                        and vol_ok and regime_ok):
                    state[n] = 1
        counts[i] = sum(state.values())

    return pd.Series(counts, index=close.index)


# ---------- 성과 계산 (backtest.py와 동일) -------------------------------
def simulate(close, volume, band_up, band_dn, confirm, volume_filter,
             volume_mult, volume_lookback, regime_filter, regime_lookback):
    cnt = on_count_series(close, volume, band_up, band_dn, confirm,
                           volume_filter, volume_mult, volume_lookback,
                           regime_filter, regime_lookback)
    weight = cnt.map(SCALAR_MAP).shift(EXEC_LAG)

    asset_ret = close.pct_change()
    cash_daily = (1 + CASH_RATE) ** (1 / TRADING_DAYS) - 1
    turnover = weight.diff().abs().fillna(0)
    cost = turnover * (COST_BPS / 10000)

    strat_ret = weight * asset_ret + (1 - weight) * cash_daily - cost
    valid = weight.notna() & asset_ret.notna()
    return strat_ret[valid], asset_ret[valid], weight[valid]


def metrics(ret: pd.Series, weight: pd.Series = None) -> dict:
    if len(ret) < 2:
        return {}
    eq = (1 + ret).cumprod()
    yrs = len(ret) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    excess = ret - ((1 + CASH_RATE) ** (1 / TRADING_DAYS) - 1)
    sharpe = excess.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else 0
    out = {"CAGR": cagr, "MDD": dd.min(), "Sharpe": sharpe,
           "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else 0}
    if weight is not None:
        out["평균비중"] = weight.mean()
        out["매매횟수"] = int((weight.diff().abs() > 0.001).sum())
    return out


# ---------- 단일 구성 실행 ------------------------------------------------
def run_one(close_data, vol_data, band_up, band_dn, confirm,
            volume_filter, volume_mult, volume_lookback,
            regime_filter, regime_lookback, verbose=True, label=""):
    rows, strat_rets, bh_rets = [], {}, {}
    for t, name in TICKERS.items():
        close = close_data.get(t)
        volume = vol_data.get(t)
        if close is None or len(close) < max(MA_PERIODS) + 60:
            continue
        sr, ar, w = simulate(close, volume, band_up, band_dn, confirm,
                              volume_filter, volume_mult, volume_lookback,
                              regime_filter, regime_lookback)
        if len(sr) < TRADING_DAYS // 2:
            continue
        strat_rets[t], bh_rets[t] = sr, ar
        m = metrics(sr, w)
        rows.append({"종목": name, "티커": t, "CAGR": m["CAGR"], "MDD": m["MDD"],
                     "Sharpe": m["Sharpe"], "평균비중": m["평균비중"],
                     "매매횟수": m["매매횟수"]})

    df = pd.DataFrame(rows)
    ps = pd.DataFrame(strat_rets).mean(axis=1).dropna()
    pb = pd.DataFrame(bh_rets).mean(axis=1).dropna()
    port_s, port_b = metrics(ps), metrics(pb)

    if verbose and len(df):
        show = df.copy()
        for c in ["CAGR", "MDD", "평균비중"]:
            show[c] = (show[c] * 100).round(1)
        show["Sharpe"] = show["Sharpe"].round(2)
        print(f"\n{'=' * 78}\n  종목별 성과  {label}\n{'=' * 78}")
        print(show.to_string(index=False))
        print(f"\n  [동일가중 포트폴리오]  CAGR {port_s['CAGR']*100:.2f}%  "
              f"MDD {port_s['MDD']*100:.2f}%  Sharpe {port_s['Sharpe']:.3f}  "
              f"(단순보유 CAGR {port_b['CAGR']*100:.2f}% / MDD {port_b['MDD']*100:.2f}%)")

    return df, port_s, port_b


# ---------- 필터 on/off 비교 ---------------------------------------------
def compare(close_data, vol_data, band_up, band_dn, confirm,
            volume_mult, volume_lookback, regime_lookback):
    combos = [
        ("필터없음 (60일선만 제외)", False, False),
        ("거래량 필터만", True, False),
        ("레짐 필터만", False, True),
        ("거래량+레짐 필터", True, True),
    ]
    results = []
    for label, vf, rf in combos:
        print(f"\n>>> 실행 중: {label}")
        _, port_s, port_b = run_one(close_data, vol_data, band_up, band_dn, confirm,
                                     vf, volume_mult, volume_lookback,
                                     rf, regime_lookback, verbose=False)
        results.append({
            "구성": label,
            "전략CAGR": port_s["CAGR"] * 100, "전략MDD": port_s["MDD"] * 100,
            "전략Sharpe": port_s["Sharpe"], "평균매매횟수(전체합)": None,
        })

    df = pd.DataFrame(results)
    for c in ["전략CAGR", "전략MDD"]:
        df[c] = df[c].round(2)
    df["전략Sharpe"] = df["전략Sharpe"].round(3)
    df = df.drop(columns=["평균매매횟수(전체합)"])

    print("\n" + "=" * 78)
    print("  필터 조합 비교 (동일가중 포트폴리오, 25종목)")
    print("=" * 78)
    print(df.to_string(index=False))
    baseline_bh = None
    print("\n  단순보유(개별종목 평균) 대비, 필터를 추가할수록 Sharpe/MDD가")
    print("  개선되는지 확인하세요. CAGR만 오르고 Sharpe가 같이 안 오르면")
    print("  거래 횟수 감소 효과일 뿐 신호 품질 개선은 아닙니다.")
    df.to_csv("backtest_v2_compare.csv", index=False, encoding="utf-8-sig")
    print("\n비교 결과 -> backtest_v2_compare.csv")
    return df


# ---------- GitHub Actions 연동 -----------------------------------------
class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.buf = []

    def write(self, s):
        self.stream.write(s)
        self.buf.append(s)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.buf)


def write_step_summary(title: str, body: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"## {title}\n\n```\n{body}\n```\n")
    print("\n(Actions 요약 페이지에 결과를 기록했습니다)")


# ---------- 진입점 ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--up", type=float, default=BAND_UP)
    ap.add_argument("--dn", type=float, default=BAND_DN)
    ap.add_argument("--no-volume-filter", action="store_true")
    ap.add_argument("--volume-mult", type=float, default=VOLUME_MULT)
    ap.add_argument("--volume-lookback", type=int, default=VOLUME_LOOKBACK)
    ap.add_argument("--no-regime-filter", action="store_true")
    ap.add_argument("--regime-lookback", type=int, default=REGIME_LOOKBACK)
    ap.add_argument("--compare", action="store_true",
                     help="필터 on/off 4조합을 비교")
    args = ap.parse_args()

    confirm = not args.no_confirm
    vf = not args.no_volume_filter
    rf = not args.no_regime_filter

    tee = Tee(sys.stdout)
    real_stdout = sys.stdout
    sys.stdout = tee
    try:
        close_data, vol_data = load_prices(args.years, use_cache=not args.no_cache)
        print(f"수신 완료: {len(close_data)}종목  |  기간 {args.years}년  |  "
              f"구성: 3개 MA(20/120/200)  |  거래량필터 {'ON' if vf else 'OFF'}  |  "
              f"레짐필터 {'ON' if rf else 'OFF'}")

        if args.compare:
            compare(close_data, vol_data, args.up, args.dn, confirm,
                    args.volume_mult, args.volume_lookback, args.regime_lookback)
            title = f"필터 비교 ({args.years}년)"
        else:
            df, port_s, port_b = run_one(
                close_data, vol_data, args.up, args.dn, confirm,
                vf, args.volume_mult, args.volume_lookback,
                rf, args.regime_lookback, verbose=True,
                label=f"(거래량필터 {'ON' if vf else 'OFF'} / 레짐필터 {'ON' if rf else 'OFF'})")
            df.to_csv("backtest_v2_result.csv", index=False, encoding="utf-8-sig")
            print("\n종목별 결과 -> backtest_v2_result.csv")
            title = f"백테스트 v2 ({args.years}년)"
    finally:
        sys.stdout = real_stdout

    write_step_summary(title, tee.text())


if __name__ == "__main__":
    main()
