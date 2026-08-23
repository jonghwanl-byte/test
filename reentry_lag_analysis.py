"""
reentry_lag_analysis.py

진짜 반등(신저점 재갱신 없음) 구간에서 MA 신호 지연 비용을 정량화한다.

두 가지 지표를 분리해서 계산:
  (A) 놓친 반등폭  = 저점 -> 스칼라 100% 복귀일 사이 QQQ 상승률 (기회비용)
  (B) 라운드트립   = 이탈 체결가 -> 재진입 체결가 (실제 손익)

전제:
  - 신호 판정은 D일 종가, 체결은 D+1 종가 (lag = +1, 룩어헤드 없음)
  - MA 20/120/200, 비대칭 히스테리시스 (진입 +1.5%, 이탈 -2.5%)
  - 포지션 스칼라: 활성 MA 3개=100%, 2개=75%, 1개=50%, 0개=0%

사용:
  pip install yfinance pandas numpy
  python reentry_lag_analysis.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- 파라미터 ----------------
TICKER = "QQQ"
START = "1999-03-10"
MAS = (20, 120, 200)
BAND_ENTRY = 0.015    # +1.5%
BAND_EXIT = -0.025    # -2.5%
SCALAR = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}
EXEC_LAG = 1          # T+1 종가 체결
CORE_WEIGHT = 0.60    # 포트폴리오 내 QQQ 비중

# 진짜 반등만: 저점 이후 RECOVER_WINDOW 안에 신저점이 없어야 함
DD_THRESHOLD = 0.10   # 고점 대비 10% 이상 하락한 국면만 대상
RECOVER_WINDOW = 250  # 거래일


def load_prices() -> pd.Series:
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    px = df["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.dropna()


def build_signals(px: pd.Series) -> pd.DataFrame:
    """히스테리시스 적용 MA 활성 상태 + 포지션 스칼라 (체결 지연 반영)."""
    out = pd.DataFrame(index=px.index)
    out["px"] = px

    active_cols = []
    for w in MAS:
        ma = px.rolling(w).mean()
        upper = ma * (1 + BAND_ENTRY)
        lower = ma * (1 + BAND_EXIT)

        state = np.zeros(len(px), dtype=bool)
        cur = False
        p = px.to_numpy()
        u = upper.to_numpy()
        l = lower.to_numpy()
        for i in range(len(px)):
            if np.isnan(u[i]):
                cur = False
            elif not cur and p[i] > u[i]:
                cur = True          # 진입: MA * 1.015 상향 돌파
            elif cur and p[i] < l[i]:
                cur = False         # 이탈: MA * 0.975 하향 이탈
            state[i] = cur
        col = f"ma{w}"
        out[col] = state
        active_cols.append(col)

    out["n_active"] = out[active_cols].sum(axis=1)
    out["target"] = out["n_active"].map(SCALAR)
    # D일 신호 -> D+1 종가 체결
    out["position"] = out["target"].shift(EXEC_LAG)
    out["fill_px"] = px  # 체결가는 그 날 종가
    return out.dropna()


def find_episodes(sig: pd.DataFrame) -> list:
    """DD_THRESHOLD 이상 하락 후 전고점을 회복한 국면을 추출."""
    px = sig["px"]
    peak = px.cummax()
    dd = px / peak - 1.0

    episodes, i, n = [], 0, len(px)
    while i < n:
        if dd.iloc[i] <= -DD_THRESHOLD:
            # 국면 시작 = 직전 고점
            start = int(np.argmax((px.iloc[: i + 1] == peak.iloc[i]).to_numpy()))
            # 회복 시점 = 전고점 재돌파
            rec = None
            for j in range(i, n):
                if px.iloc[j] >= peak.iloc[start]:
                    rec = j
                    break
            end = rec if rec is not None else n - 1
            trough = int(px.iloc[start:end + 1].idxmin() == px.index)
            trough = px.iloc[start:end + 1].argmin() + start
            episodes.append((start, trough, end))
            i = end + 1
        else:
            i += 1
    return episodes


def is_real_rebound(px: pd.Series, trough: int) -> bool:
    """저점 이후 RECOVER_WINDOW 안에 신저점이 없으면 '진짜 반등'."""
    lo = px.iloc[trough]
    fwd = px.iloc[trough + 1: trough + 1 + RECOVER_WINDOW]
    return len(fwd) > 0 and fwd.min() > lo * 0.995


def analyze(sig: pd.DataFrame) -> pd.DataFrame:
    px = sig["px"]
    pos = sig["position"]
    rows = []

    for start, trough, end in find_episodes(sig):
        if not is_real_rebound(px, trough):
            continue

        # 이탈 체결가: 국면 시작 이후 포지션이 처음 0.75 미만으로 내려간 체결일
        exit_idx = None
        for k in range(start, trough + 1):
            if pos.iloc[k] < 0.75:
                exit_idx = k
                break
        if exit_idx is None:
            continue  # 이탈 자체가 없었으면 지연 비용도 없음

        # 재진입: 저점 이후 각 스칼라 티어에 처음 도달한 체결일
        tiers = {}
        for tier in (0.50, 0.75, 1.00):
            hit = None
            for k in range(trough, min(trough + RECOVER_WINDOW, len(px))):
                if pos.iloc[k] >= tier:
                    hit = k
                    break
            tiers[tier] = hit

        full = tiers[1.00]
        if full is None:
            continue

        lo_px = px.iloc[trough]
        exit_px = px.iloc[exit_idx]
        re_px = px.iloc[full]

        rows.append({
            "고점": px.index[start].date(),
            "저점": px.index[trough].date(),
            "MDD": round(lo_px / px.iloc[start] - 1, 4),
            "이탈체결일": px.index[exit_idx].date(),
            "50%복귀": px.index[tiers[0.50]].date() if tiers[0.50] else None,
            "100%복귀": px.index[full].date(),
            "지연(거래일)": full - trough,
            "놓친반등": round(re_px / lo_px - 1, 4),
            "라운드트립": round(exit_px / re_px - 1, 4),
            "포트영향": round((exit_px / re_px - 1) * CORE_WEIGHT, 4),
        })

    return pd.DataFrame(rows)


# ---------------- 출력 / 알림 ----------------
def env(key: str, default=None):
    """GitHub Actions가 미정의 vars.*를 빈 문자열로 주입하는 문제 대응."""
    import os
    v = os.environ.get(key)
    return default if v is None or v.strip() == "" else v.strip()


def build_report(res: pd.DataFrame) -> str:
    """텔레그램 HTML 리포트 (기존 봇 스타일)."""
    if res.empty:
        return "⚠️ <b>재진입 지연 분석</b>\n분석 대상 구간이 없습니다."

    loss = res[res["라운드트립"] < 0]
    gain = res[res["라운드트립"] >= 0]

    lines = [
        f"📉 <b>{TICKER} 재진입 지연 분석</b>",
        f"<i>T+1 종가 체결 · MA {'/'.join(map(str, MAS))} · 밴드 +1.5/−2.5%</i>",
        "",
        f"• 진짜 반등 구간 · <b>{len(res)}</b>개",
        f"• 손실 구간 · <b>{len(loss)}</b>개 / 이익 구간 · <b>{len(gain)}</b>개",
        f"• 평균 재진입 지연 · <b>{res['지연(거래일)'].mean():.1f}</b>거래일",
        f"• 손실 합계(포트) · <b>{loss['포트영향'].sum():+.2%}</b>",
        f"• 순합계(포트) · <b>{res['포트영향'].sum():+.2%}</b>",
        "",
        "<b>구간별</b>",
    ]
    for _, r in res.iterrows():
        mark = "🔴" if r["라운드트립"] < 0 else "🟢"
        lines.append(
            f"{mark} {r['저점']} · MDD {r['MDD']:+.1%} · "
            f"지연 {r['지연(거래일)']}일 · 놓친반등 {r['놓친반등']:+.1%} · "
            f"<b>왕복 {r['라운드트립']:+.1%}</b>"
        )
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token = env("TELEGRAM_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[skip] 텔레그램 환경변수 없음")
        return
    import urllib.parse
    import urllib.request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    with urllib.request.urlopen(url, data=data, timeout=20) as resp:
        print(f"[telegram] {resp.status}")


if __name__ == "__main__":
    px = load_prices()
    sig = build_signals(px)
    res = analyze(sig)

    pd.set_option("display.width", 200)
    print(f"\n=== {TICKER} 진짜 반등 구간 신호 지연 분석 (T+1 체결) ===\n")
    print(res.to_string(index=False) if len(res) else "(해당 구간 없음)")

    if len(res):
        loss = res[res["라운드트립"] < 0]
        print(f"\n손실 발생 구간: {len(loss)} / {len(res)}")
        print(f"손실 합계(포트 기준): {loss['포트영향'].sum():.2%}")
        print(f"전체 합계(포트 기준): {res['포트영향'].sum():.2%}")
        print(f"평균 재진입 지연: {res['지연(거래일)'].mean():.1f} 거래일")

    # CSV 저장 (GitHub Actions 아티팩트용)
    out = env("OUTPUT_CSV", "reentry_lag_result.csv")
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {out}")

    send_telegram(build_report(res))
