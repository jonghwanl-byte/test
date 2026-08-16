#!/usr/bin/env python3
"""
KRX 금현물 vs 국제금(원화환산) 프리미엄 스위칭 룰 백테스터

전략: 괴리율(프리미엄)이 X% 이상이면 국제금 다리로, Y% 이하이면 국내 현물 다리로 전환.
      그 사이 구간에서는 직전 포지션 유지(히스테리시스).

사용법:
    python gold_premium_backtest.py --template            # 입력 CSV 서식 생성
    python gold_premium_backtest.py data.csv              # 기본 백테스트 + 그리드 서치
    python gold_premium_backtest.py data.csv --x 8 --y 1  # 특정 임계값만 상세 리포트
    python gold_premium_backtest.py data.csv --split 2026-04-18   # 기간 분할 비교
"""

import argparse
import sys
import itertools
import numpy as np
import pandas as pd

TRADING_DAYS = 250

# KRX / 각종 소스에서 흔히 쓰이는 컬럼명 후보
COL_ALIASES = {
    "date": ["date", "일자", "날짜", "기준일자", "거래일"],
    "krx": ["krx", "krx_price", "국내", "국내금", "국내가격", "종가", "krx금현물",
            "금99.99_1kg", "금 99.99_1kg", "국내금시세"],
    "intl": ["intl", "intl_price", "국제", "국제금", "국제가격", "국제금시세",
             "국제금현물", "국제금시세(원)", "국제금가격"],
}


def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("(원)", "")


def load_data(path):
    """CSV를 읽어 date / krx / intl 3개 컬럼으로 정규화한다."""
    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if df is None:
        sys.exit(f"[오류] {path} 인코딩을 읽지 못했습니다.")

    norm_map = {_norm(c): c for c in df.columns}
    resolved = {}
    for key, aliases in COL_ALIASES.items():
        for a in aliases:
            if _norm(a) in norm_map:
                resolved[key] = norm_map[_norm(a)]
                break
    missing = set(COL_ALIASES) - set(resolved)
    if missing:
        sys.exit(
            f"[오류] 컬럼을 찾지 못했습니다: {sorted(missing)}\n"
            f"       파일의 컬럼: {list(df.columns)}\n"
            f"       --template 로 표준 서식을 확인하세요."
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[resolved["date"]].astype(str).str.strip(),
                               format="mixed", dayfirst=False),
        "krx": pd.to_numeric(
            df[resolved["krx"]].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"),
        "intl": pd.to_numeric(
            df[resolved["intl"]].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"),
    })
    out = out.dropna().sort_values("date").reset_index(drop=True)
    if len(out) < 60:
        print(f"[경고] 유효 데이터가 {len(out)}행뿐입니다. 결과 신뢰도가 낮습니다.")
    out["premium"] = (out["krx"] / out["intl"] - 1.0) * 100.0
    return out


def backtest(df, x, y, lag=1, switch_cost_bps=15.0,
             fee_krx_bps=50.0, fee_intl_bps=40.0, init="KRX"):
    """
    x: 이 값 이상이면 국제금으로 (프리미엄 과열)
    y: 이 값 이하이면 국내 현물로 (프리미엄 냉각)
    lag: 신호 발생 후 실제 체결까지 지연 영업일 수 (종가 신호 -> 다음날 체결이면 1)
    switch_cost_bps: 전환 1회당 왕복 거래비용(호가 스프레드 등), bp
    fee_*_bps: 각 다리의 연간 총보수, bp
    """
    prem = df["premium"].to_numpy()
    n = len(df)

    # 신호 생성 (히스테리시스: 중간 구간은 직전 포지션 유지)
    target = np.empty(n, dtype=object)
    state = init
    for i in range(n):
        if prem[i] >= x:
            state = "INTL"
        elif prem[i] <= y:
            state = "KRX"
        target[i] = state

    # 체결 지연 반영
    pos = np.empty(n, dtype=object)
    pos[:lag] = init
    if lag > 0:
        pos[lag:] = target[:-lag]
    else:
        pos = target.copy()

    ret_krx = df["krx"].pct_change().fillna(0.0).to_numpy()
    ret_intl = df["intl"].pct_change().fillna(0.0).to_numpy()

    gross = np.where(pos == "KRX", ret_krx, ret_intl)

    # 보수 차감 (일할)
    fee_daily = np.where(pos == "KRX", fee_krx_bps, fee_intl_bps) / 1e4 / TRADING_DAYS

    # 전환 비용
    switched = np.zeros(n, dtype=bool)
    switched[1:] = pos[1:] != pos[:-1]
    cost = switched * (switch_cost_bps / 1e4)

    net = gross - fee_daily - cost
    equity = np.cumprod(1.0 + net)

    return pd.DataFrame({
        "date": df["date"], "premium": df["premium"],
        "position": pos, "ret": net, "equity": equity,
        "switch": switched,
    })


def metrics(dates, rets, equity, n_switch=None):
    if len(rets) < 2:
        return {}
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    total = equity[-1] - 1.0
    cagr = equity[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(equity)
    mdd = np.min(equity / peak - 1.0)
    m = {
        "총수익률": total, "CAGR": cagr, "연변동성": vol,
        "샤프(rf=0)": cagr / vol if vol > 0 else np.nan, "MDD": mdd,
    }
    if n_switch is not None:
        m["전환횟수"] = n_switch
    return m


def fmt_metrics(name, m):
    sw = f"  전환 {int(m['전환횟수']):>3d}회" if "전환횟수" in m else " " * 11
    return (f"{name:<22s} 총수익 {m['총수익률']:>8.2%}  CAGR {m['CAGR']:>7.2%}  "
            f"변동성 {m['연변동성']:>6.2%}  샤프 {m['샤프(rf=0)']:>5.2f}  "
            f"MDD {m['MDD']:>7.2%}{sw}")


def baselines(df, fee_krx_bps, fee_intl_bps):
    out = {}
    for leg, fee in (("krx", fee_krx_bps), ("intl", fee_intl_bps)):
        r = df[leg].pct_change().fillna(0.0).to_numpy() - fee / 1e4 / TRADING_DAYS
        eq = np.cumprod(1.0 + r)
        label = "매수후보유: 국내현물" if leg == "krx" else "매수후보유: 국제금"
        out[label] = metrics(df["date"], r, eq)
    r_krx = df["krx"].pct_change().fillna(0.0).to_numpy() - fee_krx_bps / 1e4 / TRADING_DAYS
    r_intl = df["intl"].pct_change().fillna(0.0).to_numpy() - fee_intl_bps / 1e4 / TRADING_DAYS
    r5 = 0.5 * r_krx + 0.5 * r_intl
    out["매수후보유: 50/50"] = metrics(df["date"], r5, np.cumprod(1.0 + r5))
    return out


def grid_search(df, xs, ys, **kw):
    rows = []
    for x, y in itertools.product(xs, ys):
        if y >= x:
            continue
        bt = backtest(df, x, y, **kw)
        m = metrics(bt["date"], bt["ret"].to_numpy(),
                    bt["equity"].to_numpy(), int(bt["switch"].sum()))
        rows.append({"x": x, "y": y, **m})
    g = pd.DataFrame(rows)
    # 이웃 평균 CAGR: 과최적화된 뾰족한 봉우리를 걸러내기 위함
    neigh = []
    for _, r in g.iterrows():
        sel = g[(g["x"].between(r["x"] - 1, r["x"] + 1)) &
                (g["y"].between(r["y"] - 1, r["y"] + 1))]
        neigh.append(sel["CAGR"].mean())
    g["이웃평균CAGR"] = neigh
    return g


def make_plot(df, bt, base_df, out_png):
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "Pretendard", "Gulim"):
        if cand in have:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message=".*missing from font.*")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    ax.plot(df["date"], df["premium"], lw=0.9, color="#185FA5")
    ax.axhline(0, color="#888780", lw=0.6)
    ax.fill_between(df["date"], df["premium"], 0,
                    where=df["premium"] > 0, alpha=0.15, color="#D85A30")
    ax.set_ylabel("프리미엄 (%)")
    ax.set_title("KRX 금현물 vs 국제금(원화환산) 괴리율")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(bt["date"], bt["equity"], lw=1.4, color="#0F6E56", label="스위칭 전략")
    for label, eq in base_df.items():
        ax.plot(df["date"], eq, lw=1.0, alpha=0.75, label=label)
    ax.set_ylabel("누적 수익 (배수)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    for a in axes:
        for lbl in (a.get_yticklabels()):
            lbl.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\n차트 저장: {out_png}  (한글이 깨지면 시스템에 한글 폰트를 설치하세요)")


TEMPLATE = """date,krx,intl
2025-01-02,136130,135200
2025-01-03,137450,135900
2025-01-06,139020,136400
"""


def main():
    p = argparse.ArgumentParser(description="KRX 금 프리미엄 스위칭 룰 백테스터")
    p.add_argument("csv", nargs="?", help="date,krx,intl 컬럼을 가진 CSV")
    p.add_argument("--template", action="store_true", help="입력 CSV 서식 출력")
    p.add_argument("--x", type=float, help="국제금 전환 임계값 (%%)")
    p.add_argument("--y", type=float, help="국내현물 전환 임계값 (%%)")
    p.add_argument("--lag", type=int, default=1, help="신호->체결 지연 영업일 (기본 1)")
    p.add_argument("--switch-cost", type=float, default=15.0,
                   help="전환 1회 비용 bp (기본 15 = 0.15%%)")
    p.add_argument("--fee-krx", type=float, default=50.0, help="국내 다리 연보수 bp")
    p.add_argument("--fee-intl", type=float, default=40.0, help="국제 다리 연보수 bp")
    p.add_argument("--split", type=str, help="이 날짜 기준 기간 분할 비교 (YYYY-MM-DD)")
    p.add_argument("--plot", type=str, default="premium_backtest.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.template:
        print(TEMPLATE, end="")
        print("\n# krx  = KRX 금현물 종가 (원/g)")
        print("# intl = 한국거래소 공표 국제 금 원화환산가 (원/g)")
        return
    if not args.csv:
        p.error("CSV 경로가 필요합니다. 서식은 --template 참고.")

    df = load_data(args.csv)
    kw = dict(lag=args.lag, switch_cost_bps=args.switch_cost,
              fee_krx_bps=args.fee_krx, fee_intl_bps=args.fee_intl)

    print(f"\n{'='*96}")
    print(f"데이터: {df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}  ({len(df)} 영업일)")
    pr = df["premium"]
    print(f"프리미엄  평균 {pr.mean():.2f}%  중앙값 {pr.median():.2f}%  "
          f"표준편차 {pr.std():.2f}%p  최소 {pr.min():.2f}%  최대 {pr.max():.2f}%")
    print("분위수  " + "  ".join(f"{q}%: {pr.quantile(q/100):.2f}"
                                 for q in (5, 25, 50, 75, 90, 95, 99)))
    print("=" * 96)

    base = baselines(df, args.fee_krx, args.fee_intl)

    def run_period(sub, header):
        print(f"\n--- {header} ---")
        b = baselines(sub, args.fee_krx, args.fee_intl)
        for k, m in b.items():
            print(fmt_metrics(k, m))
        if args.x is not None and args.y is not None:
            bt = backtest(sub, args.x, args.y, **kw)
            m = metrics(bt["date"], bt["ret"].to_numpy(),
                        bt["equity"].to_numpy(), int(bt["switch"].sum()))
            print(fmt_metrics(f"스위칭 x={args.x} y={args.y}", m))

    if args.split:
        cut = pd.Timestamp(args.split)
        run_period(df[df["date"] < cut].reset_index(drop=True), f"{args.split} 이전")
        run_period(df[df["date"] >= cut].reset_index(drop=True), f"{args.split} 이후")
        print()

    print("\n[벤치마크 · 전체 기간]")
    for k, m in base.items():
        print(fmt_metrics(k, m))

    if args.x is not None and args.y is not None:
        bt = backtest(df, args.x, args.y, **kw)
        m = metrics(bt["date"], bt["ret"].to_numpy(),
                    bt["equity"].to_numpy(), int(bt["switch"].sum()))
        print(fmt_metrics(f"스위칭 x={args.x} y={args.y}", m))
        hold = bt["position"].value_counts()
        print(f"\n보유일 비중: 국내현물 {hold.get('KRX',0)}일 / 국제금 {hold.get('INTL',0)}일")
    else:
        xs = np.round(np.linspace(pr.quantile(0.60), pr.quantile(0.98), 13), 1)
        ys = np.round(np.linspace(pr.quantile(0.02), pr.quantile(0.50), 11), 1)
        g = grid_search(df, xs, ys, **kw)
        if g.empty:
            print("\n[그리드] 유효한 (x, y) 조합이 없습니다.")
            return
        print("\n[그리드 서치] CAGR 상위 10")
        top = g.sort_values("CAGR", ascending=False).head(10)
        print(top[["x", "y", "CAGR", "MDD", "샤프(rf=0)", "전환횟수", "이웃평균CAGR"]]
              .to_string(index=False,
                         formatters={"CAGR": "{:.2%}".format,
                                     "MDD": "{:.2%}".format,
                                     "샤프(rf=0)": "{:.2f}".format,
                                     "이웃평균CAGR": "{:.2%}".format}))
        print("\n[그리드 서치] 이웃평균 CAGR 상위 10  <- 이쪽이 더 믿을 만합니다")
        rob = g.sort_values("이웃평균CAGR", ascending=False).head(10)
        print(rob[["x", "y", "CAGR", "MDD", "샤프(rf=0)", "전환횟수", "이웃평균CAGR"]]
              .to_string(index=False,
                         formatters={"CAGR": "{:.2%}".format,
                                     "MDD": "{:.2%}".format,
                                     "샤프(rf=0)": "{:.2f}".format,
                                     "이웃평균CAGR": "{:.2%}".format}))
        bh = max(base["매수후보유: 국내현물"]["CAGR"], base["매수후보유: 국제금"]["CAGR"])
        beat = (g["CAGR"] > bh).mean()
        print(f"\n전체 {len(g)}개 조합 중 최고 매수후보유(CAGR {bh:.2%})를 이긴 비율: {beat:.1%}")
        print("  -> 이 비율이 낮으면 특정 조합이 이겼더라도 우연일 가능성이 큽니다.")
        best = rob.iloc[0]
        bt = backtest(df, best["x"], best["y"], **kw)
        args.x, args.y = best["x"], best["y"]

    if not args.no_plot:
        base_eq = {}
        for leg, fee, label in (("krx", args.fee_krx, "매수후보유: 국내현물"),
                                ("intl", args.fee_intl, "매수후보유: 국제금")):
            r = df[leg].pct_change().fillna(0.0).to_numpy() - fee / 1e4 / TRADING_DAYS
            base_eq[label] = np.cumprod(1.0 + r)
        make_plot(df, bt, base_eq, args.plot)


if __name__ == "__main__":
    main()
