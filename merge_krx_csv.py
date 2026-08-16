#!/usr/bin/env python3
"""
KRX에서 따로 받은 두 CSV를 백테스터가 먹는 data.csv 로 병합한다.

사용법:
    # 1단계 - 어떤 컬럼이 있는지 먼저 확인
    python merge_krx_csv.py 국내.csv 국제.csv --inspect

    # 2단계 - 가격 컬럼을 지정해서 병합
    python merge_krx_csv.py 국내.csv 국제.csv --krx-col 종가 --intl-col 국내가격
"""

import argparse
import sys
import pandas as pd

DATE_HINTS = ["일자", "날짜", "기준일자", "거래일", "date"]


def read_any(path):
    for enc in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    sys.exit(f"[중지] {path} 인코딩을 읽지 못했습니다.")


def find_date_col(df, path):
    for c in df.columns:
        if any(h in str(c) for h in DATE_HINTS):
            return c
    sys.exit(f"[중지] {path} 에서 날짜 컬럼을 찾지 못했습니다. 컬럼: {list(df.columns)}")


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("krx_csv", help="금시장 일별매매정보 CSV")
    p.add_argument("intl_csv", help="국제금시세 동향 CSV")
    p.add_argument("--krx-col", help="국내 금현물 종가 컬럼명")
    p.add_argument("--intl-col", help="국제금 원화환산가 컬럼명")
    p.add_argument("--inspect", action="store_true", help="컬럼명과 샘플만 보고 종료")
    p.add_argument("-o", "--out", default="data.csv")
    a = p.parse_args()

    dom, itl = read_any(a.krx_csv), read_any(a.intl_csv)

    if a.inspect or not (a.krx_col and a.intl_col):
        for name, df in (("국내 (금시장 일별매매정보)", dom),
                         ("국제 (국제금시세 동향)", itl)):
            print(f"\n=== {name} — {len(df)}행 ===")
            print("컬럼:", list(df.columns))
            print(df.head(3).to_string(index=False))
        print("\n두 파일에서 '원/그램' 단위 가격 컬럼을 하나씩 골라")
        print("--krx-col / --intl-col 로 지정해 다시 실행하세요.")
        print("주의: 국제 쪽은 달러/트로이온스 컬럼이 아니라 '원화환산' 컬럼을 쓰세요.")
        return

    for col, df, label in ((a.krx_col, dom, "--krx-col"), (a.intl_col, itl, "--intl-col")):
        if col not in df.columns:
            sys.exit(f"[중지] {label} '{col}' 없음. 실제 컬럼: {list(df.columns)}")

    d = pd.DataFrame({
        "date": pd.to_datetime(dom[find_date_col(dom, a.krx_csv)].astype(str)
                               .str.replace("/", "-").str.strip(), format="mixed"),
        "krx": to_num(dom[a.krx_col]),
    }).dropna()
    i = pd.DataFrame({
        "date": pd.to_datetime(itl[find_date_col(itl, a.intl_csv)].astype(str)
                               .str.replace("/", "-").str.strip(), format="mixed"),
        "intl": to_num(itl[a.intl_col]),
    }).dropna()

    df = d.merge(i, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if df.empty:
        sys.exit("[중지] 겹치는 날짜가 없습니다. 두 파일의 조회 기간을 확인하세요.")

    prem = (df["krx"] / df["intl"] - 1) * 100
    if prem.abs().median() > 50:
        print("[경고] 프리미엄 중앙값이 비정상적입니다. 두 컬럼의 단위가 다를 수 있습니다"
              " (원/g vs 원/kg vs USD/oz). 컬럼 지정을 다시 확인하세요.")

    df.to_csv(a.out, index=False)
    print(f"저장: {a.out}  {len(df)}행  "
          f"({df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    print(f"프리미엄  평균 {prem.mean():.2f}%  중앙값 {prem.median():.2f}%  "
          f"최소 {prem.min():.2f}%  최대 {prem.max():.2f}%")


if __name__ == "__main__":
    main()
