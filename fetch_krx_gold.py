#!/usr/bin/env python3
"""
KRX 금현물 종가 + 국제금 원화환산가를 받아 data.csv 로 저장한다.

[중요] KRX는 공개 REST API가 아니라 웹 화면이 내부적으로 쓰는 JSON 엔드포인트를
사용한다. bld 값과 파라미터 이름이 수시로 바뀌므로, 아래 BLD/PARAMS 를
직접 확인해서 채워야 한다. 확인 방법:

  1. 크롬에서 data.krx.co.kr 접속 → [일반상품] → [금] → 일별매매정보
  2. F12 → Network 탭 → Fetch/XHR 필터
  3. 화면에서 조회 버튼 클릭
  4. getJsonData.cmd 요청을 클릭 → Payload 탭
  5. 거기 보이는 bld, isuCd, strtDd, endDd 등을 그대로 아래에 복사

  국제금시세는 [일반상품] → [금] → 국제금시세 동향 에서 같은 방식으로 확인.

pykrx 라이브러리는 주식 위주라 금시장은 커버하지 않으므로 이 방식이 필요하다.
"""

import os
import sys
import time
import datetime as dt

import pandas as pd
import requests

BASE = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

# ---- DevTools 에서 확인한 값으로 채울 것 -------------------------------
DOMESTIC = {
    "referer_menu": "MDC0201060201",   # 금시장 일별매매정보 화면
    "bld": "",                          # 예: dbms/MDC/STAT/standard/MDCSTAT...
    "extra": {"isuCd": ""},             # 금 99.99_1Kg 종목코드
    "date_col": "TRD_DD",
    "price_col": "TDD_CLSPRC",
}
INTERNATIONAL = {
    "referer_menu": "MDC0201060207",   # 국제금시세 동향 화면
    "bld": "",
    "extra": {},
    "date_col": "TRD_DD",
    "price_col": "",                    # 원화환산 g당 가격 컬럼명
}
# ----------------------------------------------------------------------


def fetch(spec, start, end, retries=3):
    if not spec["bld"]:
        sys.exit("[중지] BLD 값이 비어 있습니다. 파일 상단 주석대로 DevTools에서 확인하세요.")

    referer = ("https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd"
               f"?menuId={spec['referer_menu']}")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"),
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = {"bld": spec["bld"], "strtDd": start, "endDd": end,
               "share": "1", "money": "1", "csvxls_isNo": "false", **spec["extra"]}

    last = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE, data=payload, headers=headers, timeout=30)
            r.raise_for_status()
            js = r.json()
            rows = js.get("output") or js.get("OutBlock_1") or []
            if not rows:
                raise ValueError(f"빈 응답입니다. 응답 키: {list(js.keys())}")
            df = pd.DataFrame(rows)
            if spec["price_col"] not in df.columns:
                sys.exit(f"[중지] price_col '{spec['price_col']}' 없음. 실제 컬럼: {list(df.columns)}")
            out = pd.DataFrame({
                "date": pd.to_datetime(df[spec["date_col"]].str.replace("/", "-")),
                "price": pd.to_numeric(df[spec["price_col"]].str.replace(",", ""),
                                       errors="coerce"),
            })
            return out.dropna().sort_values("date")
        except Exception as e:                      # noqa: BLE001
            last = e
            print(f"  시도 {attempt+1}/{retries} 실패: {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    sys.exit(f"[중지] 조회 실패: {last}")


def main():
    end = dt.date.today()
    # 기존 파일이 있으면 증분만, 없으면 과거 전체
    prev = None
    if os.path.exists("data.csv"):
        prev = pd.read_csv("data.csv", parse_dates=["date"])
        start = (prev["date"].max() - pd.Timedelta(days=10)).date()
    else:
        start = dt.date(2014, 3, 24)   # KRX 금시장 개설일

    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    print(f"조회 구간: {s} ~ {e}")

    print("국내 금현물...")
    dom = fetch(DOMESTIC, s, e).rename(columns={"price": "krx"})
    time.sleep(1)                       # KRX 서버 배려. 삭제하지 말 것
    print("국제금 원화환산...")
    intl = fetch(INTERNATIONAL, s, e).rename(columns={"price": "intl"})

    df = dom.merge(intl, on="date", how="inner")
    if prev is not None:
        df = (pd.concat([prev, df])
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date"))

    df.to_csv("data.csv", index=False)
    prem = (df["krx"] / df["intl"] - 1) * 100
    print(f"저장 완료: {len(df)}행  ({df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    print(f"최근 프리미엄: {prem.iloc[-1]:.2f}%   전체 평균 {prem.mean():.2f}%")


if __name__ == "__main__":
    main()
