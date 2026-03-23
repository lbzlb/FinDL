import os
import random
import time

import pandas as pd
from curl_cffi import requests
from tqdm import tqdm

# 1. 纯字典映射，干净直观
INDEX_MAP = {
    "沪深300": "1.000300",
    "上证指数": "1.000001",
    "深证成指": "0.399001",
    "创业板指": "0.399006",
    "中证500": "1.000905",
    "上证50": "1.000016",
    "标普500": "100.SPX",
    "纳斯达克": "100.NDX",
    "道琼斯": "100.DJIA",
    "恒生指数": "100.HSI",
    "恒生科技": "124.HSTECH",
}


def fetch_kline(session, secid):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 0,
        "beg": "0",
        "end": "20500101",
        "_": int(time.time() * 1000),
    }

    resp = session.get(url, params=params)
    klines = (resp.json().get("data") or {}).get("klines", [])
    if not klines:
        return None

    cols = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    df = pd.DataFrame([k.split(",") for k in klines], columns=cols)

    df["日期"] = pd.to_datetime(df["日期"])
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df


def main():
    os.makedirs("src/local/data", exist_ok=True)
    filename = "src/local/data/macro_indices.xlsx"

    with (
        requests.Session(impersonate="chrome120", timeout=10) as session,
        pd.ExcelWriter(filename, engine="openpyxl") as writer,
    ):
        for name, secid in tqdm(INDEX_MAP.items(), desc="抓取指数"):
            try:
                df = fetch_kline(session, secid)
                if df is not None:
                    df.to_excel(writer, sheet_name=name, index=False)
                    writer.sheets[name].freeze_panes = "A2"
            except Exception as e:
                tqdm.write(f" ❌ {name} 失败: {e}")

            time.sleep(random.uniform(1, 3))


if __name__ == "__main__":
    main()
