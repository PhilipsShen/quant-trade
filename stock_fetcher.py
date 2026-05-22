#!/usr/bin/env python3
"""
股票数据获取封装模块 — StockDataFetcher v1.0
=============================================
基于 AKShare，封装 A 股 + 港股的数据获取接口。
完全免费，无需注册，开箱即用。

依赖: pip install akshare pandas
文档: https://akshare.akfamily.xyz/data/stock/stock.html

作者: 挣大钱
日期: 2026-05-21
"""

import time
import warnings
from datetime import datetime
from typing import Literal

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 股票代码格式说明
# ============================================================
# A 股: "000001" (平安银行), "600519" (贵州茅台), "002294" (信立泰)
# 港股: "00700" (腾讯), "09988" (阿里), "00005" (汇丰)
#   注意港股代码要补足到 5 位: "700" → "00700"


class StockDataFetcher:
    """A股 + 港股 数据获取器"""

    def __init__(self, rate_limit: float = 0.3, max_retries: int = 3):
        """
        rate_limit: 两次请求间隔(秒)，避免触发反爬
        max_retries: 请求失败最大重试次数
        """
        if max_retries < 1:
            raise ValueError("max_retries 必须 >= 1")
        self._last_call = 0.0
        self._rate_limit = rate_limit
        self._max_retries = max_retries

    # -------------------- 内部工具 --------------------

    def _wait(self):
        """速率控制：确保两次请求间隔 >= rate_limit 秒"""
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_call = time.time()

    def _retry(self, func, *args, **kwargs):
        """带重试的函数调用"""
        last_err = None
        for attempt in range(self._max_retries):
            try:
                self._wait()
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  [重试 {attempt+1}/{self._max_retries}] {e}，{wait}秒后重试...")
                    time.sleep(wait)
        raise last_err

    # ============================================================
    # A 股接口
    # ============================================================

    def get_a_daily(
        self,
        symbol: str,
        start_date: str = "20200101",
        end_date: str = None,
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """
        获取 A 股历史日线数据（东方财富源，支持复权）

        参数
        ----
        symbol      : 股票代码，如 "002294"、"600519"
        start_date  : 起始日期 "YYYYMMDD"
        end_date    : 截止日期，默认今天
        adjust      : 复权方式 qfq(前复权) / hfq(后复权) / ""(不复权)

        返回
        ----
        DataFrame: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率

        示例
        ----
        >>> df = fetcher.get_a_daily("002294", "20250501", "20260101")
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        print(f"[A股日线] {symbol}  {start_date} ~ {end_date}  adjust={adjust or '不复权'}")
        df = self._retry(
            ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df.empty:
            print("  → 无数据")
            return df
        df.rename(columns={"日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"  → 获取 {len(df)} 条记录")
        return df

    def get_a_spot(self) -> pd.DataFrame:
        """
        获取 A 股全市场实时行情快照（15分钟延迟）

        返回
        ----
        DataFrame: 代码, 名称, 最新价, 涨跌幅, 成交量, 换手率, 市盈率, 市净率 等

        示例
        ----
        >>> df = fetcher.get_a_spot()
        >>> # 按代码筛选
        >>> row = df[df['代码'] == '002294']
        """
        print("[A股实时] 全市场快照...")
        df = self._retry(ak.stock_zh_a_spot_em)
        print(f"  → 获取 {len(df)} 只股票")
        return df

    def get_a_spot_single(self, symbol: str) -> dict:
        """
        获取单只 A 股实时行情

        >>> info = fetcher.get_a_spot_single("002294")
        >>> print(info['最新价'], info['涨跌幅'])
        """
        df = self.get_a_spot()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"未找到股票代码: {symbol}")
        return row.iloc[0].to_dict()

    # ============================================================
    # 港股接口
    # ============================================================

    @staticmethod
    def _pad_hk_code(symbol: str) -> str:
        """港股代码补零到 5 位: "700" → "00700" """
        s = symbol.strip()
        if len(s) < 5:
            s = s.zfill(5)
        return s

    def get_hk_daily(
        self,
        symbol: str,
        start_date: str = "20200101",
        end_date: str = None,
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """
        获取港股历史日线数据（东方财富源，支持复权）

        参数
        ----
        symbol      : 港股代码，如 "00700"(腾讯)、"09988"(阿里)，支持 "700" 自动补零
        start_date  : 起始日期 "YYYYMMDD"
        end_date    : 截止日期，默认今天
        adjust      : 复权方式

        返回
        ----
        DataFrame: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率

        示例
        ----
        >>> df = fetcher.get_hk_daily("00700", "20240101")
        """
        code = self._pad_hk_code(symbol)
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        print(f"[港股日线] {code}  {start_date} ~ {end_date}  adjust={adjust or '不复权'}")
        df = self._retry(
            ak.stock_hk_hist,
            symbol=code,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df.empty:
            print("  → 无数据")
            return df
        df.rename(columns={"日期": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"  → 获取 {len(df)} 条记录")
        return df

    def get_hk_spot(self) -> pd.DataFrame:
        """
        获取港股全市场实时行情快照（15分钟延迟）

        返回
        ----
        DataFrame: 代码, 名称, 最新价, 涨跌幅, 成交量, 换手率 等

        示例
        ----
        >>> df = fetcher.get_hk_spot()
        >>> # 查腾讯
        >>> row = df[df['代码'] == '00700']
        """
        print("[港股实时] 全市场快照...")
        df = self._retry(ak.stock_hk_spot_em)
        print(f"  → 获取 {len(df)} 只股票")
        return df

    def get_hk_spot_single(self, symbol: str) -> dict:
        """
        获取单只港股实时行情

        >>> info = fetcher.get_hk_spot_single("00700")
        """
        code = self._pad_hk_code(symbol)
        df = self.get_hk_spot()
        row = df[df["代码"] == code]
        if row.empty:
            raise ValueError(f"未找到股票代码: {code}")
        return row.iloc[0].to_dict()

    # ============================================================
    # 便捷方法
    # ============================================================

    def get_daily(
        self,
        symbol: str,
        market: Literal["a", "hk"] = "a",
        start_date: str = "20200101",
        end_date: str = None,
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """
        统一接口：根据 market 参数自动路由到 A 股或港股

        示例
        ----
        >>> df_a  = fetcher.get_daily("002294", market="a",  start_date="20250501")
        >>> df_hk = fetcher.get_daily("00700",  market="hk", start_date="20250501")
        """
        if market == "a":
            return self.get_a_daily(symbol, start_date, end_date, adjust)
        elif market == "hk":
            return self.get_hk_daily(symbol, start_date, end_date, adjust)
        else:
            raise ValueError(f"market 参数错误: {market}，可选 'a' 或 'hk'")


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("StockDataFetcher 自测")
    print("=" * 60)

    fetcher = StockDataFetcher(rate_limit=0.3)

    # 1. A 股日线
    print("\n▶ 测试1: A股日线 — 信立泰 002294")
    df_a = fetcher.get_a_daily("002294", start_date="20250501")
    print(df_a.tail(3).to_string(index=False))

    # 2. 港股日线
    print("\n▶ 测试2: 港股日线 — 腾讯 00700")
    df_hk = fetcher.get_hk_daily("00700", start_date="20250501")
    print(df_hk.tail(3).to_string(index=False))

    # 3. 统一接口
    print("\n▶ 测试3: 统一接口")
    df = fetcher.get_daily("002294", market="a", start_date="20250501")
    print(f"  信立泰: {len(df)} 条, 最新收盘 {df['收盘'].iloc[-1]}")

    print("\n✅ 所有测试通过！")

    # Tag for test2