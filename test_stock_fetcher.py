import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stock_fetcher import StockDataFetcher


# ============================================================
# 夹具
# ============================================================

@pytest.fixture
def fetcher():
    return StockDataFetcher(rate_limit=0.0)


def _make_a_daily_df(n=3):
    """构造一个模拟的 A 股日线 DataFrame"""
    return pd.DataFrame({
        "日期": ["2026-05-18", "2026-05-19", "2026-05-20"][:n],
        "开盘": [28.5, 28.8, 29.0][:n],
        "收盘": [28.8, 29.0, 29.5][:n],
        "最高": [29.0, 29.2, 29.8][:n],
        "最低": [28.2, 28.5, 28.9][:n],
        "成交量": [50000, 60000, 55000][:n],
        "成交额": [1430000, 1730000, 1610000][:n],
        "振幅": [2.8, 2.4, 3.1][:n],
        "涨跌幅": [1.1, 0.7, 1.7][:n],
        "涨跌额": [0.3, 0.2, 0.5][:n],
        "换手率": [1.2, 1.5, 1.3][:n],
    })


# ============================================================
# __init__ 参数校验
# ============================================================

class TestInit:
    def test_default_values(self, fetcher):
        assert fetcher._rate_limit == 0.0
        assert fetcher._max_retries == 3

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            StockDataFetcher(max_retries=0)

    def test_max_retries_negative_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            StockDataFetcher(max_retries=-1)


# ============================================================
# _wait — 速率控制
# ============================================================

class TestWait:
    def test_first_call_no_sleep(self, fetcher):
        fetcher._last_call = 0.0
        start = time.time()
        fetcher._wait()
        assert time.time() - start < 0.05  # 几乎不休眠

    def test_consecutive_call_waits(self, fetcher):
        fetcher._rate_limit = 0.1
        fetcher._last_call = time.time()  # 刚刚调用过
        start = time.time()
        fetcher._wait()
        # 应该在 ~0.1s 后返回
        elapsed = time.time() - start
        assert 0.08 <= elapsed < 0.2


# ============================================================
# _retry — 重试逻辑
# ============================================================

class TestRetry:
    def test_success_first_attempt(self, fetcher):
        func = MagicMock(return_value="ok")
        result = fetcher._retry(func, 1, key="val")
        assert result == "ok"
        func.assert_called_once_with(1, key="val")

    def test_retry_on_failure_then_succeed(self, fetcher):
        func = MagicMock(side_effect=[ValueError("fail1"), ValueError("fail2"), "ok"])
        result = fetcher._retry(func)
        assert result == "ok"
        assert func.call_count == 3

    def test_all_retries_exhausted(self, fetcher):
        func = MagicMock(side_effect=ValueError("always fail"))
        with pytest.raises(ValueError, match="always fail"):
            fetcher._retry(func)
        assert func.call_count == 3

    def test_single_retry_disabled(self):
        fetcher = StockDataFetcher(max_retries=1, rate_limit=0.0)
        func = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            fetcher._retry(func)
        assert func.call_count == 1


# ============================================================
# _pad_hk_code — 港股代码补零
# ============================================================

class TestPadHkCode:
    def test_short_code_padded(self):
        assert StockDataFetcher._pad_hk_code("700") == "00700"

    def test_full_code_unchanged(self):
        assert StockDataFetcher._pad_hk_code("00700") == "00700"

    def test_stripped_code(self):
        assert StockDataFetcher._pad_hk_code(" 700 ") == "00700"


# ============================================================
# get_a_daily
# ============================================================

class TestGetADaily:
    def test_returns_clean_dataframe(self, fetcher):
        mock_df = _make_a_daily_df(3)
        with patch("stock_fetcher.ak.stock_zh_a_hist", return_value=mock_df):
            df = fetcher.get_a_daily("002294", start_date="20260501", end_date="20260520")
        assert len(df) == 3
        assert "date" in df.columns
        assert "日期" not in df.columns
        assert df["date"].iloc[0] == pd.Timestamp("2026-05-18")

    def test_empty_dataframe_returned_as_is(self, fetcher):
        empty_df = pd.DataFrame()
        with patch("stock_fetcher.ak.stock_zh_a_hist", return_value=empty_df):
            df = fetcher.get_a_daily("000001")
        assert df.empty

    def test_default_end_date(self, fetcher):
        mock_df = _make_a_daily_df(1)
        with patch("stock_fetcher.ak.stock_zh_a_hist", return_value=mock_df) as mock_func:
            fetcher.get_a_daily("600519", start_date="20260101")
            args, kwargs = mock_func.call_args
            assert "end_date" in kwargs
            assert kwargs["end_date"] is not None  # 自动填充为今天
            assert kwargs["adjust"] == "qfq"


# ============================================================
# get_hk_daily
# ============================================================

class TestGetHkDaily:
    def test_returns_clean_dataframe(self, fetcher):
        mock_df = _make_a_daily_df(3)
        with patch("stock_fetcher.ak.stock_hk_hist", return_value=mock_df):
            df = fetcher.get_hk_daily("00700", start_date="20260501")
        assert len(df) == 3
        assert "date" in df.columns

    def test_auto_pads_code(self, fetcher):
        mock_df = _make_a_daily_df(1)
        with patch("stock_fetcher.ak.stock_hk_hist", return_value=mock_df) as mock_func:
            fetcher.get_hk_daily("700")
            assert mock_func.call_args.kwargs["symbol"] == "00700"

    def test_empty_dataframe(self, fetcher):
        with patch("stock_fetcher.ak.stock_hk_hist", return_value=pd.DataFrame()):
            df = fetcher.get_hk_daily("00700")
        assert df.empty


# ============================================================
# get_daily — 统一路由
# ============================================================

class TestGetDaily:
    def test_routes_to_a_share(self, fetcher):
        mock_df = _make_a_daily_df(2)
        with patch("stock_fetcher.ak.stock_zh_a_hist", return_value=mock_df):
            df = fetcher.get_daily("002294", market="a", start_date="20260501")
        assert len(df) == 2

    def test_routes_to_hk_share(self, fetcher):
        mock_df = _make_a_daily_df(2)
        with patch("stock_fetcher.ak.stock_hk_hist", return_value=mock_df):
            df = fetcher.get_daily("00700", market="hk", start_date="20260501")
        assert len(df) == 2

    def test_invalid_market_raises(self, fetcher):
        with pytest.raises(ValueError, match="market"):
            fetcher.get_daily("002294", market="us")


# ============================================================
# get_a_spot / get_a_spot_single
# ============================================================

class TestASpot:
    def test_get_a_spot(self, fetcher):
        mock_df = pd.DataFrame({
            "代码": ["002294", "600519"],
            "名称": ["信立泰", "贵州茅台"],
            "最新价": [28.8, 1800.0],
        })
        with patch("stock_fetcher.ak.stock_zh_a_spot_em", return_value=mock_df):
            df = fetcher.get_a_spot()
        assert len(df) == 2

    def test_get_a_spot_single_found(self, fetcher):
        mock_df = pd.DataFrame({
            "代码": ["002294", "600519"],
            "名称": ["信立泰", "贵州茅台"],
            "最新价": [28.8, 1800.0],
        })
        with patch("stock_fetcher.ak.stock_zh_a_spot_em", return_value=mock_df):
            info = fetcher.get_a_spot_single("002294")
        assert info["名称"] == "信立泰"
        assert info["最新价"] == 28.8

    def test_get_a_spot_single_not_found(self, fetcher):
        mock_df = pd.DataFrame({"代码": ["600519"], "名称": ["贵州茅台"]})
        with patch("stock_fetcher.ak.stock_zh_a_spot_em", return_value=mock_df):
            with pytest.raises(ValueError, match="未找到"):
                fetcher.get_a_spot_single("000001")


# ============================================================
# get_hk_spot / get_hk_spot_single
# ============================================================

class TestHKSpot:
    def test_get_hk_spot(self, fetcher):
        mock_df = pd.DataFrame({
            "代码": ["00700", "09988"],
            "名称": ["腾讯控股", "阿里巴巴"],
        })
        with patch("stock_fetcher.ak.stock_hk_spot_em", return_value=mock_df):
            df = fetcher.get_hk_spot()
        assert len(df) == 2

    def test_get_hk_spot_single_auto_pads(self, fetcher):
        mock_df = pd.DataFrame({"代码": ["00700"], "名称": ["腾讯控股"]})
        with patch("stock_fetcher.ak.stock_hk_spot_em", return_value=mock_df):
            info = fetcher.get_hk_spot_single("700")
        assert info["名称"] == "腾讯控股"


# Tag for test2