import numpy as np
import pandas as pd
import pytest

from my_strategy import add_my_strategy, simulate_my_trades


def make_candles(n=50):
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.ones(n),
    }, index=idx)


def test_returns_signal_and_risk_columns_without_lookahead_error():
    base = make_candles()
    htf = make_candles(80)
    result = add_my_strategy(base, htf, risk_reward=2)
    for column in ["my_signal", "my_position", "my_entry", "my_stop", "my_target", "my_ob_midpoint"]:
        assert column in result.columns
    assert set(result["my_signal"].unique()).issubset({-1, 0, 1})


def test_rejects_invalid_risk_reward():
    with pytest.raises(ValueError):
        add_my_strategy(make_candles(), make_candles(80), risk_reward=3)


def test_rejects_short_swing_lookback():
    with pytest.raises(ValueError):
        add_my_strategy(make_candles(), make_candles(80), swing_lookback=2)



def test_trade_simulator_records_take_profit_and_projected_levels():
    candles = make_candles(3)
    candles["my_signal"] = 0
    candles["my_entry"] = np.nan
    candles["my_stop"] = np.nan
    candles["my_target"] = np.nan
    candles.iloc[0, candles.columns.get_loc("my_signal")] = 1
    candles.iloc[0, candles.columns.get_loc("my_entry")] = 100.0
    candles.iloc[0, candles.columns.get_loc("my_stop")] = 95.0
    candles.iloc[0, candles.columns.get_loc("my_target")] = 110.0
    candles.iloc[1, candles.columns.get_loc("open")] = 101.0
    candles.iloc[1, candles.columns.get_loc("high")] = 114.0
    candles.iloc[1, candles.columns.get_loc("low")] = 100.0
    trades = simulate_my_trades(candles)
    assert len(trades) == 1
    assert trades.iloc[0]["Outcome"] == "TP HIT"
    assert trades.iloc[0]["Entry"] == 101.0
    assert trades.iloc[0]["Take-profit"] == 113.0


def test_trade_simulator_assumes_stop_first_if_both_levels_hit():
    candles = make_candles(3)
    candles["my_signal"] = 0
    candles["my_entry"] = np.nan
    candles["my_stop"] = np.nan
    candles["my_target"] = np.nan
    candles.iloc[0, candles.columns.get_loc("my_signal")] = 1
    candles.iloc[0, candles.columns.get_loc("my_entry")] = 100.0
    candles.iloc[0, candles.columns.get_loc("my_stop")] = 95.0
    candles.iloc[0, candles.columns.get_loc("my_target")] = 110.0
    candles.iloc[1, candles.columns.get_loc("open")] = 100.0
    candles.iloc[1, candles.columns.get_loc("high")] = 111.0
    candles.iloc[1, candles.columns.get_loc("low")] = 94.0
    trades = simulate_my_trades(candles)
    assert trades.iloc[0]["Outcome"] == "SL HIT"
