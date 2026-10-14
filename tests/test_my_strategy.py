import numpy as np
import pandas as pd
import pytest

from my_strategy import add_my_strategy


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
