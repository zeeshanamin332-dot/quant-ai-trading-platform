import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest


def make_data(closes, positions):
    return pd.DataFrame({"close": closes, "test_position": positions})


def test_signal_is_delayed_until_next_candle_and_fees_apply_to_turnover():
    data = make_data([100.0, 110.0, 99.0], [1, 1, -1])
    result, _ = run_backtest(data, "test_position", fee_rate=0.001, bars_per_year=365)

    assert result["position"].tolist() == [0.0, 1.0, 1.0]
    assert result["market_return"].tolist() == pytest.approx([0.0, 0.1, -0.1])
    assert result["turnover"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert result["fees"].tolist() == pytest.approx([0.0, 0.001, 0.0])
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.099, -0.1])


def test_reversal_counts_two_units_of_turnover():
    data = make_data([100.0, 110.0, 99.0], [1, -1, -1])
    result, _ = run_backtest(data, "test_position", fee_rate=0.001, bars_per_year=365)

    assert result["position"].tolist() == [0.0, 1.0, -1.0]
    assert result["turnover"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert result["fees"].tolist() == pytest.approx([0.0, 0.001, 0.002])
    assert result["strategy_return"].iloc[2] == pytest.approx(0.098)


def test_equity_and_drawdown_are_compounded_from_strategy_returns():
    data = make_data([100.0, 110.0, 99.0], [1, 1, 1])
    result, metrics = run_backtest(data, "test_position", fee_rate=0.0, bars_per_year=365)

    assert result["equity"].tolist() == pytest.approx([1.0, 1.1, 0.99])
    assert result["drawdown"].tolist() == pytest.approx([0.0, 0.0, -0.1])
    assert metrics["Strategy return"] == "-1.00%"
    assert metrics["Maximum drawdown"] == "-10.00%"


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="empty dataset"):
        run_backtest(pd.DataFrame(), "test_position", 0.001, 365)
    with pytest.raises(ValueError, match="Missing position column"):
        run_backtest(make_data([100.0], [0]), "missing", 0.001, 365)
    with pytest.raises(ValueError, match="non-negative"):
        run_backtest(make_data([100.0], [0]), "test_position", -0.001, 365)
    with pytest.raises(ValueError, match="positive integer"):
        run_backtest(make_data([100.0], [0]), "test_position", 0.001, 0)
