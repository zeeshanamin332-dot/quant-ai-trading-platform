"""Small, testable backtest calculations used by the Streamlit app."""

import numpy as np
import pandas as pd


def run_backtest(data: pd.DataFrame, position_col: str, fee_rate: float, bars_per_year: int):
    """Simulate close-to-close returns using the previous candle's signal.

    Positions are assumed to be established at the prior candle close. A change
    from +1 to -1 has turnover 2, representing closing one unit and opening one
    unit in the opposite direction. Fees are a simplified proportional model.
    """
    if data.empty:
        raise ValueError("Cannot backtest an empty dataset.")
    if position_col not in data.columns:
        raise ValueError(f"Missing position column: {position_col}")
    if not np.isfinite(fee_rate) or fee_rate < 0:
        raise ValueError("fee_rate must be a finite, non-negative number.")
    if not isinstance(bars_per_year, (int, np.integer)) or bars_per_year <= 0:
        raise ValueError("bars_per_year must be a positive integer.")
    if not np.isfinite(data[position_col].to_numpy(dtype=float)).all():
        raise ValueError("Position signals must be finite numbers.")

    result = data.copy()
    # A signal observed at candle close is used only for the next candle's return.
    result["position"] = result[position_col].shift(1).fillna(0)
    result["market_return"] = result["close"].pct_change().fillna(0)
    result["turnover"] = result["position"].diff().abs()
    result.iloc[0, result.columns.get_loc("turnover")] = abs(result["position"].iloc[0])
    result["turnover"] = result["turnover"].fillna(0)
    result["fees"] = result["turnover"] * fee_rate
    result["strategy_return"] = result["position"] * result["market_return"] - result["fees"]
    result["equity"] = (1 + result["strategy_return"]).cumprod()
    result["buy_hold"] = (1 + result["market_return"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1

    returns = result["strategy_return"]
    vol = returns.std(ddof=1)
    sharpe = np.sqrt(bars_per_year) * returns.mean() / vol if pd.notna(vol) and vol > 0 else np.nan
    metrics = {
        "Strategy return": f"{(result['equity'].iloc[-1] - 1) * 100:.2f}%",
        "Buy & hold return": f"{(result['buy_hold'].iloc[-1] - 1) * 100:.2f}%",
        "Maximum drawdown": f"{result['drawdown'].min() * 100:.2f}%",
        "Estimated annualized Sharpe": f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A",
        "Position changes": f"{int(result['turnover'].gt(0).sum())}",
        "Estimated fees (return units)": f"{result['fees'].sum():.4f}",
    }
    return result, metrics
