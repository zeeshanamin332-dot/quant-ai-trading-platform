"""Rule-based liquidity-sweep strategy (research indicator, not financial advice)."""

import numpy as np
import pandas as pd


def _order_blocks(htf: pd.DataFrame, end_time: pd.Timestamp, duration: pd.Timedelta):
    """Find the most recent completed bullish and bearish order-block midpoints.

    A bullish block is the last bearish candle within five bars before a strong
    bullish displacement candle. The bearish definition is mirrored. This is a
    transparent approximation, not a universal definition of an order block.
    """
    available = htf.loc[(htf.index + duration) <= end_time]
    if len(available) < 12:
        return None, None
    bodies = (available["close"] - available["open"]).abs()
    average = bodies.shift(1).rolling(10, min_periods=5).mean()
    bullish = bearish = None
    rows = available.reset_index()
    for i in range(10, len(rows)):
        row = rows.iloc[i]
        body = abs(row["close"] - row["open"])
        threshold = average.iloc[i]
        if not np.isfinite(threshold) or threshold <= 0 or body < 1.5 * threshold:
            continue
        if row["close"] > row["open"]:
            for j in range(i - 1, max(-1, i - 6), -1):
                prior = rows.iloc[j]
                if prior["close"] < prior["open"]:
                    bullish = (float(prior["low"]), float(prior["high"]),
                               (float(prior["low"]) + float(prior["high"])) / 2)
                    break
        elif row["close"] < row["open"]:
            for j in range(i - 1, max(-1, i - 6), -1):
                prior = rows.iloc[j]
                if prior["close"] > prior["open"]:
                    bearish = (float(prior["low"]), float(prior["high"]),
                               (float(prior["low"]) + float(prior["high"])) / 2)
                    break
    return bullish, bearish


def add_my_strategy(
    candles: pd.DataFrame,
    higher_timeframe: pd.DataFrame,
    swing_lookback: int = 10,
    risk_reward: int = 2,
    base_timeframe: str = "1h",
    higher_timeframe_name: str = "1h",
) -> pd.DataFrame:
    """Return candles with BUY/SELL events, approximate position, stop and target.

    Rules are deliberately explicit:
    - sweep: wick breaches the prior swing level and candle closes back inside;
    - rejection: relevant wick is at least 1.2x the candle body;
    - second sweep/rejection follows within four candles;
    - displacement breaks the prior opposite-colour candle wick, with body at
      least 1.2x the average body of the prior ten candles;
    - displacement candle touches and closes on the correct side of the latest
      completed higher-timeframe order-block midpoint.
    """
    if candles.empty:
        raise ValueError("No candles supplied for My strategy.")
    if higher_timeframe.empty:
        raise ValueError("No higher-timeframe candles supplied.")
    if swing_lookback < 3:
        raise ValueError("Swing lookback must be at least 3.")
    if risk_reward not in (1, 2):
        raise ValueError("Risk-to-reward must be 1 or 2.")
    if base_timeframe not in {"15m", "1h", "4h", "1d"}:
        raise ValueError("Unsupported base timeframe.")
    if higher_timeframe_name not in {"15m", "1h", "4h", "1d"}:
        raise ValueError("Unsupported higher timeframe.")

    base_seconds = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    htf_duration = pd.Timedelta(seconds=base_seconds[higher_timeframe_name])
    base_duration = pd.Timedelta(seconds=base_seconds[base_timeframe])
    data = candles.copy()
    data["my_signal"] = 0
    data["my_position"] = 0.0
    data["my_entry"] = np.nan
    data["my_stop"] = np.nan
    data["my_target"] = np.nan
    data["my_ob_midpoint"] = np.nan
    if len(data) < max(swing_lookback + 15, 30):
        return data

    body = (data["close"] - data["open"]).abs()
    avg_body = body.shift(1).rolling(10, min_periods=5).mean()
    lower_wick = data[["open", "close"]].min(axis=1) - data["low"]
    upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
    swing_low = data["low"].shift(1).rolling(swing_lookback).min()
    swing_high = data["high"].shift(1).rolling(swing_lookback).max()

    # Store initial sweep/rejection candidates and search only forward in time.
    buy_candidates = []
    sell_candidates = []
    for i in range(swing_lookback, len(data)):
        row = data.iloc[i]
        if pd.notna(swing_low.iloc[i]) and row["low"] < swing_low.iloc[i] and row["close"] > swing_low.iloc[i] and lower_wick.iloc[i] >= max(body.iloc[i] * 1.2, 0):
            buy_candidates.append((i, float(row["low"])))
        if pd.notna(swing_high.iloc[i]) and row["high"] > swing_high.iloc[i] and row["close"] < swing_high.iloc[i] and upper_wick.iloc[i] >= max(body.iloc[i] * 1.2, 0):
            sell_candidates.append((i, float(row["high"])))

    used = set()
    for i in range(swing_lookback + 10, len(data)):
        row = data.iloc[i]
        if not np.isfinite(avg_body.iloc[i]) or avg_body.iloc[i] <= 0:
            continue
        end_time = data.index[i] + base_duration
        bull_ob, bear_ob = _order_blocks(higher_timeframe, end_time, htf_duration)
        prior_red = next((j for j in range(i - 1, max(-1, i - 6), -1)
                          if data.iloc[j]["close"] < data.iloc[j]["open"]), None)
        prior_green = next((j for j in range(i - 1, max(-1, i - 6), -1)
                            if data.iloc[j]["close"] > data.iloc[j]["open"]), None)

        # BUY: first low sweep/rejection, second sweep of that rejection low,
        # then bullish displacement through a recent red candle high.
        if prior_red is not None and row["close"] > row["open"] and row["close"] > data.iloc[prior_red]["high"] and body.iloc[i] >= 1.2 * avg_body.iloc[i]:
            for first_i, first_low in reversed(buy_candidates):
                if first_i >= i or i - first_i > 8 or first_i in used:
                    continue
                second = next((j for j in range(first_i + 1, i)
                               if data.iloc[j]["low"] < first_low
                               and data.iloc[j]["close"] > first_low
                               and lower_wick.iloc[j] >= body.iloc[j] * 1.2), None)
                if second is None or bull_ob is None:
                    continue
                ob_low, ob_high, midpoint = bull_ob
                if row["low"] <= midpoint <= row["high"] and row["close"] > midpoint:
                    stop = min(first_low, float(data.iloc[second]["low"]))
                    entry = float(row["close"])
                    if stop < entry:
                        data.iloc[i, data.columns.get_loc("my_signal")] = 1
                        data.iloc[i, data.columns.get_loc("my_entry")] = entry
                        data.iloc[i, data.columns.get_loc("my_stop")] = stop
                        data.iloc[i, data.columns.get_loc("my_target")] = entry + risk_reward * (entry - stop)
                        data.iloc[i, data.columns.get_loc("my_ob_midpoint")] = midpoint
                        used.add(first_i)
                        break

        # SELL: mirror the BUY logic at swing highs.
        if data.iloc[i]["my_signal"] == 0 and prior_green is not None and row["close"] < row["open"] and row["close"] < data.iloc[prior_green]["low"] and body.iloc[i] >= 1.2 * avg_body.iloc[i]:
            for first_i, first_high in reversed(sell_candidates):
                if first_i >= i or i - first_i > 8 or first_i in used:
                    continue
                second = next((j for j in range(first_i + 1, i)
                               if data.iloc[j]["high"] > first_high
                               and data.iloc[j]["close"] < first_high
                               and upper_wick.iloc[j] >= body.iloc[j] * 1.2), None)
                if second is None or bear_ob is None:
                    continue
                ob_low, ob_high, midpoint = bear_ob
                if row["low"] <= midpoint <= row["high"] and row["close"] < midpoint:
                    stop = max(first_high, float(data.iloc[second]["high"]))
                    entry = float(row["close"])
                    if stop > entry:
                        data.iloc[i, data.columns.get_loc("my_signal")] = -1
                        data.iloc[i, data.columns.get_loc("my_entry")] = entry
                        data.iloc[i, data.columns.get_loc("my_stop")] = stop
                        data.iloc[i, data.columns.get_loc("my_target")] = entry - risk_reward * (stop - entry)
                        data.iloc[i, data.columns.get_loc("my_ob_midpoint")] = midpoint
                        used.add(first_i)
                        break

    # Carry the last signal as a research position until the opposite signal.
    active = 0.0
    for i, signal in enumerate(data["my_signal"].to_numpy()):
        if signal:
            active = float(signal)
        data.iloc[i, data.columns.get_loc("my_position")] = active
    return data



def simulate_my_trades(candles: pd.DataFrame) -> pd.DataFrame:
    """Simulate non-overlapping trades from strategy signals using candle OHLC.

    Signals are known only after their candle closes, so entries use the next
    candle's open. If TP and SL are both inside one candle's range, assume SL
    first (conservative because OHLC data does not reveal intrabar ordering).
    """
    columns = [
        "Signal time", "Entry time", "Exit time", "Side", "Entry",
        "Stop-loss", "Take-profit", "Outcome", "Exit price", "P/L %",
        "R multiple", "Bars held",
    ]
    if candles.empty or "my_signal" not in candles:
        return pd.DataFrame(columns=columns)

    trades = []
    i = 0
    while i < len(candles) - 1:
        signal = int(candles["my_signal"].iloc[i])
        if signal == 0:
            i += 1
            continue

        entry_i = i + 1
        entry = float(candles["open"].iloc[entry_i])
        stop = float(candles["my_stop"].iloc[i])
        target = float(candles["my_target"].iloc[i])
        if not np.isfinite([entry, stop, target]).all():
            i += 1
            continue
        risk = entry - stop if signal > 0 else stop - entry
        if risk <= 0:
            i += 1
            continue
        # Keep the risk distance from the signal's projected levels, but anchor
        # the actual trade levels to the next candle open to avoid lookahead.
        reward_multiple = abs(float(candles["my_target"].iloc[i]) - float(candles["my_entry"].iloc[i])) / max(
            abs(float(candles["my_entry"].iloc[i]) - float(candles["my_stop"].iloc[i])), 1e-12
        )
        stop = entry - risk if signal > 0 else entry + risk
        target = entry + risk * reward_multiple if signal > 0 else entry - risk * reward_multiple

        outcome = "OPEN"
        exit_price = float(candles["close"].iloc[-1])
        exit_i = len(candles) - 1
        for j in range(entry_i, len(candles)):
            high = float(candles["high"].iloc[j])
            low = float(candles["low"].iloc[j])
            if signal > 0:
                sl_hit, tp_hit = low <= stop, high >= target
            else:
                sl_hit, tp_hit = high >= stop, low <= target
            if sl_hit:
                outcome, exit_price, exit_i = "SL HIT", stop, j
                break
            if tp_hit:
                outcome, exit_price, exit_i = "TP HIT", target, j
                break

        pnl_pct = ((exit_price - entry) / entry * 100) * signal
        r_multiple = ((exit_price - entry) * signal) / risk
        trades.append({
            "Signal time": candles.index[i],
            "Entry time": candles.index[entry_i],
            "Exit time": candles.index[exit_i] if outcome != "OPEN" else pd.NaT,
            "Side": "BUY" if signal > 0 else "SELL",
            "Entry": entry,
            "Stop-loss": stop,
            "Take-profit": target,
            "Outcome": outcome,
            "Exit price": exit_price if outcome != "OPEN" else np.nan,
            "P/L %": pnl_pct if outcome != "OPEN" else np.nan,
            "R multiple": r_multiple if outcome != "OPEN" else np.nan,
            "Bars held": exit_i - entry_i,
        })
        # Do not overlap simulated positions. Continue after the exit candle.
        i = max(exit_i + 1, i + 1)

    return pd.DataFrame(trades, columns=columns)
