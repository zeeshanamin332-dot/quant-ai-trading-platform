import numpy as np
import pandas as pd
import streamlit as st
import ccxt
from backtest import run_backtest
from my_strategy import add_my_strategy

st.set_page_config(
    page_title="Quant AI Trading Platform",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quant AI Trading Platform")
st.caption("Crypto research dashboard • MACD • Bollinger Bands • Liquidity sweeps • Backtesting")
st.warning(
    "Research only — this app does not place trades. Backtests are simulations, "
    "and past performance does not predict future results."
)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_candles(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    exchange = ccxt.kraken({"enableRateLimit": True})
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not candles:
        raise ValueError("The exchange returned no candles.")
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df = df.astype(float)

    # Kraken can include the currently forming candle. Exclude it until its
    # full interval has elapsed so signals use completed candles only.
    timeframe_seconds = {"15m": 900, "1h": 3600, "4h": 4 * 3600, "1d": 24 * 3600}
    if timeframe not in timeframe_seconds:
        raise ValueError(f"Unsupported candle interval: {timeframe}")
    if not df.empty:
        last_candle_open = df.index[-1]
        candle_close_time = last_candle_open + pd.Timedelta(
            seconds=timeframe_seconds[timeframe]
        )
        if candle_close_time > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]

    if df.empty or (df["close"] <= 0).any():
        raise ValueError("No completed, valid candles were returned.")
    if not np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("The candle data contains non-finite values.")
    if (df["high"] < df["low"]).any():
        raise ValueError("The candle data contains a high price below its low.")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("The candle data contains a high below the open or close.")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("The candle data contains a low above the open or close.")
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    prev_macd = data["macd"].shift(1)
    prev_sig = data["macd_signal"].shift(1)
    data["macd_event"] = np.select(
        [
            (data["macd"] > data["macd_signal"]) & (prev_macd <= prev_sig),
            (data["macd"] < data["macd_signal"]) & (prev_macd >= prev_sig),
        ],
        [1, -1],
        default=0,
    )
    data["macd_position"] = data["macd_event"].replace(0, np.nan).ffill().fillna(0)

    data["bb_mid"] = data["close"].rolling(20).mean()
    rolling_std = data["close"].rolling(20).std(ddof=0)
    data["bb_upper"] = data["bb_mid"] + 2 * rolling_std
    data["bb_lower"] = data["bb_mid"] - 2 * rolling_std
    prev_close = data["close"].shift(1)
    prev_lower = data["bb_lower"].shift(1)
    prev_upper = data["bb_upper"].shift(1)
    data["bb_event"] = np.select(
        [
            (data["close"] < data["bb_lower"]) & (prev_close >= prev_lower),
            (data["close"] > data["bb_upper"]) & (prev_close <= prev_upper),
        ],
        [1, -1],
        default=0,
    )
    # A deliberately simple one-candle Bollinger exposure for the first version.
    data["bb_position"] = data["bb_event"]
    return data


with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Crypto pair", ["BTC/USD", "ETH/USD", "SOL/USD", "BTC/USDT", "ETH/USDT"])
    timeframe = st.selectbox("Candle interval", ["1h", "4h", "1d"])
    limit = st.slider("Number of candles", min_value=100, max_value=720, value=500, step=100)
    fee_pct = st.number_input("Estimated fee per unit of turnover (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    strategy = st.selectbox("Strategy to review", ["MACD crossover", "Bollinger Bands", "My strategy"])
    if strategy == "My strategy":
        higher_timeframe = st.selectbox("Order-block timeframe", ["15m", "1h", "4h"], index=2, help="For true higher-timeframe confluence, choose an interval above your chart interval.")
        risk_reward_label = st.selectbox("Risk-to-reward", ["1:1", "1:2"], index=1)
        risk_reward = int(risk_reward_label.split(":")[1])
        swing_lookback = st.slider("Swing liquidity lookback", min_value=5, max_value=30, value=10, step=1)
    refresh = st.button("🔄 Refresh market data", use_container_width=True)

if refresh:
    fetch_candles.clear()

bars_per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}[timeframe]

try:
    with st.spinner("Loading candles from Kraken…"):
        candles = fetch_candles(symbol, timeframe, limit)
    data = add_indicators(candles)
    fee_rate = fee_pct / 100.0
    if strategy == "My strategy":
        with st.spinner(f"Checking {higher_timeframe} order blocks and liquidity sweeps…"):
            htf_candles = fetch_candles(symbol, higher_timeframe, min(max(limit, 100), 720))
        data = add_my_strategy(
            data, htf_candles, swing_lookback=swing_lookback,
            risk_reward=risk_reward, base_timeframe=timeframe,
            higher_timeframe_name=higher_timeframe,
        )
        pos_col = "my_position"
    else:
        pos_col = "macd_position" if strategy == "MACD crossover" else "bb_position"
    results, metrics = run_backtest(data, pos_col, fee_rate, bars_per_year)

    latest = data.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"{latest['close']:,.4f} USD" if symbol.endswith("/USD") else f"{latest['close']:,.4f}")
    c2.metric("Candles loaded", f"{len(data):,}")
    c3.metric("Strategy", strategy)
    if strategy == "MACD crossover":
        if latest["macd_position"] > 0:
            state = "LONG"
        elif latest["macd_position"] < 0:
            state = "SHORT"
        else:
            state = "NO POSITION"
    elif strategy == "Bollinger Bands":
        state = "LONG EVENT" if latest["bb_event"] > 0 else ("SHORT EVENT" if latest["bb_event"] < 0 else "NO NEW SIGNAL")
    else:
        state = "BUY" if latest["my_signal"] > 0 else ("SELL" if latest["my_signal"] < 0 else "NO NEW SIGNAL")
    c4.metric("Latest signal", state)
    st.caption(
        "Latest completed candle opened at "
        f"{data.index[-1].strftime('%Y-%m-%d %H:%M UTC')}. "
        "The currently forming candle is excluded from indicators and backtesting."
    )

    st.subheader("Price and indicators")
    price_chart = data[["close", "bb_mid", "bb_upper", "bb_lower"]].rename(
        columns={"close": "Close", "bb_mid": "BB middle", "bb_upper": "BB upper", "bb_lower": "BB lower"}
    )
    if strategy == "My strategy":
        price_chart["BUY signal"] = data["low"].where(data["my_signal"] > 0)
        price_chart["SELL signal"] = data["high"].where(data["my_signal"] < 0)
    st.line_chart(price_chart, height=350)

    if strategy == "My strategy":
        st.subheader("My strategy — latest setup")
        recent_signals = data.loc[data["my_signal"] != 0, ["my_signal", "my_entry", "my_stop", "my_target", "my_ob_midpoint"]].tail(10).copy()
        if recent_signals.empty:
            st.info("No confirmed My strategy setup in the loaded candles. Try another pair, interval, or more candles.")
        else:
            recent_signals["Signal"] = recent_signals["my_signal"].map({1: "BUY", -1: "SELL"})
            recent_signals = recent_signals[["Signal", "my_entry", "my_stop", "my_target", "my_ob_midpoint"]].rename(columns={
                "my_entry": "Entry", "my_stop": "Stop-loss", "my_target": "Take-profit", "my_ob_midpoint": "HTF OB 50% midpoint"
            })
            st.dataframe(recent_signals.sort_index(ascending=False), use_container_width=True)
        st.caption(f"Risk-to-reward selected: 1:{risk_reward}. Stop and target are calculated from the setup invalidation level. The current portfolio backtest does not yet simulate stop-loss/take-profit exits; treat its performance metrics as provisional, not as an RR-specific result.")
    else:
        st.subheader("MACD")
        st.line_chart(data[["macd", "macd_signal"]].rename(
            columns={"macd": "MACD", "macd_signal": "Signal line"}
        ), height=220)

    st.subheader("Backtest results")
    metric_cols = st.columns(3)
    for i, (name, value) in enumerate(metrics.items()):
        metric_cols[i % 3].metric(name, value)
    st.line_chart(results[["equity", "buy_hold"]].rename(
        columns={"equity": "Strategy equity (starting at 1.0)", "buy_hold": "Buy & hold (starting at 1.0)"}
    ), height=300)

    with st.expander("View latest candle rows"):
        st.dataframe(data.tail(30), use_container_width=True)

    csv = results.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download backtest results (CSV)",
        data=csv,
        file_name=f"{symbol.replace('/', '_').lower()}_{timeframe}_backtest.csv",
        mime="text/csv",
    )
    if strategy == "My strategy":
        st.caption("My strategy uses mechanical approximations: swing sweeps, wick/body rejection, a second sweep, displacement through a prior opposite-colour candle, and a completed higher-timeframe order-block midpoint. Signals may differ from discretionary chart reading. No trades are placed.")
    st.caption(
        "Model assumptions: signal is applied on the next candle; fee is approximated from changes in exposure; "
        "slippage, funding, spread, and real order execution are not modeled. The Bollinger rule is a simple "
        "one-candle event exposure and should be refined before strategy comparison."
    )
except Exception as exc:
    st.error("Could not load or analyze market data.")
    st.info("Check your internet connection, try another pair/interval, or refresh the app. The exchange may temporarily limit requests.")
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
