import numpy as np
import pandas as pd
import streamlit as st
import ccxt

st.set_page_config(
    page_title="Quant AI Trading Platform",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quant AI Trading Platform")
st.caption("Crypto research dashboard • Historical candles • MACD • Bollinger Bands • Backtesting")
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
    if df.empty or (df["close"] <= 0).any():
        raise ValueError("The candle data was empty or invalid.")
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

def run_backtest(data: pd.DataFrame, position_col: str, fee_rate: float, bars_per_year: int):
    result = data.copy()
    # Signals formed on candle close are applied from the following candle.
    result["position"] = result[position_col].shift(1).fillna(0)
    result["market_return"] = result["close"].pct_change().fillna(0)
    result["turnover"] = result["position"].diff().abs()
    if len(result):
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

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Crypto pair", ["BTC/USD", "ETH/USD", "SOL/USD", "BTC/USDT", "ETH/USDT"])
    timeframe = st.selectbox("Candle interval", ["1h", "4h", "1d"])
    limit = st.slider("Number of candles", min_value=100, max_value=720, value=500, step=100)
    fee_pct = st.number_input("Estimated fee per unit of turnover (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    strategy = st.selectbox("Strategy to review", ["MACD crossover", "Bollinger Bands"])
    refresh = st.button("🔄 Refresh market data", use_container_width=True)

if refresh:
    fetch_candles.clear()

bars_per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}[timeframe]

try:
    with st.spinner("Loading candles from Kraken…"):
        candles = fetch_candles(symbol, timeframe, limit)
    data = add_indicators(candles)
    fee_rate = fee_pct / 100.0
    pos_col = "macd_position" if strategy == "MACD crossover" else "bb_position"
    results, metrics = run_backtest(data, pos_col, fee_rate, bars_per_year)

    latest = data.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"{latest['close']:,.4f} USD" if symbol.endswith("/USD") else f"{latest['close']:,.4f}")
    c2.metric("Candles loaded", f"{len(data):,}")
    c3.metric("Strategy", strategy)
    if strategy == "MACD crossover":
        state = "LONG" if latest["macd_position"] > 0 else "SHORT / EXIT"
    else:
        state = "LONG EVENT" if latest["bb_event"] > 0 else ("SHORT EVENT" if latest["bb_event"] < 0 else "NO NEW SIGNAL")
    c4.metric("Latest signal", state)

    st.subheader("Price and indicators")
    price_chart = data[["close", "bb_mid", "bb_upper", "bb_lower"]].rename(
        columns={"close": "Close", "bb_mid": "BB middle", "bb_upper": "BB upper", "bb_lower": "BB lower"}
    )
    st.line_chart(price_chart, height=350)

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
    st.caption(
        "Model assumptions: signal is applied on the next candle; fee is approximated from changes in exposure; "
        "slippage, funding, spread, and real order execution are not modeled. The Bollinger rule is a simple "
        "one-candle event exposure and should be refined before strategy comparison."
    )
except Exception as exc:
    st.error("Could not load or analyze market data.")
    st.info("Check your internet connection, try another pair/interval, or refresh the app. The exchange may temporarily limit requests.")
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
