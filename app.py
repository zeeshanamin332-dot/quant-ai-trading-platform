import numpy as np
import pandas as pd
import streamlit as st
import ccxt
import plotly.graph_objects as go
from backtest import run_backtest
from my_strategy import add_my_strategy, simulate_my_trades
from agents import ask_ai_agent, fetch_crypto_news, trades_to_context

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
    limit = st.slider("Number of candles", min_value=100, max_value=5000, value=1000, step=100)
    fee_pct = st.number_input("Estimated fee per unit of turnover (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    strategy = st.selectbox("Strategy to review", ["MACD crossover", "Bollinger Bands", "My strategy"])
    if strategy == "My strategy":
        higher_timeframe = st.selectbox("Order-block timeframe", ["15m", "1h", "4h", "1d"], index={"1h": 2, "4h": 3, "1d": 3}[timeframe], help="Choose an interval above the chart interval where possible. The app has no interval above 1d.")
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

    st.subheader("Candlestick chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["open"], high=data["high"], low=data["low"], close=data["close"],
        name=f"{symbol} OHLC",
    ))
    if strategy == "Bollinger Bands":
        for col, label in [("bb_mid", "BB middle"), ("bb_upper", "BB upper"), ("bb_lower", "BB lower")]:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=label, line={"width": 1}))
    if strategy == "MACD crossover":
        buys = data["macd_event"] > 0
        sells = data["macd_event"] < 0
        fig.add_trace(go.Scatter(x=data.index[buys], y=data.loc[buys, "low"] * 0.999, mode="markers", name="MACD buy", marker={"symbol": "triangle-up", "size": 11}))
        fig.add_trace(go.Scatter(x=data.index[sells], y=data.loc[sells, "high"] * 1.001, mode="markers", name="MACD sell", marker={"symbol": "triangle-down", "size": 11}))
    trades = pd.DataFrame()
    if strategy == "My strategy":
        buys = data["my_signal"] > 0
        sells = data["my_signal"] < 0
        fig.add_trace(go.Scatter(x=data.index[buys], y=data.loc[buys, "low"], mode="markers", name="BUY setup", marker={"symbol": "triangle-up", "size": 13, "color": "#16a34a"}))
        fig.add_trace(go.Scatter(x=data.index[sells], y=data.loc[sells, "high"], mode="markers", name="SELL setup", marker={"symbol": "triangle-down", "size": 13, "color": "#dc2626"}))
        trades = simulate_my_trades(data)
        if not trades.empty:
            entered = trades["Entry time"].notna()
            fig.add_trace(go.Scatter(x=trades.loc[entered, "Entry time"], y=trades.loc[entered, "Entry"], mode="markers", name="Trade entry", marker={"symbol": "circle", "size": 9, "color": "#2563eb"}))
            tp = trades["Outcome"] == "TP HIT"
            sl = trades["Outcome"] == "SL HIT"
            opened = trades["Outcome"] == "OPEN"
            fig.add_trace(go.Scatter(x=trades.loc[tp, "Exit time"], y=trades.loc[tp, "Exit price"], mode="markers+text", text=["TP HIT"] * int(tp.sum()), textposition="top center", name="Take-profit hit", marker={"symbol": "star", "size": 14, "color": "#16a34a"}))
            fig.add_trace(go.Scatter(x=trades.loc[sl, "Exit time"], y=trades.loc[sl, "Exit price"], mode="markers+text", text=["SL HIT"] * int(sl.sum()), textposition="bottom center", name="Stop-loss hit", marker={"symbol": "x", "size": 13, "color": "#dc2626"}))
            fig.add_trace(go.Scatter(x=trades.loc[opened, "Entry time"], y=trades.loc[opened, "Entry"], mode="markers", name="Trade still open", marker={"symbol": "circle-open", "size": 12, "color": "#d97706"}))
            if not trades.empty:
                latest_trade = trades.iloc[-1]
                for col, color, dash in [("Entry", "#2563eb", "solid"), ("Stop-loss", "#dc2626", "dash"), ("Take-profit", "#16a34a", "dash")]:
                    fig.add_hline(y=float(latest_trade[col]), line_dash=dash, line_color=color, annotation_text=f"Latest {col}: {float(latest_trade[col]):,.4f}", annotation_position="right")
    fig.update_layout(
        height=620, xaxis_rangeslider_visible=False, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
    )
    fig.update_yaxes(title_text="Price")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "responsive": True, "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"]})

    if strategy == "My strategy":
        st.subheader("My strategy — projected trade levels and outcomes")
        if trades.empty:
            st.info("No confirmed setup produced a simulated trade in the loaded candles. Try another pair, interval, or more candles.")
        else:
            latest_trade = trades.iloc[-1]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Latest trade", f"{latest_trade['Side']} / {latest_trade['Outcome']}")
            p2.metric("Entry (projected)", f"{latest_trade['Entry']:,.4f}")
            p3.metric("Stop-loss (projected)", f"{latest_trade['Stop-loss']:,.4f}")
            p4.metric("Take-profit (projected)", f"{latest_trade['Take-profit']:,.4f}")
            summary = trades.copy()
            summary["Signal time"] = pd.to_datetime(summary["Signal time"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
            summary["Entry time"] = pd.to_datetime(summary["Entry time"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
            summary["Exit time"] = pd.to_datetime(summary["Exit time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
            st.dataframe(summary.sort_values("Entry time", ascending=False), use_container_width=True, hide_index=True)
            done = trades[trades["Outcome"].isin(["TP HIT", "SL HIT"])]
            wins = int((done["Outcome"] == "TP HIT").sum())
            losses = int((done["Outcome"] == "SL HIT").sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Take-profits hit", wins)
            m2.metric("Stop-losses hit", losses)
            m3.metric("Closed trades", len(done))
            m4.metric("Win rate", f"{wins / len(done) * 100:.1f}%" if len(done) else "—")
        st.caption(f"Risk-to-reward selected: 1:{risk_reward}. Entries are simulated at the next candle open after a confirmed signal. If TP and SL are both inside the same candle, SL is assumed first as a conservative rule. OPEN trades have not hit either level yet. These are OHLC-based simulations, not actual executed trades.")
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
