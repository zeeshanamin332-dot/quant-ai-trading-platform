"""Optional AI helpers and crypto-news RSS retrieval for the Streamlit app."""
from __future__ import annotations

import os
from typing import Any

import feedparser
import pandas as pd
from openai import OpenAI

NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
]


def _api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def fetch_crypto_news(limit: int = 15) -> list[dict[str, str]]:
    """Fetch public RSS headlines; article contents remain at the publisher."""
    items: list[dict[str, str]] = []
    for publisher, url in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:limit]:
                items.append({
                    "Publisher": publisher,
                    "Headline": str(entry.get("title", "Untitled")),
                    "Published": str(entry.get("published", entry.get("updated", "Date not supplied"))),
                    "Summary": str(entry.get("summary", ""))[:600],
                    "URL": str(entry.get("link", "")),
                })
        except Exception:
            continue
    items.sort(key=lambda item: item["Published"], reverse=True)
    return items[:limit]


def ask_ai_agent(task: str, question: str, context: str = "", model: str = "gpt-4o-mini") -> str:
    """Run a focused agent; requires the user's own OpenAI API key."""
    key = _api_key()
    if not key:
        raise RuntimeError(
            "AI agents need an OpenAI API key. Add OPENAI_API_KEY to Streamlit Cloud "
            "App settings → Secrets, or set it as an environment variable locally."
        )
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "You are a cautious crypto-market research assistant, not a financial adviser. "
                "Never claim certainty or invent live facts. Distinguish observed data from hypotheses. "
                "Explain assumptions, missing data, and uncertainty. Do not recommend leverage or guarantee outcomes. "
                + task
            )},
            {"role": "user", "content": f"Context/data:\n{context[:18000]}\n\nRequest:\n{question}"},
        ],
    )
    return response.choices[0].message.content or "The agent returned no text."


def trades_to_context(trades: pd.DataFrame, candles: pd.DataFrame, strategy: str) -> str:
    """Prepare a compact, bounded context for post-trade review."""
    trade_text = trades.tail(20).to_string(index=False) if not trades.empty else "No simulated trades available."
    bars = candles[["open", "high", "low", "close", "volume"]].tail(30).round(6).to_string()
    return f"Strategy: {strategy}\nRecent simulated trades:\n{trade_text}\nRecent OHLCV candles:\n{bars}"
