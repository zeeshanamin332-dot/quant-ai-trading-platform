
import os
from datetime import datetime, timezone
from typing import Any, Optional

import feedparser
import streamlit as st
from groq import Groq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEFAULT_MODEL = "llama-3.3-70b-versatile"

CRYPTO_NEWS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
}

SYSTEM_PROMPT = """
You are a crypto research assistant for an educational trading platform.

Your responsibilities:
- Explain market news and trading concepts clearly.
- Analyze supplied trade records objectively.
- Explain possible reasons a trade succeeded or failed.
- Identify strategy weaknesses, risks, and assumptions.
- Answer general questions about cryptocurrency and technical analysis.

Important rules:
- Never guarantee profits or future price movements.
- Never invent prices, news, indicators, or trade details.
- Distinguish facts from hypotheses.
- If information is missing, clearly state what is missing.
- Historical performance does not guarantee future results.
- Do not claim to execute trades or access live exchange data
  unless that information is explicitly supplied.
- This platform is for research and education, not financial advice.
"""


# --------------------------------------------------
# Groq client and API key
# --------------------------------------------------

def get_groq_api_key() -> Optional[str]:
    """Read the Groq API key from environment variables or Streamlit Secrets."""

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if api_key:
        return str(api_key).strip()

    return None


def get_groq_client() -> Groq:
    """Create a Groq client or raise a helpful configuration error."""

    api_key = get_groq_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your "
            "Streamlit Community Cloud Secrets."
        )

    return Groq(api_key=api_key)


# --------------------------------------------------
# General AI agent
# --------------------------------------------------

def ask_ai_agent(
    task: str,
    question: str,
    context: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> str:
    """
    Ask Groq to complete an AI research task.

    Args:
        task: The purpose of the request.
        question: The user's question or instruction.
        context: Relevant news, market data, or trade records.
        model: Groq-supported model ID.
        temperature: Lower values generally produce more
            consistent responses.
        max_tokens: Maximum generated response length.

    Returns:
        The AI-generated response as a string.
    """

    if not question or not question.strip():
        return "Please enter a question."

    client = get_groq_client()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"CONTEXT:\n"
                f"{context.strip() if context else 'No additional context provided.'}"
                f"\n\nQUESTION:\n{question.strip()}"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content

        if not answer:
            return "The AI model returned an empty response."

        return answer.strip()

    except Exception as exc:
        # Avoid exposing API credentials or sensitive request details.
        error_message = str(exc)

        if "401" in error_message or "authentication" in error_message.lower():
            return (
                "Groq authentication failed. Check that GROQ_API_KEY "
                "is valid and configured correctly."
            )

        if "429" in error_message or "rate limit" in error_message.lower():
            return (
                "The Groq API rate limit was reached. Please wait "
                "a moment and try again."
            )

        return (
            "The AI request failed. Check your Groq model name, "
            "API availability, and account limits, then try again."
        )


# --------------------------------------------------
# Crypto news agent
# --------------------------------------------------

def fetch_crypto_news(limit: int = 15) -> list[dict[str, Any]]:
    """
    Fetch recent entries from public crypto news RSS feeds.

    Returns a list of dictionaries containing source, title,
    link, published time, and summary.

    RSS availability and feed formats may change.
    """

    limit = max(1, min(int(limit), 50))
    news_items: list[dict[str, Any]] = []

    for source, feed_url in CRYPTO_NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                summary = entry.get("summary", "").strip()
                published = (
                    entry.get("published")
                    or entry.get("updated")
                    or "Publication time unavailable"
                )

                if not title:
                    continue

                news_items.append(
                    {
                        "source": source,
                        "title": title,
                        "link": link,
                        "published": published,
                        "summary": summary,
                    }
                )

        except Exception:
            # Continue if one publisher's feed is unavailable.
            continue

    # Prefer entries with parsed publication dates when available.
    def sort_key(item: dict[str, Any]) -> float:
        try:
            parsed = feedparser.parse(
                f"<rss><channel><item>"
                f"<pubDate>{item['published']}</pubDate>"
                f"</item></channel></rss>"
            )

            if parsed.entries:
                date_struct = parsed.entries[0].get("published_parsed")
                if date_struct:
                    from time import mktime
                    return mktime(date_struct)

        except Exception:
            pass

        return 0.0

    news_items.sort(key=sort_key, reverse=True)

    return news_items[:limit]


def analyze_crypto_news(
    news_items: list[dict[str, Any]],
    question: str = (
        "Summarize these headlines, identify possible market themes, "
        "and explain uncertainty. Do not infer price movements "
        "from headlines alone."
    ),
    model: str = DEFAULT_MODEL,
) -> str:
    """Use Groq to summarize and analyze fetched crypto news."""

    if not news_items:
        return (
            "No news items are available. Check your internet connection "
            "or try again later."
        )

    news_context = "\n\n".join(
        [
            (
                f"Source: {item.get('source', 'Unknown')}\n"
                f"Published: {item.get('published', 'Unknown')}\n"
                f"Headline: {item.get('title', 'Untitled')}\n"
                f"Summary: {item.get('summary', 'No summary available')}\n"
                f"Link: {item.get('link', '')}"
            )
            for item in news_items
        ]
    )

    return ask_ai_agent(
        task=(
            "Analyze the supplied cryptocurrency news. Separate "
            "reported information from interpretation. Discuss "
            "potential market relevance without claiming certainty."
        ),
        question=question,
        context=news_context,
        model=model,
    )


# --------------------------------------------------
# Trade data formatting
# --------------------------------------------------

def trades_to_context(
    trades: Any,
    candles: Any = None,
    strategy: str = "Unknown",
) -> str:
    """
    Convert trade records and optional candle data into text
    suitable for an AI prompt.

    Supports pandas DataFrames and common iterable records.
    """

    sections = [
        f"Strategy: {strategy}",
        f"Analysis generated at: {datetime.now(timezone.utc).isoformat()}",
    ]

    if trades is None:
        sections.append("Trade records: None supplied.")
        return "\n\n".join(sections)

    try:
        if hasattr(trades, "empty") and trades.empty:
            sections.append("Trade records: No trades available.")
        elif hasattr(trades, "to_dict"):
            records = trades.to_dict(orient="records")
            sections.append(
                "Trade records:\n" + _format_records(records, max_rows=50)
            )
        else:
            records = list(trades)
            sections.append(
                "Trade records:\n" + _format_records(records, max_rows=50)
            )

    except Exception:
        sections.append("Trade records could not be formatted.")

    if candles is not None:
        try:
            if hasattr(candles, "empty") and candles.empty:
                sections.append("Candle data: No candles available.")
            elif hasattr(candles, "tail"):
                recent = candles.tail(30)

                if hasattr(recent, "to_dict"):
                    records = recent.reset_index().to_dict(orient="records")
                    sections.append(
                        "Recent candles (up to 30):\n"
                        + _format_records(records, max_rows=30)
                    )
            else:
                sections.append(
                    "Candle data was supplied but could not be summarized."
                )

        except Exception:
            sections.append("Candle data could not be formatted.")

    return "\n\n".join(sections)


def _format_records(
    records: list[Any],
    max_rows: int = 50,
) -> str:
    """Format record dictionaries with a prompt-size safeguard."""

    if not records:
        return "No records."

    lines = []

    for index, record in enumerate(records[:max_rows], start=1):
        if isinstance(record, dict):
            line = ", ".join(
                f"{key}={value}"
                for key, value in record.items()
            )
        else:
            line = str(record)

        # Prevent unusually large values from bloating AI requests.
        lines.append(f"{index}. {line[:1200]}")

    if len(records) > max_rows:
        lines.append(
            f"... {len(records) - max_rows} additional records omitted."
        )

    return "\n".join(lines)


# --------------------------------------------------
# Trade review agent
# --------------------------------------------------

def review_trades(
    trades: Any,
    candles: Any = None,
    strategy: str = "Unknown",
    question: str = (
        "Review the trades. Explain what worked, what failed, "
        "and what additional data would be needed to judge the results. "
        "Do not assume that correlation proves the reason for an outcome."
    ),
    model: str = DEFAULT_MODEL,
) -> str:
    """Review historical trades with Groq."""

    context = trades_to_context(
        trades=trades,
        candles=candles,
        strategy=strategy,
    )

    return ask_ai_agent(
        task=(
            "Review historical trading results. Identify observable "
            "patterns, wins and losses, possible explanations, "
            "and risks. Clearly distinguish observations from hypotheses."
        ),
        question=question,
        context=context,
        model=model,
    )


# --------------------------------------------------
# Strategy limitations agent
# --------------------------------------------------

def analyze_strategy_limitations(
    strategy: str,
    strategy_details: str = "",
    backtest_results: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """Identify assumptions, blind spots, and risks in a strategy."""

    context = (
        f"Strategy name: {strategy}\n\n"
        f"Strategy details:\n"
        f"{strategy_details or 'No strategy description supplied.'}\n\n"
        f"Backtest results:\n"
        f"{backtest_results or 'No backtest results supplied.'}"
    )

    question = (
        "Identify this strategy's main limitations, including "
        "overfitting, market-regime changes, fees, slippage, spread, "
        "liquidity, data quality, look-ahead bias, and sample size "
        "where relevant. Explain which limitations are confirmed "
        "by the supplied information and which are only potential risks. "
        "Suggest practical tests to evaluate robustness."
    )

    return ask_ai_agent(
        task="Evaluate a cryptocurrency trading strategy's limitations.",
        question=question,
        context=context,
        model=model,
    )


# --------------------------------------------------
# General Q&A agent
# --------------------------------------------------

def answer_crypto_question(
    question: str,
    context: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """Answer general questions about crypto and trading."""

    return ask_ai_agent(
        task=(
            "Answer the user's cryptocurrency, technical analysis, "
            "backtesting, or platform question clearly. Explain "
            "important assumptions and uncertainty."
        ),
        question=question,
        context=context,
        model=model,
    )
