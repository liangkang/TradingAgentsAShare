"""Sentiment analyst: one sentiment report from three sources.

The node fetches its sources before calling the model and puts them in the
prompt, so the model reports on data it was given rather than inventing posts:

  1. News headlines: Yahoo Finance
  2. StockTwits messages: the cashtag stream, with Bullish/Bearish tags
  3. Reddit posts: r/wallstreetbets, r/stocks, r/investing, or crypto communities for a crypto pair

A China A-share ticker swaps the US-centric StockTwits and Reddit for China
social and attention signals (East Money popularity, Xueqiu discussion rank,
Baidu bullish/bearish votes, Weibo attention), with its news from AKShare-backed
East Money/CNInfo through the configured news vendor.

Each source is trimmed to the analysis window. With a TypeSafe key, the social
posts are screened by Jev first (see post_screen). These feeds serve recent items
and are not archived, so a historical run's sentiment inputs are not
point-in-time.

The report is a SentimentReport through structured output where the provider
supports it and free text otherwise, so the band, score and confidence header
reads the same across providers.
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    is_cn_ticker,
)
from tradingagents.agents.post_screen import jev_screen
from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.tools import get_news
from tradingagents.dataflows.vendors.china_social import fetch_china_social_sentiment
from tradingagents.dataflows.vendors.reddit import (
    CRYPTO_SUBREDDITS,
    DEFAULT_SUBREDDITS,
    fetch_reddit_posts,
    subreddits_for,
)
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def _news_source_label(ticker: str) -> str:
    """Return the human-readable news source for the ticker market."""
    return "akshare (East Money) + CNInfo" if is_cn_ticker(ticker) else "Yahoo Finance"


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)

        # StockTwits and Reddit are US-centric: for a China A-share they carry
        # negligible signal and the requests often time out from behind certain
        # firewalls, so that market reads China social/attention signals instead.
        china_social_block = None
        stocktwits_block = reddit_block = ""
        subreddits = DEFAULT_SUBREDDITS
        if is_cn_ticker(ticker):
            china_social_block = fetch_china_social_sentiment(ticker, end_date)
        else:
            # Pass the analysis window so a historical run trims social posts to it
            # instead of leaking today's chatter into a backtest (#1220).
            screen = jev_screen(ticker)
            stocktwits_block = fetch_stocktwits_messages(
                ticker, limit=30, start_date=start_date, end_date=end_date, screen=screen
            )
            subreddits = subreddits_for(ticker)
            reddit_block = fetch_reddit_posts(
                ticker, subreddits, start_date=start_date, end_date=end_date, screen=screen
            )

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            subreddits=subreddits,
            china_social_block=china_social_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what your tools support; another agent decides the trade."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
    china_social_block: str | None = None,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    if is_cn_ticker(ticker):
        data_sources_intro = (
            "China-market news and social/attention signals. StockTwits and "
            "Reddit are US-centric and are skipped for this China A-share ticker"
        )
        social_sections = f"""### China A-share social and attention signals
Current-snapshot signals from East Money popularity, Xueqiu discussion,
Baidu Finance bullish/bearish votes, and Weibo attention. Individual sources
may be unavailable, especially for a historical analysis date.

<start_of_china_social>
{china_social_block or "<unavailable — China social data was not collected>"}
<end_of_china_social>"""
        analysis_guidance = """1. **Separate attention from direction.** East Money,
Xueqiu, and Weibo rankings show participation or popularity; a high rank is
not evidence of bullishness. Treat Baidu's bullish/bearish vote as the explicit
directional social signal.

2. **Look for cross-source divergences.** Compare directional votes with
company-news framing and with attention. High attention plus negative news is
different from high attention plus positive news.

3. **Do not invent posts or opinions.** These ranking interfaces do not provide
a representative corpus of user messages. Describe only the supplied ranks,
votes, keywords, and news evidence.

4. **Respect point-in-time limits.** If the China social block says current
snapshots were unavailable for a historical date, do not use present-day
rankings as historical evidence and lower confidence accordingly.

5. **Distinguish opinion from event.** Votes and popularity are audience
signals; company announcements and news are events and should normally carry
more evidentiary weight.

6. **Be honest about data limits.** Flag every unavailable source and reduce
confidence when directional vote data or timely news is missing.

7. **Identify recurring narrative themes, catalysts, and risks** across news,
hot keywords, votes, and attention without treating sentiment as a price call.

8. **Past sentiment is not predictive.** Frame conclusions as one signal for
the trader to weigh alongside fundamentals and technicals."""
    else:
        data_sources_intro = "three complementary data sources"
        social_sections, analysis_guidance = _us_social_sections(
            stocktwits_block, reddit_block, subreddits
        )

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on {data_sources_intro} that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — {_news_source_label(ticker)}, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

{social_sections}

## How to analyze this data (best practices)

{analysis_guidance}

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


def _us_social_sections(
    stocktwits_block: str, reddit_block: str, subreddits: tuple[str, ...]
) -> tuple[str, str]:
    """The StockTwits/Reddit data sections and their analysis guidance."""
    if subreddits == DEFAULT_SUBREDDITS:
        character = "r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term"
    elif len(subreddits) > len(CRYPTO_SUBREDDITS):
        character = f"r/{subreddits[0]} leans toward the coin's holders; r/CryptoCurrency and r/CryptoMarkets are broader"
    else:
        character = "r/CryptoCurrency and r/CryptoMarkets are broad crypto communities"
    social_sections = f"""### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — {", ".join(f"r/{sub}" for sub in subreddits)} (past 7 days)
Community discussion, without vote or comment counts. Subreddit character matters ({character}).

<start_of_reddit>
{reddit_block}
<end_of_reddit>"""
    analysis_guidance = """1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone. A block headed "Screened by Jev" has had off-topic posts removed; its stance count is a classifier's read of every on-topic post fetched, labelled or not, of which the posts listed are a sample. Read it alongside the user tags.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Read Reddit posts for substance.** The feed carries no vote or comment counts, so judge a post by its body excerpt, not its title alone, and do not infer engagement.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call."""
    return social_sections, analysis_guidance
