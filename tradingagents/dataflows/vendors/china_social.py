"""China A-share attention and sentiment signals backed by AKShare.

The public analyst contract remains unchanged: this module turns several
China-focused sources into one prompt-ready text block. Most upstream
interfaces expose only a *current* snapshot, so they are deliberately not
queried for historical analysis dates to avoid look-ahead bias.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any

import pandas as pd


def _bare_code(ticker: str) -> str:
    upper = ticker.strip().upper()
    for suffix in (".SH", ".SS", ".SZ"):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def _ak_symbol(ticker: str) -> str:
    upper = ticker.strip().upper()
    prefix = "SZ" if upper.endswith(".SZ") else "SH"
    return prefix + _bare_code(ticker)


def _normalise_code(value: Any) -> str:
    text = str(value).strip().upper()
    if text.endswith((".SH", ".SS", ".SZ")):
        text = text.rsplit(".", 1)[0]
    if text.startswith(("SH", "SZ")):
        text = text[2:]
    return text.zfill(6) if text.isdigit() else text


def _matching_row(frame: pd.DataFrame | None, ticker: str) -> tuple[int, pd.Series] | None:
    if frame is None or frame.empty:
        return None
    target = _normalise_code(_bare_code(ticker))
    code_columns = [
        column
        for column in ("代码", "股票代码", "证券代码", "名称/代码", "symbol", "code")
        if column in frame.columns
    ]
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        for column in code_columns:
            value = row.get(column)
            # Baidu can return values such as "宁德时代 300750".
            if target and target in _normalise_code(value):
                return position, row
    return None


def _clean(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _fields(row: pd.Series, names: tuple[str, ...]) -> str:
    values = [f"{name}: {_clean(row.get(name))}" for name in names if _clean(row.get(name))]
    return "; ".join(values)


def _company_name(ak: Any, ticker: str) -> str:
    try:
        frame = ak.stock_individual_info_em(symbol=_bare_code(ticker))
        if frame is None or frame.empty:
            return ""
        if {"item", "value"}.issubset(frame.columns):
            for _, row in frame.iterrows():
                if _clean(row.get("item")) in {"股票简称", "名称", "股票名称"}:
                    return _clean(row.get("value"))
        for column in ("股票简称", "名称", "股票名称"):
            if column in frame.columns:
                return _clean(frame.iloc[0].get(column))
    except Exception:
        pass
    return ""


def _eastmoney_popularity(ak: Any, ticker: str) -> str:
    source = "East Money popularity"
    try:
        frame = ak.stock_hot_rank_em()
        match = _matching_row(frame, ticker)
        if match is None:
            return f"- {source}: unavailable (ticker not present in the current ranking)"
        position, row = match
        summary = _fields(row, ("当前排名", "股票名称", "最新价", "涨跌幅"))
        if not summary:
            summary = f"list position: {position}"

        extras: list[str] = []
        try:
            detail = ak.stock_hot_rank_detail_em(symbol=_ak_symbol(ticker))
            if detail is not None and not detail.empty:
                recent = detail.tail(1).iloc[0]
                detail_text = _fields(
                    recent,
                    ("时间", "排名", "新晋粉丝", "铁杆粉丝", "粉丝增长率"),
                )
                if detail_text:
                    extras.append(f"latest trend: {detail_text}")
        except Exception:
            pass

        try:
            keywords = ak.stock_hot_keyword_em(symbol=_ak_symbol(ticker))
            if keywords is not None and not keywords.empty:
                keyword_column = next(
                    (
                        column
                        for column in ("概念名称", "关键词", "keyword")
                        if column in keywords.columns
                    ),
                    None,
                )
                if keyword_column:
                    top_keywords = [
                        _clean(value)
                        for value in keywords[keyword_column].head(5).tolist()
                        if _clean(value)
                    ]
                    if top_keywords:
                        extras.append("hot keywords: " + ", ".join(top_keywords))
        except Exception:
            pass

        suffix = f"; {'; '.join(extras)}" if extras else ""
        return f"- {source}: {summary}{suffix}"
    except Exception as exc:
        return f"- {source}: unavailable ({type(exc).__name__})"


def _xueqiu_discussion(ak: Any, ticker: str) -> str:
    source = "Xueqiu discussion ranking"
    try:
        frame = ak.stock_hot_tweet_xq(symbol="最热门")
        match = _matching_row(frame, ticker)
        if match is None:
            return f"- {source}: unavailable (ticker not present in the current ranking)"
        position, row = match
        summary = _fields(row, ("股票简称", "关注", "最新价"))
        position_text = f"rank/list position: {position}"
        return f"- {source}: {position_text}" + (f"; {summary}" if summary else "")
    except Exception as exc:
        return f"- {source}: unavailable ({type(exc).__name__})"


def _baidu_vote(ak: Any, ticker: str) -> str:
    source = "Baidu Finance bullish/bearish vote"
    try:
        frame = ak.stock_zh_vote_baidu(symbol=_bare_code(ticker), indicator="股票")
        if frame is None or frame.empty:
            return f"- {source}: unavailable (empty response)"
        row = frame.iloc[0]
        summary = _fields(row, ("周期", "看涨", "看跌", "看涨比例", "看跌比例"))
        return f"- {source}: {summary or 'unavailable (no vote fields)'}"
    except Exception as exc:
        return f"- {source}: unavailable ({type(exc).__name__})"


def _weibo_attention(ak: Any, ticker: str, company_name: str) -> str:
    source = "Weibo 7-day attention"
    if not company_name:
        return f"- {source}: unavailable (company name could not be resolved)"
    try:
        frame = ak.stock_js_weibo_report(time_period="CNDAY7")
        if frame is None or frame.empty:
            return f"- {source}: unavailable (empty response)"
        name_column = next(
            (column for column in ("name", "名称", "股票名称") if column in frame.columns),
            None,
        )
        if not name_column:
            return f"- {source}: unavailable (name field missing)"
        matches = frame[
            frame[name_column].astype(str).str.contains(company_name, regex=False, na=False)
        ]
        if matches.empty:
            return f"- {source}: unavailable (ticker not present in the current ranking)"
        row = matches.iloc[0]
        rank = int(matches.index[0]) + 1 if isinstance(matches.index[0], int) else ""
        rank_text = f"rank/list position: {rank}" if rank else "present in ranking"
        rate = _clean(row.get("rate", row.get("热度")))
        return f"- {source}: {rank_text}" + (f"; attention: {rate}" if rate else "")
    except Exception as exc:
        return f"- {source}: unavailable ({type(exc).__name__})"


def fetch_china_social_sentiment(ticker: str, analysis_date: str) -> str:
    """Return current China A-share social/attention signals.

    The signature is intentionally small and stable for use by the existing
    sentiment analyst. Current-snapshot endpoints are called only when
    ``analysis_date`` is today.
    """
    try:
        requested_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
    except ValueError:
        return (
            "## China A-share social and attention signals\n"
            f"- unavailable: invalid analysis date {analysis_date!r}"
        )

    if requested_date != date.today():
        date_context = "historical" if requested_date < date.today() else "future"
        return (
            "## China A-share social and attention signals\n"
            f"Analysis date: {analysis_date}\n"
            "- unavailable: AKShare popularity, discussion, vote, and attention "
            "interfaces provide current snapshots rather than point-in-time history. "
            f"They were not queried for this {date_context} analysis to prevent "
            "look-ahead bias or date misrepresentation."
        )

    import akshare as ak

    company_name = _company_name(ak, ticker)
    jobs: tuple[Callable[[], str], ...] = (
        lambda: _eastmoney_popularity(ak, ticker),
        lambda: _xueqiu_discussion(ak, ticker),
        lambda: _baidu_vote(ak, ticker),
        lambda: _weibo_attention(ak, ticker, company_name),
    )
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        signals = list(executor.map(lambda job: job(), jobs))

    return (
        "## China A-share social and attention signals\n"
        f"Observed at: {analysis_date} (current snapshots)\n"
        "Important: popularity/attention measures participation, not bullishness. "
        "Use the Baidu vote as the explicit directional signal and do not infer "
        "positive sentiment from a high rank alone.\n" + "\n".join(signals)
    )
