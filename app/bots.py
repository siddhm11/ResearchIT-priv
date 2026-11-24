"""
Who is asking: a person, or a crawler, answer engine or link preview.

Why it exists (docs/AEO-PLAN.md, step 1). Public pages carry ranked-surface
links (/p/{id}?qid=...), the front page writes exposure rows and every page
mints a reader cookie. A crawler following those links would write clicks,
exposures and phantom readers into the log the reranker will one day train on
(CLAUDE.md §3.11). And a fetch by ChatGPT-User or Claude-User is the earliest
sign that an AI answer used a page, which nothing counted.

Routing and counting only, never access control: a User-Agent can say anything.
Every class gets the same HTML a reader gets (no cloaking); a bot just leaves
no trace in reader data and never triggers a model call.

Classes:
    search      builds a search index an answer engine retrieves from
    user_fetch  fetched live because a person asked an assistant
    training    collects pages for model training
    preview     draws a link preview (reads only the head tags)
    other       anything else that calls itself a bot, crawler or spider
"""
from __future__ import annotations

import re

# Ordered: the first class with a matching token wins. Tokens are matched
# case-insensitively as substrings, so "Googlebot-Image" counts as Googlebot.
# Sources: developers.openai.com/api/docs/bots, support.claude.com/en/articles/8896518,
# docs.perplexity.ai (crawlers), support.claude.com/en/articles/18234133 (Claude-Agent).
CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("user_fetch", ("ChatGPT-User", "Claude-User", "Claude-Agent", "Perplexity-User",
                    "MistralAI-User", "DuckAssistBot")),
    ("search", ("OAI-SearchBot", "Claude-SearchBot", "PerplexityBot", "Googlebot", "Google-InspectionTool",
                "bingbot", "BingPreview", "Applebot", "DuckDuckBot", "YandexBot", "Baiduspider",
                "Amazonbot", "YouBot", "PetalBot", "Bravebot")),
    ("training", ("GPTBot", "ClaudeBot", "anthropic-ai", "CCBot", "Google-Extended", "Applebot-Extended",
                  "Bytespider", "meta-externalagent", "cohere-ai", "Diffbot", "Timpibot")),
    ("preview", ("Slackbot", "Twitterbot", "facebookexternalhit", "LinkedInBot", "Discordbot",
                 "TelegramBot", "WhatsApp", "redditbot", "Mastodon", "Bluesky", "Embedly", "iframely")),
)

# "bot" as a product token ("SemrushBot/7", "MJ12bot;") or a word of its own,
# never inside a word: "CUBOT P30" is an Android phone. HeadlessChrome is not
# matched either: it is how browser tests (and some readers' tools) run.
_GENERIC = re.compile(r"bot[/;)_-]|(?<![a-z])bot\b|crawl|spider|slurp|fetcher|python-requests|"
                      r"curl/|wget/|httpx|go-http-client|scrapy|axios/", re.IGNORECASE)
_LOWER = tuple((cls, tuple((t, t.lower()) for t in tokens)) for cls, tokens in CLASSES)


def identify(user_agent: str | None) -> tuple[str, str] | None:
    """(class, name) for a known bot, ("other", "other") for an unknown one, None for a person.

    The name is the matched token as written above, so counts group by product
    ("Googlebot", not every version string). An empty User-Agent is a script.
    """
    ua = (user_agent or "").strip()
    if not ua:
        return ("other", "empty")
    low = ua.lower()
    for cls, tokens in _LOWER:
        for name, token in tokens:
            if token in low:
                return (cls, name)
    if _GENERIC.search(ua):
        return ("other", "other")
    return None


def classify(user_agent: str | None) -> str | None:
    """The bot class, or None for a person."""
    found = identify(user_agent)
    return found[0] if found else None


HEAD_FLAG = "researchit.head"     # set on the scope by main.HeadAsGet


def is_bot(request) -> bool:
    """True when this request comes from anything but a person's browser.

    A HEAD request counts too: it renders the page like a GET (main.HeadAsGet)
    but nobody reads the result, so it must not log clicks or exposures either.
    """
    if getattr(request, "scope", {}).get(HEAD_FLAG):
        return True
    return classify(request.headers.get("user-agent")) is not None


# Referrers that mean a person clicked a link inside an AI answer.
AI_REFERRERS: tuple[tuple[str, str], ...] = (
    ("chatgpt.com", "chatgpt"), ("chat.openai.com", "chatgpt"), ("claude.ai", "claude"),
    ("perplexity.ai", "perplexity"), ("gemini.google.com", "gemini"), ("copilot.microsoft.com", "copilot"),
    ("bing.com/chat", "copilot"), ("chat.mistral.ai", "mistral"), ("meta.ai", "meta"),
    ("you.com", "you"), ("phind.com", "phind"), ("kagi.com", "kagi"),
)


def ai_referral(request) -> str | None:
    """The answer engine a visit came from, by utm_source or Referer, or None."""
    utm = (request.query_params.get("utm_source") or "").lower()
    ref = (request.headers.get("referer") or "").lower()
    for host, name in AI_REFERRERS:
        if host in utm or ("//" + host) in ref or ("." + host) in ref:
            return name
    return None


def page_type(path: str) -> str:
    """A coarse page type for counts: paper, trending, lasting, topic, report, collection, home, other."""
    if path == "/":
        return "home"
    if path.startswith("/p/"):
        return "paper"
    if path.startswith("/trending/lasting"):
        return "lasting"
    if path.startswith("/trending/topics/") or path.startswith("/topics/"):
        return "topic"
    if path.startswith("/trending/reports") or path[10:14].isdigit() and path.startswith("/trending/"):
        return "report"
    if path.startswith("/trending"):
        return "trending"
    if path.startswith("/collections"):
        return "collection"
    if path in ("/robots.txt", "/sitemap.xml") or path.startswith("/sitemaps/"):
        return "crawl_file"
    if path in ("/about", "/privacy", "/compare"):
        return "about"
    return "other"
