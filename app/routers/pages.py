"""
/about and /privacy: the facts an assistant checks before it names a tool,
and the policy both agent directories require (docs/AEO-PLAN.md).

ChatGPT's search for a shortlisted tool reads the tool's own site and drops
tools whose site lacks key facts (Seer Interactive, Nov 2026). These pages say
what ResearchIT is, what it covers, what is free and where its numbers come
from, in plain sentences an answer engine can lift. Every number on them is
read live from the services that own it, never typed in.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import config
from app.templates_env import corpus_label, templates

router = APIRouter()

SPACE_URL = "https://huggingface.co/spaces/siddhm11/ResearchIT"


def _facts() -> dict:
    """Live numbers for the About page; a source that is not loaded is left out."""
    facts: dict = {"corpus": corpus_label()}
    try:
        from app import hn_svc
        stats = hn_svc.stats()
        if stats.get("lasting"):
            facts["lasting"] = stats["lasting"]
            facts["hn_links"] = stats.get("mentions") or 0
    except Exception as e:
        print(f"[pages] HN facts unavailable: {e}")
    try:
        from app import collections_svc
        facts["collections"] = len(collections_svc.load_all())
    except Exception as e:
        print(f"[pages] collections unavailable: {e}")
    return facts


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    facts = _facts()
    return templates.TemplateResponse(request, "about.html", {
        "facts": facts, "space_url": SPACE_URL, "contact_url": config.CONTACT_URL,
        "og_title": "About ResearchIT",
        "og_description": (f"ResearchIT is a free reading feed and search engine for arXiv: {facts['corpus']} "
                           "papers, updated nightly, no sign-up. How it ranks papers and where its data comes from."),
        "jsonld": {
            "@context": "https://schema.org",
            "@type": "WebApplication",
            "name": "ResearchIT",
            "url": config.PUBLIC_URL + "/",
            "applicationCategory": "ReferenceApplication",
            "operatingSystem": "Web",
            "isAccessibleForFree": True,
            "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
            "description": (f"A personal reading feed and plain-language search for {facts['corpus']} arXiv papers, "
                            "with research-topic trends and the papers Hacker News keeps returning to."),
            "sameAs": [SPACE_URL],
        },
    })


@router.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    return templates.TemplateResponse(request, "privacy.html", {
        "contact_url": config.CONTACT_URL,
        "og_title": "Privacy — ResearchIT",
        "og_description": ("What ResearchIT stores (an anonymous cookie and what you save, pass and open), "
                           "who processes it, how long it is kept and how to remove it."),
    })


# ── /compare ─────────────────────────────────────────────────────────────────
# Checked 24 Nov 2026 from each product's own pages, papers or code (sources in
# each row). Prices nobody publishes are left out, never guessed. Re-check
# quarterly (docs/AEO-PLAN.md, phase 3); change CHECKED when you do.

CHECKED = "24 November 2026"
COMPARE = [
    {"name": "ResearchIT", "free": "Free", "signup": "No", "coverage": "arXiv, about 2.0M papers, new ones nightly",
     "picks": "Learns separate interests from the papers you save; plain-words search", "delivery": "Web",
     "when": "You want a feed of arXiv papers without an account, plus topic trends and Hacker News history",
     "source": "/about"},
    {"name": "Scholar Inbox", "free": "Free (per its project page)", "signup": "Yes, you rate papers to start",
     "coverage": "arXiv, bioRxiv, medRxiv, ChemRxiv and conference proceedings",
     "picks": "A model trained on your thumbs up and down", "delivery": "Web digest and email",
     "when": "You want a daily email and to train it with explicit ratings",
     "source": "https://arxiv.org/abs/2504.08385"},
    {"name": "alphaXiv", "free": "Free, with a paid plan", "signup": "For the personal feed and library",
     "coverage": "arXiv, mostly AI", "picks": "Researchers you follow, a For You feed, custom feeds",
     "delivery": "Web, iOS app, browser extension",
     "when": "You want to read and annotate PDFs and discuss papers with others",
     "source": "https://www.alphaxiv.org"},
    {"name": "Semantic Scholar feeds", "free": "Free", "signup": "Yes", "coverage": "All fields, 214M+ papers",
     "picks": "Papers you add to library folders and rate", "delivery": "Email with one-line summaries, web",
     "when": "Your field is outside arXiv, or you need citation data",
     "source": "https://www.semanticscholar.org/faq/research-feed-alert"},
    {"name": "Hugging Face Daily Papers", "free": "Free", "signup": "Only for the email",
     "coverage": "Papers picked by the community each day, mostly ML", "picks": "Not personalised: upvotes",
     "delivery": "Web, daily email", "when": "You want what the ML community is talking about today, with code and models",
     "source": "https://huggingface.co/papers"},
    {"name": "Paper Espresso", "free": "Free, open source", "signup": "No",
     "coverage": "Hugging Face Daily Papers", "picks": "Not personalised", "delivery": "Web, open dataset",
     "when": "You want AI summaries and longer topic histories of the Daily Papers lists",
     "source": "https://arxiv.org/abs/2604.04562"},
    {"name": "ResearchRabbit", "free": "Free tier; paid plan $10/month billed yearly", "signup": "Yes",
     "coverage": "All fields, 310M+ articles", "picks": "Seed papers you add, through citations",
     "delivery": "Web", "when": "You want to explore outward from papers you already know, by citation",
     "source": "https://www.researchrabbit.ai/pricing"},
    {"name": "Connected Papers", "free": "Free for 5 graphs a month", "signup": "For more graphs",
     "coverage": "All fields", "picks": "Not a feed: a similarity graph around one paper", "delivery": "Web",
     "when": "You want a picture of the work around one paper",
     "source": "https://connectedpapers.substack.com/p/announcing-premium-accounts-multi-22-07-28"},
    {"name": "Google Scholar alerts", "free": "Free", "signup": "An email address; a profile for recommendations",
     "coverage": "All published literature", "picks": "Keywords, authors you follow, your own profile",
     "delivery": "Email", "when": "You need every field and citation alerts on your own papers",
     "source": "https://scholar.google.com/intl/en/scholar/help.html"},
    {"name": "arXiv email listings", "free": "Free", "signup": "An email address",
     "coverage": "arXiv, one category at a time", "picks": "Not personalised: every new paper in the category",
     "delivery": "Daily email", "when": "You must not miss a single paper in a small category",
     "source": "https://info.arxiv.org/help/subscribe.html"},
    {"name": "zotero-arxiv-daily", "free": "Free, open source (you pay for the AI model it calls)",
     "signup": "Set up on GitHub, with Zotero and an email server",
     "coverage": "arXiv, bioRxiv, medRxiv, ChemRxiv", "picks": "Similarity to your Zotero library",
     "delivery": "Daily email", "when": "You already keep your reading in Zotero and can run a GitHub workflow",
     "source": "https://github.com/TideDra/zotero-arxiv-daily"},
    {"name": "arxiv-sanity-lite", "free": "Free, open source", "signup": "You host it yourself",
     "coverage": "The arXiv categories you configure", "picks": "A classifier per tag you create",
     "delivery": "Web, optional email",
     "when": "You want something small you can read and run yourself (unchanged since 2023)",
     "source": "https://github.com/karpathy/arxiv-sanity-lite"},
    {"name": "Elicit", "free": "Free basic plan; Pro $49/month", "signup": "Yes", "coverage": "138M+ papers",
     "picks": "Not a feed: searches and research reports you ask for; scheduled Routines on Pro",
     "delivery": "Web, scheduled email", "when": "You are doing a systematic review and need data pulled into tables",
     "source": "https://elicit.com/pricing"},
    {"name": "Consensus", "free": "Free with monthly limits", "signup": "For more than a few searches",
     "coverage": "400M+ papers", "picks": "Not a feed: answers questions from the literature", "delivery": "Web",
     "when": "You want an evidence-backed answer to a research question",
     "source": "https://consensus.app/.well-known/pricing.md"},
]


@router.get("/compare", response_class=HTMLResponse)
async def compare(request: Request):
    return templates.TemplateResponse(request, "compare.html", {
        "rows": COMPARE, "checked": CHECKED,
        "og_title": "Personalised arXiv feeds and paper tools compared (2026)",
        "og_description": ("Scholar Inbox, alphaXiv, Semantic Scholar, Hugging Face Daily Papers, ResearchRabbit, "
                           "Google Scholar, arXiv listings and more: what each covers, what it costs, whether it "
                           f"needs an account, and when to choose it. Checked {CHECKED}."),
    })
