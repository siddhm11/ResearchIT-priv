"""
Hacker News mentions of arXiv papers, recorded for the "On Hacker News" block
on paper pages and the Lasting list (doc 06 changelog, 2026-11-03).

Where they come from
--------------------
The HN Search API by Algolia (https://hn.algolia.com/api) indexes every story
and comment since 2007 and needs no key. A search returns at most 1,000 hits,
so a time range is split in half until each part holds 1,000 or fewer. Every
comment whose text names arXiv, alphaXiv, ar5iv or Hugging Face Papers and every
story submitted with such a URL is fetched; the regexes below then decide which
papers it actually links. Measured on 3 Nov 2026 over 2007 to that day: about
160,000 hits, 49,436 links to 28,379 papers, in roughly 750 requests.

What is stored (Turso, all keyed so a re-run overwrites instead of adding)
--------------------------------------------------------------------------
  hn_mentions  one row per (HN item, paper): who, when, which thread, and
               whether the item is the story the paper was submitted as
  hn_comments  the plain text of each comment that links a paper, cut to
               TEXT_MAX characters, for labelling how a paper is used later
  hn_threads   each thread a mention sits in: title, points, comment count
  hn_state     fetched_to (the end of the last complete fetch, unix seconds)
               and loaded_at (when it finished), which the app polls

Each run fetches from fetched_to minus OVERLAP_S, so stories still collecting
points and comments edited after posting are read again for three days. The
watermark moves only when every window came back, so a failed window is
fetched again next time. Nothing here ever fails the daily run.

Standard library and httpx only, like the rest of the daily job.
"""
from __future__ import annotations

import datetime as dt
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

SEARCH_BY_DATE = "https://hn.algolia.com/api/v1/search_by_date"
SEARCH = "https://hn.algolia.com/api/v1/search"
FIRST_TS = 1167609600              # 2007-01-01, before HN's first item
OVERLAP_S = 3 * 86400
MAX_HITS = 1000                    # Algolia's ceiling per search
PROBE_FIRST_S = 30 * 86400         # longer windows are counted before being fetched
LOOKUP_BATCH = 100                 # threads looked up per request
WRITE_BATCH = 200                  # rows per INSERT
TEXT_MAX = 1200
ATTRS = ("objectID,author,created_at_i,story_id,comment_text,story_title,story_url,"
         "title,url,points,num_comments")

# (kind, search parameters). Comments are matched on their text, stories on
# their URL only, so a story merely titled "arXiv ..." is not a submission.
QUERIES: tuple[tuple[str, dict], ...] = (
    ("comment", {"query": "arxiv", "tags": "comment"}),
    ("comment", {"query": "alphaxiv", "tags": "comment"}),
    ("comment", {"query": "ar5iv", "tags": "comment"}),
    ("comment", {"query": "huggingface.co/papers", "tags": "comment"}),
    ("story", {"query": "arxiv", "tags": "story", "restrictSearchableAttributes": "url"}),
    ("story", {"query": "alphaxiv", "tags": "story", "restrictSearchableAttributes": "url"}),
    ("story", {"query": "ar5iv", "tags": "story", "restrictSearchableAttributes": "url"}),
    ("story", {"query": "huggingface.co/papers", "tags": "story", "restrictSearchableAttributes": "url"}),
)


# ── Which papers an item links (pure) ────────────────────────────────────────

_HOSTS = (r"(?:www\.)?(?:arxiv\.org|export\.arxiv\.org|alphaxiv\.org|ar5iv\.org|ar5iv\.labs\.arxiv\.org"
          r"|huggingface\.co/papers|arxiv-sanity-lite\.com|arxiv-sanity\.com|semanticscholar\.org/arxiv)")
_NEW = re.compile(_HOSTS + r"/(?:[a-z]+/)*?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_OLD = re.compile(r"(?:arxiv\.org|ar5iv\.org|ar5iv\.labs\.arxiv\.org)/(?:abs|pdf|html)/"
                  r"([a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})", re.I)
_CITE = re.compile(r"\barxiv:\s?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_TAGS = re.compile(r"<[^>]+>")


def _valid_new(aid: str, last_year: int) -> bool:
    """yymm.nnnn ids ran from April 2007; five digits after the dot from 2015."""
    yy, mm = int(aid[:2]), int(aid[2:4])
    if not (1 <= mm <= 12 and 7 <= yy <= last_year % 100):
        return False
    return len(aid.split(".")[1]) == (4 if yy < 15 else 5)


def arxiv_ids(text: str | None, last_year: int) -> list[str]:
    """The arXiv ids `text` links or cites, in order, each once, without a
    version suffix. Old-style ids take their canonical form, archive and
    number: "math.GT/0309136" is stored as "math/0309136", as the corpus has it."""
    if not text:
        return []
    t = html.unescape(text)
    found = [m for m in _NEW.findall(t) if _valid_new(m, last_year)]
    found += [m for m in _CITE.findall(t) if _valid_new(m, last_year)]
    for m in _OLD.findall(t):
        archive, number = m.split("/")
        found.append(f"{archive.split('.')[0].lower()}/{number}")
    return list(dict.fromkeys(found))


def plain(comment_html: str | None) -> str:
    """A comment's HTML as text: paragraphs on their own lines, entities decoded."""
    t = (comment_html or "").replace("<p>", "\n\n")
    t = html.unescape(_TAGS.sub("", t))
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _int(v) -> int | None:
    return v if type(v) is int and v >= 0 else None


def rows_from_hit(kind: str, hit: dict, checked_on: str, last_year: int) -> dict:
    """What one search hit contributes: mention rows, a comment row, and the
    thread as far as the hit itself describes it.

    A story's paper is in its URL (or, rarely, its title); a comment's papers
    are in its text. Items without an id or a time are dropped."""
    item, ts = str(hit.get("objectID") or ""), _int(hit.get("created_at_i"))
    if not item.isdigit() or ts is None:
        return {"mentions": [], "comment": None, "thread": None, "fallback": None}
    author = hit.get("author") or ""
    if kind == "story":
        ids = arxiv_ids(f"{hit.get('url') or ''} {hit.get('title') or ''}", last_year)
        story = item
        thread = (story, " ".join((hit.get("title") or "").split())[:300], (hit.get("url") or "")[:500],
                  _int(hit.get("points")), _int(hit.get("num_comments")), ts, checked_on)
        comment = None
    else:
        ids = arxiv_ids(hit.get("comment_text"), last_year)
        story = str(hit.get("story_id") or "")
        thread = None
        comment = (item, story, plain(hit.get("comment_text"))[:TEXT_MAX], len(ids), checked_on) if ids else None
    if not story.isdigit():
        return {"mentions": [], "comment": None, "thread": None, "fallback": None}
    mentions = [(item, aid, kind, author, ts, story, checked_on) for aid in ids]
    # Comments name their thread; the lookup fills in its points, and this is
    # kept only if the lookup cannot find it (a deleted story, a poll).
    fallback = None
    if kind == "comment" and ids:
        fallback = (story, " ".join((hit.get("story_title") or "").split())[:300],
                    (hit.get("story_url") or "")[:500], None, None, None, checked_on)
    return {"mentions": mentions, "comment": comment, "thread": thread if ids else None,
            "fallback": fallback}


# ── Fetching ─────────────────────────────────────────────────────────────────

def _get(client: httpx.Client, url: str, params: dict, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            r = client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
    raise AssertionError("unreachable")


def fetch_window(client: httpx.Client, base: dict, lo: int, hi: int, out: list, stats: dict) -> None:
    """Every hit of `base` created in [lo, hi) into `out`, splitting the range
    while it holds more than MAX_HITS. Long ranges are counted first
    (hitsPerPage=0 costs almost nothing), so the backfill never downloads a
    thousand hits only to throw them away."""
    nf = f"created_at_i>={lo},created_at_i<{hi}"
    if hi - lo > PROBE_FIRST_S:
        n = _get(client, SEARCH_BY_DATE, {**base, "numericFilters": nf, "hitsPerPage": 0})["nbHits"]
        stats["requests"] += 1
        if n == 0:
            return
        if n > MAX_HITS:
            mid = (lo + hi) // 2
            fetch_window(client, base, lo, mid, out, stats)
            fetch_window(client, base, mid, hi, out, stats)
            return
    d = _get(client, SEARCH_BY_DATE, {**base, "numericFilters": nf, "hitsPerPage": MAX_HITS,
                                      "attributesToRetrieve": ATTRS, "attributesToHighlight": ""})
    stats["requests"] += 1
    hits = d.get("hits") or []
    if d.get("nbHits", 0) > len(hits):
        if hi - lo > 60:
            mid = (lo + hi) // 2
            fetch_window(client, base, lo, mid, out, stats)
            fetch_window(client, base, mid, hi, out, stats)
            return
        stats["lost"] += d["nbHits"] - len(hits)
    out.extend(hits)


def _year_ranges(lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) cut at each 1 January (UTC), so a backfill runs years in parallel."""
    cuts, y = [lo], dt.datetime.fromtimestamp(lo, dt.timezone.utc).year + 1
    while True:
        edge = int(dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp())
        if edge >= hi:
            break
        cuts.append(edge)
        y += 1
    cuts.append(hi)
    return list(zip(cuts, cuts[1:]))


def fetch_items(client: httpx.Client, lo: int, hi: int, workers: int = 1) -> tuple[list[tuple[str, dict]], dict]:
    """(kind, hit) for every query over [lo, hi), and request counts.

    Raises if any window fails after its retries: the caller must not move
    the watermark past a hole."""
    stats = {"requests": 0, "lost": 0}
    jobs = [(kind, base, a, b) for kind, base in QUERIES for a, b in _year_ranges(lo, hi)]

    def run(job):
        kind, base, a, b = job
        hits: list = []
        part = {"requests": 0, "lost": 0}
        fetch_window(client, base, a, b, hits, part)
        return kind, hits, part

    out: list[tuple[str, dict]] = []
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for kind, hits, part in pool.map(run, jobs):
            out += [(kind, h) for h in hits]
            stats["requests"] += part["requests"]
            stats["lost"] += part["lost"]
    return out, stats


def lookup_threads(client: httpx.Client, story_ids: list[str], checked_on: str,
                   stats: dict | None = None) -> dict[str, tuple]:
    """Thread rows for stories by id, LOOKUP_BATCH a request. Ids Algolia does
    not return as stories (deleted, polls, jobs) are left out."""
    out: dict[str, tuple] = {}
    for i in range(0, len(story_ids), LOOKUP_BATCH):
        chunk = story_ids[i:i + LOOKUP_BATCH]
        d = _get(client, SEARCH, {"tags": "story,(" + ",".join(f"story_{s}" for s in chunk) + ")",
                                  "hitsPerPage": len(chunk), "attributesToRetrieve": ATTRS,
                                  "attributesToHighlight": ""})
        if stats is not None:
            stats["requests"] += 1
        for h in d.get("hits") or []:
            sid, ts = str(h.get("objectID") or ""), _int(h.get("created_at_i"))
            if sid in chunk and ts is not None:
                out[sid] = (sid, " ".join((h.get("title") or "").split())[:300], (h.get("url") or "")[:500],
                            _int(h.get("points")), _int(h.get("num_comments")), ts, checked_on)
    return out


# ── Storing ──────────────────────────────────────────────────────────────────

MENTIONS_DDL = ("CREATE TABLE IF NOT EXISTS hn_mentions ("
                "item_id TEXT NOT NULL, arxiv_id TEXT NOT NULL, kind TEXT NOT NULL, "
                "author TEXT NOT NULL, created_at INTEGER NOT NULL, story_id TEXT NOT NULL, "
                "checked_on TEXT NOT NULL, PRIMARY KEY (item_id, arxiv_id))")
MENTIONS_INDEX = "CREATE INDEX IF NOT EXISTS hn_mentions_paper ON hn_mentions (arxiv_id)"
COMMENTS_DDL = ("CREATE TABLE IF NOT EXISTS hn_comments ("
                "item_id TEXT PRIMARY KEY, story_id TEXT NOT NULL, text TEXT NOT NULL, "
                "papers INTEGER NOT NULL, checked_on TEXT NOT NULL)")
THREADS_DDL = ("CREATE TABLE IF NOT EXISTS hn_threads ("
               "story_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL, "
               "points INTEGER, comments INTEGER, created_at INTEGER, checked_on TEXT NOT NULL)")
STATE_DDL = "CREATE TABLE IF NOT EXISTS hn_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
DDL = (MENTIONS_DDL, MENTIONS_INDEX, COMMENTS_DDL, THREADS_DDL, STATE_DDL)


def _arg(v) -> dict:
    if v is None:
        return {"type": "null"}
    return {"type": "integer" if isinstance(v, int) else "text", "value": str(v)}


def upsert(head: str, rows: list[tuple], conflict: str) -> dict:
    """One multi-row INSERT; NULLs pass as NULL, not as the text 'None'."""
    marks = "(" + ", ".join("?" * len(rows[0])) + ")"
    return {"sql": f"{head} VALUES {', '.join([marks] * len(rows))} {conflict}",
            "args": [_arg(v) for row in rows for v in row]}


def statements(mentions: list[tuple], comments: list[tuple], threads: list[tuple]) -> list[list[dict]]:
    """Transactions of at most WRITE_BATCH rows per table. A thread that was
    looked up keeps its points if a later row only knows its title (the
    fallback from a comment), so a lookup that misses never wipes a count."""
    txs: list[list[dict]] = []
    for i in range(0, len(mentions), WRITE_BATCH):
        txs.append([upsert("INSERT INTO hn_mentions (item_id, arxiv_id, kind, author, created_at, story_id, "
                           "checked_on)", mentions[i:i + WRITE_BATCH],
                           "ON CONFLICT (item_id, arxiv_id) DO UPDATE SET author = excluded.author, "
                           "checked_on = excluded.checked_on")])
    for i in range(0, len(comments), WRITE_BATCH):
        txs.append([upsert("INSERT INTO hn_comments (item_id, story_id, text, papers, checked_on)",
                           comments[i:i + WRITE_BATCH],
                           "ON CONFLICT (item_id) DO UPDATE SET text = excluded.text, "
                           "papers = excluded.papers, checked_on = excluded.checked_on")])
    for i in range(0, len(threads), WRITE_BATCH):
        txs.append([upsert("INSERT INTO hn_threads (story_id, title, url, points, comments, created_at, "
                           "checked_on)", threads[i:i + WRITE_BATCH],
                           "ON CONFLICT (story_id) DO UPDATE SET "
                           "title = CASE WHEN excluded.title != '' THEN excluded.title ELSE hn_threads.title END, "
                           "url = CASE WHEN excluded.url != '' THEN excluded.url ELSE hn_threads.url END, "
                           "points = COALESCE(excluded.points, hn_threads.points), "
                           "comments = COALESCE(excluded.comments, hn_threads.comments), "
                           "created_at = COALESCE(excluded.created_at, hn_threads.created_at), "
                           "checked_on = excluded.checked_on")])
    return txs


def collect(items: list[tuple[str, dict]], checked_on: str, last_year: int) -> dict:
    """Mention, comment and thread rows from fetched hits, each item once."""
    mentions: dict[tuple, tuple] = {}
    comments: dict[str, tuple] = {}
    threads: dict[str, tuple] = {}
    fallbacks: dict[str, tuple] = {}
    for kind, hit in items:
        r = rows_from_hit(kind, hit, checked_on, last_year)
        for m in r["mentions"]:
            mentions.setdefault((m[0], m[1]), m)
        if r["comment"]:
            comments[r["comment"][0]] = r["comment"]
        if r["thread"]:
            threads[r["thread"][0]] = r["thread"]
        if r.get("fallback"):
            fallbacks.setdefault(r["fallback"][0], r["fallback"])
    return {"mentions": list(mentions.values()), "comments": list(comments.values()),
            "threads": threads, "fallbacks": fallbacks}


def harvest(st, now_ts: int, since_ts: int | None = None, workers: int = 1,
            client: httpx.Client | None = None) -> dict:
    """Fetch and store everything from the watermark (or `since_ts`) to now.

    `st` is daily_refresh.Stores (its t and t_many). Never raises: the result
    says what happened, with an `error` when the run stopped early."""
    out: dict = {"mentions": 0, "comments": 0, "threads": 0, "papers": 0, "requests": 0, "lost": 0}
    own = client is None
    client = client or httpx.Client(timeout=60, headers={"User-Agent": "ResearchIT daily refresh"})
    try:
        for sql in DDL:
            st.t(sql)
        if since_ts is None:
            got = st.t("SELECT value FROM hn_state WHERE key = 'fetched_to'")
            since_ts = int(got[0][0]) - OVERLAP_S if got else now_ts - OVERLAP_S
        since_ts = max(FIRST_TS, since_ts)
        out["from"] = dt.datetime.fromtimestamp(since_ts, dt.timezone.utc).isoformat(timespec="seconds")
        checked_on = dt.datetime.fromtimestamp(now_ts, dt.timezone.utc).date().isoformat()
        last_year = dt.datetime.fromtimestamp(now_ts, dt.timezone.utc).year
        items, stats = fetch_items(client, since_ts, now_ts, workers)
        out["requests"], out["lost"] = stats["requests"], stats["lost"]
        rows = collect(items, checked_on, last_year)
        need = sorted(set(rows["fallbacks"]) - set(rows["threads"]))
        looked = lookup_threads(client, need, checked_on, out)
        threads = {**rows["fallbacks"], **looked, **rows["threads"]}
        for tx in statements(rows["mentions"], rows["comments"], list(threads.values())):
            st.t_many(tx)
        out.update(mentions=len(rows["mentions"]), comments=len(rows["comments"]), threads=len(threads),
                   papers=len({m[1] for m in rows["mentions"]}),
                   threads_not_found=len(set(need) - set(looked)))
        loaded = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        st.t_many([upsert("INSERT INTO hn_state (key, value)", [("fetched_to", str(now_ts)), ("loaded_at", loaded)],
                          "ON CONFLICT (key) DO UPDATE SET value = excluded.value")])
        out["fetched_to"] = now_ts
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        if own:
            client.close()
    return out
