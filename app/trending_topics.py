"""
Rising topics in Hugging Face Daily Papers: the /trending page and the topic line
on the shelf's cards (doc 06 changelog, 2026-10-24).

Once a day, the last WEEKS weeks of Daily Papers that scripts/daily_refresh.py
stored in Turso are grouped into K topics the way a reader's saves are grouped
into interests (CLAUDE.md §3.3: Ward on L2-normalised BGE-M3 vectors, medoids,
Hungarian-matched ids), and each topic's share of every week's list is followed.

What counts as a trend, measured on 26 weeks (5,198 papers, 24 Oct 2026):

  * Not "the last 4 weeks against the rest". At 40 topics, 5 moved by two
    standard errors where shuffled weeks give 1.7 on average and up to 4.
  * A 26-week trend line does hold up: the slope of each topic's weekly share,
    against the same slope with every paper's week shuffled (PERMUTATIONS
    times, weekly totals kept), Benjamini-Hochberg across topics, q < Q_MAX.
    Four topics passed at both 24 and 40 topics.
  * Shares, never counts: the lists ran from 101 to 404 papers a week.

A topic is named by a phrase from its papers' titles, Hugging Face keywords and
summaries, chosen for being common inside it and rare outside it. The name must
appear in NAME_COVERAGE of its papers or the topic gets none: Hugging Face's
own top keywords covered a median 21% of a topic, and one topic that passed the
test was a mix whose best phrase ("benchmark") covered 25%. No model writes the
names (doc 06 defers LLM cluster summaries to Phase 8).

Switch TRENDING_PAGE (app/features.py): off does nothing, shadow computes the
topics and shows nothing, preview shows the page and the topic line to the owner,
on shows everyone.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter
from datetime import date as Date, datetime, timedelta, timezone

import numpy as np
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import squareform

from app import features, qdrant_svc, turso_svc
from app.recommend.clustering import InterestCluster, _find_medoid, stabilize_cluster_ids

SWITCH = "TRENDING_PAGE"
K = 24                      # topics; 24 and 40 found the same trends
WEEKS = 26
EDGE_WEEKS = 4              # the "from" and "to" shares average this many weeks
PERMUTATIONS = 2000
Q_MAX = 0.05
NAME_COVERAGE = 0.30
ALSO_TERMS = 3
MIN_PAPERS = 500            # fewer stored papers than this and nothing is computed
TOP_PAPERS = 3
BEST_PAPERS = 10            # a topic's own page lists this many of its most upvoted
VECTOR_BATCH = 500

_state: dict = {"topics": [], "topic_of": {}, "computed_at": None, "papers": 0,
                "weeks": 0, "first_week": None, "error": None, "scheduler": "off"}
_task: asyncio.Task | None = None


def _refresh_seconds() -> int:
    return max(3600, int(os.getenv("TRENDING_TOPICS_REFRESH_S", "86400")))


# ── Names (pure) ─────────────────────────────────────────────────────────────

# A phrase never starts or ends with one of these.
_FUNCTION = set("""a an the of for and or to in on with via from by is are be we our us using use toward
towards beyond through into its their this that these those as at can how what when which while without
under over across between new novel approach approaches method methods framework frameworks paper
propose proposed show shows study more less than not no yes do does vs""".split())
# Too general to name a topic: a phrase made only of these is skipped ("large
# language models"), though one may sit inside a longer name ("world models").
_GENERIC = set("""model models learning large language llm llms data based training task tasks performance
results system systems agent agents ai deep neural network networks generation multimodal benchmark
benchmarks evaluation dataset datasets reasoning foundation technical report""".split())
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_SEGMENT = re.compile(r"[|.;:,()\[\]!?]")


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def phrases(text: str) -> set[str]:
    """Every 1- to 3-word phrase of `text` a topic could be named by, last word
    made singular. Phrases never cross a field or a clause."""
    out = set()
    for segment in _SEGMENT.split(text.lower()):
        words = _WORD.findall(segment)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if gram[0] in _FUNCTION or gram[-1] in _FUNCTION or all(g in _GENERIC for g in gram):
                    continue
                out.add(" ".join(gram[:-1] + [_singular(gram[-1])]))
    return out


def paper_text(p: dict) -> str:
    return " | ".join([p.get("title") or "", " | ".join(p.get("keywords") or []), p.get("summary") or ""])


def name_topic(members: list[set[str]], df: Counter, n_all: int) -> tuple[str | None, float, list[str]]:
    """(name, the share of members carrying it, up to ALSO_TERMS other phrases).

    The name is the most distinctive phrase that NAME_COVERAGE of the topic
    carries. A single word gives way to a longer phrase that contains it and
    is nearly as common ("world" -> "world model"). No phrase that common, no
    name: the topic is then left out of anything a reader sees."""
    size = len(members)
    counts = Counter(g for m in members for g in m)
    cands = []
    for g, c in counts.items():
        cov = c / size
        lift = cov / (df[g] / n_all)
        if c >= 8 and lift >= 5:
            cands.append((cov * math.log(lift), g, cov))
    cands.sort(reverse=True)
    named = [(sc, g, cov) for sc, g, cov in cands if cov >= NAME_COVERAGE]
    if not named:
        return None, max((cov for _s, _g, cov in cands), default=0.0), []
    _score, name, cov = named[0]
    if " " not in name:
        longer = [(c2, g) for _s, g, c2 in cands if " " in g and name in g.split() and c2 >= max(0.25, cov * 0.35)]
        if longer:
            cov, name = max(longer)
    also = []
    for _s, g, c2 in cands:
        if len(also) == ALSO_TERMS:
            break
        if c2 >= 0.12 and not any(g in t or t in g for t in [name] + also):
            also.append(g)
    return name, cov, also


def display_name(phrase: str, titles: list[str]) -> str:
    """The phrase as titles write it most often ("Vision-Language-Action
    Models"), plural when titles say it so; sentence case otherwise."""
    head, _, last = phrase.rpartition(" ")
    stem = last[:-1] if last.endswith("y") else last
    pat = re.compile(r"\b" + re.escape(head + " " if head else "") + re.escape(stem) + r"(?:y|ies|s|es)?\b",
                     re.IGNORECASE)
    forms = Counter(m.group(0) for t in titles for m in pat.finditer(t))
    if not forms:
        return phrase[:1].upper() + phrase[1:]
    # A topic reads as a plural ("Robots"). A one-word name takes it when titles
    # use it a tenth as often as the commonest form: "robot" is mostly an
    # adjective there (70 to 12 on 24 Oct), while "memories" (7 to 220) is not
    # how the field names its topic. A longer name needs half as often.
    top, n = forms.most_common(1)[0]
    floor = max(1, math.ceil(n / 10)) if " " not in phrase else (n + 1) // 2
    plural = [(c, f) for f, c in forms.items() if f.lower() != phrase and f.lower().endswith("s") and c >= floor]
    form = max(plural)[1] if plural else top
    return form[:1].upper() + form[1:]


# ── Trend test (pure) ────────────────────────────────────────────────────────

def _slopes(counts: np.ndarray) -> np.ndarray:
    """Each topic's weekly share trend, in share points per 4 weeks."""
    shares = counts / counts.sum(0, keepdims=True)
    x = np.arange(counts.shape[1]) - (counts.shape[1] - 1) / 2
    return (shares * x).sum(1) / (x * x).sum() * 4


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    q = np.empty_like(p, dtype=float)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, p[i] * len(p) / rank)
        q[i] = running
    return q


def trend_test(topic: np.ndarray, week: np.ndarray, k: int, w: int,
               permutations: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weekly counts k x w, slopes, q values) for papers labelled `topic` (0..k-1)
    listed in `week` (0..w-1). Every week must hold at least one paper."""
    counts = np.bincount(topic * w + week, minlength=k * w).reshape(k, w).astype(float)
    observed = _slopes(counts)
    rng = np.random.default_rng(seed)
    beat = np.zeros(k)
    for _ in range(permutations):
        shuffled = np.bincount(topic * w + rng.permutation(week), minlength=k * w).reshape(k, w)
        beat += np.abs(_slopes(shuffled.astype(float))) >= np.abs(observed)
    p = (beat + 1) / (permutations + 1)
    return counts, observed, benjamini_hochberg(p)


# ── One computation (pure) ───────────────────────────────────────────────────

def week_index(listed_on: str, first_monday: Date) -> int:
    return (Date.fromisoformat(listed_on) - first_monday).days // 7


def compute(rows: list[dict], vectors: dict, previous: list[dict] | None = None,
            k: int | None = None, permutations: int | None = None) -> dict:
    """Topics from stored Daily Papers `rows` (turso_svc.fetch_hf_daily) and their
    `vectors`. `previous` topics, when given, lend their ids to the matching new
    ones (Hungarian on medoids, as for a reader's interests)."""
    k, permutations = k or K, permutations or PERMUTATIONS
    rows = [r for r in rows if vectors.get(r["arxiv_id"]) is not None]
    if len(rows) < MIN_PAPERS:
        raise ValueError(f"only {len(rows)} papers with vectors")
    newest = max(Date.fromisoformat(r["listed_on"]) for r in rows)
    first_monday = newest - timedelta(days=newest.weekday() + 7 * (WEEKS - 1))
    rows = [r for r in rows if Date.fromisoformat(r["listed_on"]) >= first_monday]
    weeks = np.array([week_index(r["listed_on"], first_monday) for r in rows])
    held = sorted(set(weeks.tolist()))                 # drop weeks with no list at all
    remap = {wk: i for i, wk in enumerate(held)}
    weeks = np.array([remap[wk] for wk in weeks])
    w = len(held)

    X = np.stack([np.asarray(vectors[r["arxiv_id"]], dtype=np.float32) for r in rows])
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    # Euclidean distances of unit vectors from one matrix product, not pdist's loop.
    D = np.sqrt(np.clip(2 - 2 * (X @ X.T), 0, None)).astype(np.float64)
    np.fill_diagonal(D, 0)
    labels = fcluster(ward(squareform(D, checks=False)), k, criterion="maxclust") - 1
    k = int(labels.max()) + 1

    counts, slopes, q = trend_test(labels, weeks, k, w, permutations)
    shares = counts / counts.sum(0, keepdims=True)
    texts = [phrases(paper_text(r)) for r in rows]
    df = Counter(g for t in texts for g in t)

    clusters, topics = [], []
    for t in range(k):
        idx = np.flatnonzero(labels == t)
        members = [rows[i] for i in idx]
        name, coverage, also = name_topic([texts[i] for i in idx], df, len(rows))
        medoid = idx[_find_medoid(X[idx], X[idx].mean(0))]
        clusters.append(InterestCluster(cluster_idx=t, medoid_paper_id=rows[medoid]["arxiv_id"],
                                        medoid_embedding=X[medoid], paper_ids=[r["arxiv_id"] for r in members],
                                        importance=float(len(idx))))
        best = sorted(members, key=lambda r: (-r["upvotes"], r["arxiv_id"]))[:BEST_PAPERS]
        top = best[:TOP_PAPERS]
        rolling = [float(shares[t, max(0, i - EDGE_WEEKS + 1):i + 1].mean()) for i in range(w)]
        topics.append({
            "name": display_name(name, [r["title"] for r in members]) if name else None,
            "also": also,
            "coverage": round(coverage, 3), "papers": len(idx),
            "slope": float(slopes[t]), "q": float(q[t]),
            "from_share": float(shares[t, :EDGE_WEEKS].mean()), "to_share": float(shares[t, -EDGE_WEEKS:].mean()),
            "rolling": rolling,
            "top": [{"arxiv_id": r["arxiv_id"], "title": r["title"], "upvotes": r["upvotes"]} for r in top],
            "best": [{"arxiv_id": r["arxiv_id"], "title": r["title"], "upvotes": r["upvotes"],
                      "listed_on": r["listed_on"]} for r in best]})

    stable = stabilize_cluster_ids(clusters, previous_clusters(previous)) if previous else clusters
    for c, topic in zip(stable, topics):
        topic["id"] = c.cluster_idx
        topic["medoid_paper_id"] = c.medoid_paper_id
        topic["_medoid"] = c.medoid_embedding
    topic_of = {r["arxiv_id"]: topics[t]["id"] for r, t in zip(rows, labels)}
    return {"topics": topics, "topic_of": topic_of, "papers": len(rows), "weeks": w,
            "first_week": first_monday.isoformat()}


def previous_clusters(topics: list[dict]) -> list[InterestCluster]:
    return [InterestCluster(cluster_idx=t["id"], medoid_paper_id=t["medoid_paper_id"],
                            medoid_embedding=t["_medoid"], paper_ids=[], importance=t["papers"])
            for t in topics if t.get("_medoid") is not None]


# ── What the pages read ──────────────────────────────────────────────────────

def is_trend(t: dict) -> bool:
    return t["name"] is not None and t["q"] < Q_MAX


def rising() -> list[dict]:
    return sorted((t for t in _state["topics"] if is_trend(t) and t["slope"] > 0), key=lambda t: -t["slope"])


def cooling() -> list[dict]:
    return sorted((t for t in _state["topics"] if is_trend(t) and t["slope"] < 0), key=lambda t: t["slope"])


def steady() -> list[dict]:
    """Named topics with no trend the test can stand behind, biggest share first.
    Shown as "steady": real groups of papers, with no claim that they move."""
    return sorted((t for t in _state["topics"] if t["name"] is not None and not is_trend(t)),
                  key=lambda t: -t["to_share"])


def label(t: dict) -> str:
    """"rising", "cooling" or "steady": what the page may say about a named topic."""
    if not is_trend(t):
        return "steady"
    return "rising" if t["slope"] > 0 else "cooling"


def get(topic_id: int) -> dict | None:
    """A named topic by id, or None (unnamed topics are never shown)."""
    return next((t for t in _state["topics"] if t["id"] == topic_id and t["name"] is not None), None)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def related(topic: dict, n: int = 4) -> list[dict]:
    """The named topics nearest to `topic` by meaning (their middle papers)."""
    if topic.get("_medoid") is None:
        return []
    me = _unit(topic["_medoid"])
    others = [(float(_unit(t["_medoid"]) @ me), t) for t in _state["topics"]
              if t["id"] != topic["id"] and t["name"] is not None and t.get("_medoid") is not None]
    return [t for _s, t in sorted(others, key=lambda x: -x[0])[:n]]


def nearest(vector) -> dict | None:
    """The topic whose middle paper is closest to `vector`, named or not.

    Labels a paper the daily grouping has not seen (listed since, or never on
    the lists) without grouping again: 24 dot products. Ward groups are not
    exactly the regions around their middle papers, so this is the best guess
    for one paper, not a re-run of the grouping."""
    meds = [t for t in _state["topics"] if t.get("_medoid") is not None]
    if vector is None or not meds:
        return None
    v = _unit(vector)
    return max(meds, key=lambda t: float(_unit(t["_medoid"]) @ v))


def topic_of(arxiv_id: str, vector=None) -> int | None:
    """The topic id of a paper: from the daily grouping, else its nearest topic."""
    tid = _state["topic_of"].get(arxiv_id)
    if tid is not None:
        return tid
    t = nearest(vector)
    return t["id"] if t else None


def topic_line(arxiv_id: str) -> dict | None:
    """The rising topic a paper belongs to, for the card's topic line, or None."""
    tid = _state["topic_of"].get(arxiv_id)
    for t in rising():
        if t["id"] == tid:
            return {"id": t["id"], "name": t["name"], "from": t["from_share"], "to": t["to_share"],
                    "since": _state["first_week"]}
    return None


def coverage() -> dict:
    """What the topics were computed from, for the page's footnote."""
    return {"first_week": _state["first_week"], "weeks": _state["weeks"], "papers": _state["papers"],
            "computed_at": _state["computed_at"], "failed": bool(_state["error"])}


def stats() -> dict:
    return {"switch": features.mode(SWITCH), "scheduler": _state["scheduler"], "papers": _state["papers"],
            "weeks": _state["weeks"], "topics": len(_state["topics"]), "rising": len(rising()),
            "cooling": len(cooling()), "computed_at": _state["computed_at"], "error": _state["error"]}


# ── Refresh ──────────────────────────────────────────────────────────────────

async def refresh(today: Date | None = None) -> dict:
    """Recompute from Turso and Qdrant; on any failure the last topics stay."""
    today = today or datetime.now(timezone.utc).date()
    since = (today - timedelta(weeks=WEEKS + 1)).isoformat()
    rows = await turso_svc.fetch_hf_daily(since)
    ids = [r["arxiv_id"] for r in rows]
    vectors: dict = {}
    for i in range(0, len(ids), VECTOR_BATCH):
        vectors.update(await qdrant_svc.get_paper_vectors(ids[i:i + VECTOR_BATCH], remember=False))
    result = await asyncio.to_thread(compute, rows, vectors, _state["topics"] or None)
    _state.update(result, error=None, computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"[trending_topics] {result['papers']} papers, {len(result['topics'])} topics, "
          f"{len(rising())} rising, {len(cooling())} cooling")
    return stats()


async def _loop() -> None:
    while True:
        try:
            await refresh()
            _state["scheduler"] = "running"
        except Exception as e:  # keep the last good topics; try again next round
            _state["error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"[trending_topics] refresh failed: {_state['error']}")
        await asyncio.sleep(_refresh_seconds())


async def start() -> None:
    global _task
    if not features.active(SWITCH) or _task is not None:
        return
    _state["scheduler"] = "starting"
    _task = asyncio.create_task(_loop(), name="trending-topics")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None
