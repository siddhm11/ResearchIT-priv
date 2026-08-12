"""
Curated reading collections.

GET  /collections                     – index of every curated collection
GET  /collections/{slug}              – anchors, plus medoid-matched recent work
POST /api/collections/{slug}/follow   – seed the user's profile from the anchors
POST /api/collections/{slug}/unfollow

Why this exists
---------------
Cold start asks a new user to pick arXiv categories, but `cs.LG` is ~302k
papers -- far too coarse to retrieve against. A collection is a much sharper
signal, and following one replays its anchors through the ordinary save path,
which puts the user straight past the >=5-save threshold into Tier 1
multi-interest clustering. It skips the weakest part of the pipeline entirely.

Semantic Scholar's Research Feeds work the same way -- the feed is seeded from
a curated Library folder, with a stated minimum of "5 relevant papers... and 3
non-relevant". Every collection here carries at least 6 anchors, so one click
clears that budget.
"""
import uuid

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse

from app import arxiv_svc, collections_svc, db, turso_svc, user_state as us
from app.config import COOKIE_NAME
from app.templates_env import templates

router = APIRouter()


async def _hydrate(arxiv_ids: list[str], state) -> dict[str, dict]:
    """arxiv_id -> metadata dict, with saved/dismissed flags for the card."""
    if not arxiv_ids:
        return {}
    meta = await turso_svc.fetch_metadata_batch(arxiv_ids)
    missing = [a for a in arxiv_ids if a not in meta]
    if missing:
        try:
            meta.update(await arxiv_svc.fetch_metadata_batch(missing))
        except Exception as e:
            print(f"[collections] arXiv fallback failed for {len(missing)}: {e}")
    try:
        await db.cache_turso_metadata_batch(list(meta.values()))
    except Exception as e:
        print(f"[collections] metadata cache write skipped: {e}")

    saved = set(state.positive_list) if state else set()
    for aid, m in meta.items():
        m["saved"] = aid in saved
        m["dismissed"] = False
    return meta


@router.get("/collections", response_class=HTMLResponse)
async def collections_index(
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
    cols = collections_svc.load_all()
    followed = await db.get_followed_slugs(user_id)

    # Preview the first three titles per collection, the same two-tier
    # index -> detail shape Hugging Face Collections uses. Titles only: this
    # page must stay one metadata round trip regardless of collection count.
    preview_ids = [a["id"] for c in cols for a in c["anchors"][:3]]
    meta = await turso_svc.fetch_metadata_batch(preview_ids) if preview_ids else {}

    view = []
    for c in cols:
        previews = [
            (meta.get(a["id"]) or {}).get("title", "").strip()
            for a in c["anchors"][:3]
        ]
        view.append({
            **c,
            "previews": [p for p in previews if p],
            "following": c["slug"] in followed,
        })

    resp = templates.TemplateResponse(
        request, "collections.html", {"collections": view},
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.get("/collections/{slug}", response_class=HTMLResponse)
async def collection_detail(
    slug: str,
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())
    col = collections_svc.get(slug)
    if col is None:
        resp = templates.TemplateResponse(
            request, "collections.html",
            {"collections": [], "not_found": slug}, status_code=404,
        )
        resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
        return resp

    state = await us.ensure_loaded(user_id)
    anchor_ids = [a["id"] for a in col["anchors"]]

    # The extension is best-effort. A vector-store hiccup should cost the
    # "recent work" section, never the curated list -- which is the part with
    # editorial value and the part that works with no network at all.
    try:
        extended_ids = await collections_svc.extend(slug, exclude=us.all_seen(user_id))
    except Exception as e:
        print(f"[collections] extend failed for {slug}: {e}")
        extended_ids = []

    meta = await _hydrate(anchor_ids + extended_ids, state)

    anchors = []
    for i, a in enumerate(col["anchors"]):
        m = meta.get(a["id"])
        if m:
            anchors.append({**m, "note": a["note"], "ordinal": i + 1})

    extended = [meta[a] for a in extended_ids if a in meta]
    followed = await db.get_followed_slugs(user_id)

    resp = templates.TemplateResponse(
        request, "collection_detail.html",
        {
            "collection": col,
            "anchors": anchors,
            "extended": extended,
            "following": slug in followed,
        },
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.post("/api/collections/{slug}/follow", response_class=HTMLResponse)
async def follow(
    slug: str,
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Seed the user's profile from a collection's anchors.

    Each anchor goes through the ordinary save path, so the EWMA update, the
    interaction log and the clustering threshold all behave exactly as they do
    for a hand-saved paper. No separate code path means no separate bugs, and
    the profile a follow produces is indistinguishable from one built by hand.
    """
    user_id = user_id or str(uuid.uuid4())
    col = collections_svc.get(slug)
    if col is None:
        return _fragment(request, user_id, None, False, 0)

    ids = [a["id"] for a in col["anchors"]]
    state = await us.ensure_loaded(user_id)
    already = set(state.positive_list)
    seeded = 0
    vectorised = 0

    for pos, aid in enumerate(ids):
        if aid in already:
            continue
        try:
            if await _save_one(user_id, aid, pos, slug):
                vectorised += 1
            seeded += 1
        except Exception as e:
            # One unresolvable paper must not abandon the rest of the seed.
            print(f"[collections] seeding {aid} from {slug} failed: {e}")

    # Saving and profile-seeding are different things, and only the second one
    # makes the feed change. If the vector store cannot resolve the anchors the
    # follow still "works" -- papers land in the library -- while doing nothing
    # at all to recommendations. Say so loudly rather than let it pass.
    #
    # The usual cause is a partially configured vector store. A dev machine
    # with only the primary shard set resolves nothing after ~2021, so every
    # modern collection seeds zero vectors while looking fine.
    if seeded and vectorised < seeded:
        print(f"[collections] {slug}: seeded {seeded} saves but only {vectorised} "
              f"vectors -- profile is under-trained. Check shard configuration.")

    await db.follow_collection(user_id, slug)
    return _fragment(request, user_id, col, True, seeded)


@router.post("/api/collections/{slug}/unfollow", response_class=HTMLResponse)
async def unfollow(
    slug: str,
    request: Request,
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    """Stop following.

    Deliberately does NOT unsave the anchors or rewind the profile: the EWMA is
    a lossy running average with no exact inverse, so "undoing" a follow would
    leave a permanent smudge rather than restoring the previous state. The
    papers stay in the library, where the user can remove them individually.
    """
    user_id = user_id or str(uuid.uuid4())
    await db.unfollow_collection(user_id, slug)
    return _fragment(request, user_id, collections_svc.get(slug), False, 0)


async def _save_one(user_id: str, arxiv_id: str, position: int, slug: str) -> bool:
    """Record one anchor exactly as events.py records a manual save.

    Returns True if the EWMA profile was actually updated -- i.e. the vector
    store resolved this paper. A save without a vector is a library entry that
    teaches the recommender nothing, and the caller needs to tell those apart.
    """
    import numpy as np
    from app import qdrant_svc
    from app.recommend import profiles

    await db.log_interaction(
        user_id=user_id,
        paper_id=arxiv_id,
        event_type="save",
        source="collection",
        position=position,
        query_id=f"collection:{slug}",
        ranker_version="collection_seed",
        candidate_source=f"collection:{slug}",
        cluster_id=None,
        # Deterministic by construction: following a collection shows exactly
        # these papers, so the propensity genuinely is 1.0 (CLAUDE.md 3.11).
        propensity=1.0,
        policy_id="collection_seed",
    )
    us.record_positive(user_id, arxiv_id)

    vectors = await qdrant_svc.get_paper_vectors([arxiv_id])
    vec = vectors.get(arxiv_id)
    if not vec:
        return False
    await profiles.update_on_save(user_id, np.asarray(vec, dtype=np.float32))
    return True


def _fragment(request: Request, user_id: str, col: dict | None,
              following: bool, seeded: int):
    resp = templates.TemplateResponse(
        request, "partials/follow_button.html",
        {"collection": col, "following": following, "seeded": seeded},
    )
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp
