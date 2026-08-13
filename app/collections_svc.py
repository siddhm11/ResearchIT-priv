"""
Curated reading collections.

Curated content lives in the REPO (data/collections/*.json), not the database.
DB_PATH is /tmp on an ephemeral filesystem and Hugging Face has withdrawn
persistent storage, so anything hand-authored into SQLite evaporates on the
next rebuild. Keeping the anchors in git also makes an edit reviewable in a
diff, which is what curation actually needs.

Who follows what IS user data and lives in db.collection_follows, replicated
by turso_sync like everything else.

JSON rather than YAML deliberately: pyyaml is only present as a transitive
dependency of transformers, so a YAML loader would import fine in production
and fail in CI, where the test job installs the app dependencies minus the ML
stack. Nothing here should need a dependency the tests do not have.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

COLLECTIONS_DIR = os.getenv(
    "COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "collections"),
)

# How many medoid-matched recent papers to show under the curated anchors.
EXTEND_LIMIT = int(os.getenv("COLLECTION_EXTEND_LIMIT", "12"))


@lru_cache(maxsize=1)
def load_all() -> list[dict]:
    """Every collection, ordered by title. Cached; call cache_clear() in tests.

    A malformed file is skipped rather than fatal -- one bad edit should not
    take down the index page for the other seven.
    """
    out: list[dict] = []
    if not os.path.isdir(COLLECTIONS_DIR):
        print(f"[collections] no directory at {COLLECTIONS_DIR}")
        return out

    for name in sorted(os.listdir(COLLECTIONS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(COLLECTIONS_DIR, name)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            slug = str(data["slug"]).strip()
            anchors = [
                {"id": str(a["id"]).strip(), "note": str(a.get("note", "")).strip()}
                for a in data["anchors"] if a.get("id")
            ]
            if not slug or not anchors:
                raise ValueError("slug and at least one anchor are required")
            out.append({
                "slug": slug,
                "title": str(data.get("title") or slug),
                "blurb": str(data.get("blurb", "")),
                "anchors": anchors,
                "count": len(anchors),
            })
        except Exception as e:
            print(f"[collections] skipping {name}: {type(e).__name__}: {e}")

    out.sort(key=lambda c: c["title"])
    return out


def get(slug: str) -> dict | None:
    slug = (slug or "").strip()
    return next((c for c in load_all() if c["slug"] == slug), None)


def anchor_ids(slug: str) -> list[str]:
    c = get(slug)
    return [a["id"] for a in c["anchors"]] if c else []


def all_anchor_ids() -> set[str]:
    return {a["id"] for c in load_all() for a in c["anchors"]}


async def medoid(slug: str) -> np.ndarray | None:
    """The centre of a collection, as the medoid of its anchor vectors.

    Medoid rather than centroid, for the same reason clustering.py uses one:
    it is an actual paper in the set, so the "recent work in this area" list is
    anchored to something real rather than to a point in space that may sit
    between two unrelated sub-topics.

    Returns None when too few anchors resolve to vectors -- callers then simply
    omit the extension section rather than showing something arbitrary.
    """
    from app import qdrant_svc
    from app.recommend.clustering import _find_medoid

    ids = anchor_ids(slug)
    if not ids:
        return None
    try:
        vectors = await qdrant_svc.get_paper_vectors(ids)
    except Exception as e:
        print(f"[collections] vector fetch failed for {slug}: {e}")
        return None

    found = [vectors[i] for i in ids if i in vectors]
    if len(found) < 2:
        print(f"[collections] {slug}: only {len(found)} anchor vectors -- no medoid")
        return None

    arr = np.asarray(found, dtype=np.float32)
    # L2-normalise before averaging, matching clustering.py: cosine geometry is
    # what the rest of the pipeline assumes.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    centroid = arr.mean(axis=0)
    return arr[_find_medoid(arr, centroid)]


async def extend(slug: str, limit: int = EXTEND_LIMIT,
                 exclude: set[str] | None = None) -> list[str]:
    """Recent papers near the collection medoid, excluding the anchors.

    This is what keeps a curated list from going stale without the curator
    touching it: the anchors are the editorial judgement, and the vector index
    supplies whatever has appeared since.
    """
    from app import qdrant_svc

    vec = await medoid(slug)
    if vec is None:
        return []

    skip = set(exclude or set()) | set(anchor_ids(slug))
    try:
        # search_by_vector returns a list of arxiv_id STRINGS, not dicts, and
        # applies exclude_ids itself. Treating the results as dicts extracted
        # None from every hit and made this silently return [] -- the curated
        # half of the page rendered perfectly while the extension never
        # appeared, which is exactly how it reached production unnoticed.
        hits = await qdrant_svc.search_by_vector(
            vec.tolist(), limit=limit + len(skip), exclude_ids=skip,
        )
    except Exception as e:
        print(f"[collections] extend failed for {slug}: {e}")
        return []

    out: list[str] = []
    for aid in hits:
        if aid and aid not in skip and aid not in out:
            out.append(aid)
        if len(out) >= limit:
            break
    return out
