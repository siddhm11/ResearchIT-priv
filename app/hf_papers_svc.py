"""Read-only Hugging Face Daily Papers adapter for offline/shadow evaluation.

Not called by the serving feed. Source observations are candidates, not relevance
labels or a measured trend velocity. A failed fetch must not become an empty
successful snapshot. The public endpoint is versionless: validate every record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re

import httpx

ENDPOINT = "https://huggingface.co/api/daily_papers"
_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    title: str
    abstract: str
    published_at: str
    source: str
    source_url: str
    observed_at: str
    source_position: int
    upvotes: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_snapshot(payload: object, observed_at: datetime) -> dict:
    """Normalize a bounded source response, rejecting malformed/future records.

    Keep paper.publishedAt, never the wrapper's feed publication timestamp.
    Deduplicate arXiv versions. Missing upvotes remain unknown, not zero.
    """
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must have a timezone")
    if not isinstance(payload, list):
        raise ValueError("Expected a Daily Papers list")
    if len(payload) > 1000:
        raise ValueError("Daily Papers response exceeds 1000 records")
    seen: set[str] = set()
    papers = []
    rejected = duplicates = 0
    for position, record in enumerate(payload):
        paper = record.get("paper") if isinstance(record, dict) else None
        if not isinstance(paper, dict):
            rejected += 1
            continue
        aid, title, date = paper.get("id"), paper.get("title"), paper.get("publishedAt")
        if (not isinstance(aid, str) or not _ID.fullmatch(aid)
                or not isinstance(title, str) or not title.strip()
                or not isinstance(date, str)):
            rejected += 1
            continue
        try:
            published = datetime.fromisoformat(date.replace("Z", "+00:00"))
            if published.tzinfo is None or published > observed_at:
                raise ValueError("Invalid publication time")
        except ValueError:
            rejected += 1
            continue
        aid = re.sub(r"v\d+$", "", aid)
        if aid in seen:
            duplicates += 1
            continue
        seen.add(aid)
        votes = paper.get("upvotes")
        votes = votes if type(votes) is int and votes >= 0 else None
        summary = paper.get("summary")
        papers.append(PaperCandidate(
            aid, title.strip()[:2000], summary[:30000] if isinstance(summary, str) else "",
            published.astimezone(timezone.utc).isoformat(), "huggingface_daily",
            f"https://huggingface.co/papers/{aid}",
            observed_at.astimezone(timezone.utc).isoformat(), position, votes,
        ).to_dict())
    return {"source_url": ENDPOINT, "observed_at": observed_at.isoformat(),
            "papers": papers, "rejected": rejected, "duplicates": duplicates,
            "input_count": len(payload), "trend_velocity": None}


async def fetch_snapshot(client: httpx.AsyncClient, *, now: datetime | None = None) -> dict:
    """One bounded, read-only request. Caller owns scheduling and persistence."""
    async with client.stream("GET", ENDPOINT, timeout=15) as response:
        response.raise_for_status()
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > 4_000_000:
                raise ValueError("Daily Papers response exceeds 4 MB")
    import json
    return normalize_snapshot(json.loads(body), now or datetime.now(timezone.utc))
