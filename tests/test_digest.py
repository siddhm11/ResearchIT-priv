"""
The weekly digest (app/digest_svc.py, app/routers/digest.py, app/email_svc.py):
new papers in a reader's interests from the last 7 days, by private link, RSS
or (when configured) email with double opt-in and one-click unsubscribe.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import digest_svc

NOW = datetime.now(timezone.utc)
THIS_WEEK = (NOW - timedelta(days=2)).strftime("%Y-%m-%d")      # inside any window the pages use
LONG_AGO = (NOW - timedelta(days=60)).strftime("%Y-%m-%d")
UA = {"user-agent": "Mozilla/5.0 (Macintosh) Chrome/131 Safari/537.36"}


def _paper(aid, published, title=None):
    return {"arxiv_id": aid, "title": title or f"Paper {aid}", "published": published, "category": "cs.LG",
            "authors": '["Ada Lovelace", "Alan Turing"]', "abstract": f"We study {aid} carefully. More follows."}


# Two interests: A (importance 8) and B (importance 2). A's searches return
# a01..a20, B's b01..b20; odd ones are old, even ones this week.
PAPERS = {}
for p in ("a", "b"):
    for i in range(1, 21):
        aid = f"{NOW:%y%m}.{p}{i:02d}"
        PAPERS[aid] = _paper(aid, THIS_WEEK if i % 2 == 0 else LONG_AGO)


@pytest.fixture
def world(tmp_path, monkeypatch):
    import app.config as cfg
    import app.db as db_mod
    import app.qdrant_svc as q
    import app.turso_svc as turso
    import app.user_state as us
    db_path = str(tmp_path / "t.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    us._cache.clear()
    digest_svc._cache.clear()
    asyncio.run(db_mod.init_db())

    def vec(x):
        v = np.zeros(1024, dtype=np.float32)
        v[x] = 1
        return v

    async def search_recent(vector, limit=20, exclude_ids=None):
        prefix = "a" if vector[0] == 1 else "b"
        ids = [a for a in PAPERS if a[5] == prefix and a not in (exclude_ids or set())]
        return [{"arxiv_id": a, "score": 1 - i / 100} for i, a in enumerate(ids)][:limit]

    monkeypatch.setattr(q, "search_recent", search_recent)
    monkeypatch.setattr(turso, "fetch_metadata_batch",
                        AsyncMock(side_effect=lambda ids: {i: dict(PAPERS[i]) for i in ids if i in PAPERS}))

    async def clusters(user_id):
        if user_id != "reader":
            return []
        return [{"cluster_idx": 0, "importance": 8.0, "paper_ids": json.dumps([]), "medoid_paper_id": "x",
                 "medoid_embedding_blob": vec(0).tobytes(), "computed_at": ""},
                {"cluster_idx": 1, "importance": 2.0, "paper_ids": json.dumps([]), "medoid_paper_id": "y",
                 "medoid_embedding_blob": vec(1).tobytes(), "computed_at": ""}]
    monkeypatch.setattr(db_mod, "get_user_clusters", clusters)
    return db_mod


def test_quota_keeps_the_minor_interest_and_only_new_papers(world):
    d = asyncio.run(digest_svc.build("reader", now=NOW))
    assert d["kind"] == "interests"
    sizes = [len(s["papers"]) for s in d["sections"]]
    # 10 slots, importance 8:2, floor 3 per interest -> 8 and 3? allocate_quotas
    # floors first: max(floor(8), 3)=8, max(floor(2), 3)=3.
    assert sizes == [8, 3]
    ids = [p["arxiv_id"] for s in d["sections"] for p in s["papers"]]
    assert all(PAPERS[a]["published"] == THIS_WEEK for a in ids)            # the window holds
    assert len(set(ids)) == len(ids)
    assert [p["position"] for s in d["sections"] for p in s["papers"]] == list(range(len(ids)))


def test_saved_and_passed_papers_are_left_out(world, monkeypatch):
    import app.user_state as us
    monkeypatch.setattr(us, "all_seen", lambda uid: {f"{NOW:%y%m}.a02", f"{NOW:%y%m}.b02"})
    d = asyncio.run(digest_svc.build("reader", now=NOW))
    ids = {p["arxiv_id"] for s in d["sections"] for p in s["papers"]}
    assert f"{NOW:%y%m}.a02" not in ids and f"{NOW:%y%m}.b02" not in ids


def test_window_months_cover_submission_before_publication():
    assert digest_svc.window_months(datetime(2026, 12, 3, tzinfo=timezone.utc), 7) == {"2611", "2612"}
    assert digest_svc.window_months(datetime(2026, 11, 27, tzinfo=timezone.utc), 7) == {"2611"}


def test_floor_never_overruns_with_many_interests():
    assert digest_svc.floor_for(2) == 3
    assert digest_svc.floor_for(5) == 2
    assert digest_svc.floor_for(7) == 1


def test_a_reader_without_interests_gets_the_week_labelled_as_such(world, monkeypatch):
    from app import trending_shelf
    monkeypatch.setattr(trending_shelf, "week_top",
                        lambda n=10, exclude=frozenset(), seen=frozenset(): [_paper("2611.h01", THIS_WEEK, "Top")])
    d = asyncio.run(digest_svc.build("newcomer", now=NOW))
    assert d["kind"] == "week" and d["sections"][0]["papers"][0]["title"] == "Top"


# ── pages ───────────────────────────────────────────────────────────────────
@pytest.fixture
def client(world):
    from app.main import app
    with TestClient(app) as c:
        c.cookies.set("researchit_uid", "reader")
        yield c


def _cookie_name():
    from app.config import COOKIE_NAME
    return COOKIE_NAME


@pytest.fixture
def reader(client):
    client.cookies.clear()
    client.cookies.set(_cookie_name(), "reader")
    return client


def test_my_digest_page_lists_papers_with_ranked_links_and_a_private_link(reader, world):
    r = reader.get("/digest", headers=UA)
    assert r.status_code == 200
    assert "New papers in your interests" in r.text and f"Paper {NOW:%y%m}.a02" in r.text
    assert "sf=digest" in r.text and "pol=digest_v1" in r.text
    assert 'name="robots" content="noindex"' in r.text
    row = asyncio.run(world.get_digest_by_user("reader"))
    assert row["token"] in r.text and row["token"] != "reader"          # never the user id
    assert "Send it to me" not in r.text                                 # email not configured
    # Shown papers are exposures (§3.11).
    import aiosqlite

    async def count():
        async with aiosqlite.connect(world.DB_PATH) as conn:
            return (await (await conn.execute(
                "SELECT COUNT(*) FROM feed_exposures WHERE policy_id = 'digest_v1'")).fetchone())[0]
    assert asyncio.run(count()) == 11


def test_the_link_and_the_feed_work_from_anywhere(reader, world):
    reader.get("/digest", headers=UA)
    token = asyncio.run(world.get_digest_by_user("reader"))["token"]
    reader.cookies.clear()
    page = reader.get(f"/digest/{token}", headers=UA)
    assert page.status_code == 200 and f"Paper {NOW:%y%m}.a02" in page.text and "Get it every week" not in page.text
    feed = reader.get(f"/digest/{token}/feed.xml")
    assert feed.status_code == 200 and feed.headers["content-type"].startswith("application/rss+xml")
    assert f"<item><title>Paper {NOW:%y%m}.a02</title>" in feed.text and "qid=" not in feed.text
    assert reader.get("/digest/not-a-token").status_code == 404
    assert reader.get("/digest/not-a-token/feed.xml").status_code == 404


def test_a_crawler_gets_no_row(client, world):
    client.cookies.clear()
    r = client.get("/digest", headers={"user-agent": "Googlebot/2.1"})
    assert r.status_code == 200 and "set-cookie" not in r.headers


# ── email ───────────────────────────────────────────────────────────────────
@pytest.fixture
def mail(monkeypatch):
    from app import email_svc
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("DIGEST_FROM", "ResearchIT <digest@example.org>")
    monkeypatch.setenv("DIGEST_CRON_KEY", "cron-secret")
    sent = []

    async def send(to, subject, html, text, headers=None):
        sent.append({"to": to, "subject": subject, "html": html, "text": text, "headers": headers or {}})
        return "id"
    monkeypatch.setattr(email_svc, "send", send)
    return sent


def test_email_double_opt_in_send_and_one_click_unsubscribe(reader, world, mail):
    same = {**UA, "sec-fetch-site": "same-origin"}
    assert "Send it to me" in reader.get("/digest", headers=UA).text
    r = reader.post("/digest/email", data={"email": "reader@example.org"}, headers=same, follow_redirects=False)
    assert r.status_code == 303 and "check-inbox" in r.headers["location"]
    assert len(mail) == 1 and mail[0]["to"] == "reader@example.org"
    row = asyncio.run(world.get_digest_by_user("reader"))
    assert row["status"] == "pending" and row["confirm_token"] in mail[0]["html"]

    # A GET (a mail scanner opening the link) does not subscribe.
    assert reader.get(f"/digest/confirm/{row['confirm_token']}", headers=UA).status_code == 200
    assert asyncio.run(world.get_digest_by_user("reader"))["status"] == "pending"
    reader.post(f"/digest/confirm/{row['confirm_token']}", headers=UA)
    assert asyncio.run(world.get_digest_by_user("reader"))["status"] == "active"

    # The weekly send: refused without the key, then one email with unsubscribe headers.
    assert reader.post("/digest/send").status_code == 403
    out = reader.post("/digest/send", headers={"authorization": "Bearer cron-secret"}).json()
    assert out == {"configured": True, "sent": 1, "skipped_empty": 0, "failed": 0}
    email = mail[-1]
    assert email["subject"].startswith("11 new papers for you this week")
    assert f"Paper {NOW:%y%m}.a02" in email["html"] and "sf=digest_email" in email["text"]
    assert email["headers"]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    # Not sent again within the week.
    assert reader.post("/digest/send", headers={"authorization": "Bearer cron-secret"}).json()["sent"] == 0

    # One-click unsubscribe: a POST from the mail provider, no site headers.
    unsub = email["headers"]["List-Unsubscribe"].strip("<>").replace("https://siddhm11-researchit.hf.space", "")
    reader.cookies.clear()
    assert reader.post(unsub, data={"List-Unsubscribe": "One-Click"}).status_code == 200
    row = asyncio.run(world.get_digest_by_user("reader"))
    assert row["status"] == "unsubscribed" and row["email"] is None


def test_email_sign_up_is_refused_from_another_site_or_with_a_bad_address(reader, world, mail):
    r = reader.post("/digest/email", data={"email": "x@example.org"},
                    headers={**UA, "sec-fetch-site": "cross-site"}, follow_redirects=False)
    assert r.status_code == 303 and not mail
    r = reader.post("/digest/email", data={"email": "not an email"},
                    headers={**UA, "sec-fetch-site": "same-origin"}, follow_redirects=False)
    assert "bad-email" in r.headers["location"] and not mail


def test_send_without_email_configured_says_so(reader, monkeypatch):
    monkeypatch.setenv("DIGEST_CRON_KEY", "k")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    r = reader.post("/digest/send", headers={"authorization": "Bearer k"})
    assert r.status_code == 503 and r.json()["configured"] is False


def test_robots_keeps_digests_out():
    from app.routers.seo import ROBOTS
    assert "Disallow: /digest" in ROBOTS


def test_since_last_visit_box(reader, world):
    import aiosqlite

    async def visited(day):
        async with aiosqlite.connect(world.DB_PATH) as conn:
            await conn.execute("INSERT INTO reader_days (day, user_id) VALUES (?, 'reader')", (day,))
            await conn.commit()
    hx = {**UA, "hx-request": "true"}
    assert reader.get("/digest/since", headers=hx).text == ""          # never here before
    asyncio.run(visited((NOW - timedelta(days=3)).strftime("%Y-%m-%d")))
    r = reader.get("/digest/since", headers=hx)
    assert "Since your last visit" in r.text and "new papers in your interests" in r.text
    assert "sf=since_last_visit" in r.text and "/digest?days=" in r.text
    reader.cookies.clear()
    assert reader.get("/digest/since", headers=hx).text == ""          # no reader, no box
