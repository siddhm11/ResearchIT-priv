"""app/named_paper.py: a recalled title counts only if it is in the corpus, and
the paper it names goes first only with the switch on."""
import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest

from app import config, groq_svc, hybrid_search_svc as hs, local_meta, named_paper


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    path = tmp_path / "metadata.sqlite"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT,
                 abstract_preview TEXT, categories TEXT, primary_topic TEXT, update_date TEXT,
                 citation_count INTEGER, influential_citations INTEGER)""")
    rows = [("1512.03385", "Deep Residual Learning for Image Recognition", 240000),
            ("2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a\n  Reward Model", 10760),
            ("2111.06377", "Masked Autoencoders Are Scalable Vision Learners", 9000),
            ("1706.03762", "Attention Is All You Need", 194000),
            ("2203.99999", "Attention is all you need", 4),
            ("1902.04094", "BERT has a Mouth, and It Must Speak", 300)]
    c.executemany("INSERT INTO papers (arxiv_id, title, citation_count) VALUES (?, ?, ?)", rows)
    c.execute("""CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract_preview, content='papers',
                 content_rowid='rowid', tokenize='porter unicode61')""")
    c.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    c.commit()
    c.close()
    monkeypatch.setattr(local_meta, "SIDECAR_PATH", str(path))
    monkeypatch.setattr(local_meta, "_conn", None)
    monkeypatch.setattr(local_meta, "_probed", False)
    monkeypatch.setattr(local_meta, "_available", False)
    return path


@pytest.mark.parametrize("recalled, expected", [
    ("Deep Residual Learning for Image Recognition", "1512.03385"),
    ("deep residual learning for image recognition.", "1512.03385"),
    ("Deep Residual Learning for Image Recognitions", "1512.03385"),        # near-exact still counts
    ("Attention Is All You Need", "1706.03762"),                            # same title twice: most cited
    ("Direct Preference Optimization", None),                               # only the part before a colon
    ("Deep Residual Learning for", None),                                   # a prefix
    ("React: The Library for Web and Native User Interfaces", None),        # not a paper we hold
    ("Residual", None),                                                     # one word is never enough
])
def test_find_title(sidecar, recalled, expected):
    hit = named_paper.find_title(recalled)
    assert (hit[0] if hit else None) == expected


def test_find_title_without_sidecar(monkeypatch):
    monkeypatch.setattr(local_meta, "connection", lambda: None)
    assert named_paper.find_title("Attention Is All You Need") is None


@pytest.mark.parametrize("reply, title", [
    ("Deep Residual Learning for Image Recognition", "Deep Residual Learning for Image Recognition"),
    ('"Attention Is All You Need" (Vaswani et al., 2017)', "Attention Is All You Need"),
    ("NONE", None),
    ("None.", None),
    ("", None),
    ("Adam: A Method for Stochastic Optimization\nThis paper introduced Adam.", "Adam: A Method for Stochastic Optimization"),
])
def test_reply_parsing(reply, title):
    assert groq_svc._named_title(reply) == title


@pytest.mark.parametrize("query, mode, expected", [
    ("resnet", "on", True), ("lora paper", "shadow", True), ("one two three four", "on", True),
    ("one two three four five", "on", False), ("resnet", "off", False), ("2305.18290", "on", False),
])
def test_wanted(monkeypatch, query, mode, expected):
    monkeypatch.setenv(named_paper.SWITCH, mode)
    assert named_paper.wanted(query) is expected


async def test_resolve_drops_a_title_we_do_not_hold(sidecar, monkeypatch):
    monkeypatch.setattr(groq_svc, "name_paper", AsyncMock(return_value="React: The Library for Web"))
    assert await named_paper.resolve("react") is None
    monkeypatch.setattr(groq_svc, "name_paper", AsyncMock(return_value="Deep Residual Learning for Image Recognition"))
    assert (await named_paper.resolve("resnet"))["arxiv_id"] == "1512.03385"


@pytest.fixture
def search_world(monkeypatch):
    monkeypatch.setattr(config, "SPARSE_BACKEND", "fts")
    monkeypatch.setattr(config, "SEARCH_BGE_RERANK", False)
    monkeypatch.setattr(hs.fts_svc, "is_available", lambda: True)
    monkeypatch.setattr(hs.groq_svc, "rewrite", AsyncMock(side_effect=lambda q: q))
    monkeypatch.setattr(hs.embed_svc, "encode_query", lambda text: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(hs.turso_svc, "fetch_metadata_batch", AsyncMock(return_value={}))
    monkeypatch.setattr(hs.arxiv_svc, "fetch_metadata_batch", AsyncMock(return_value={}))
    monkeypatch.setattr(hs.fts_svc, "search_sparse", AsyncMock(return_value=[
        {"arxiv_id": "1902.04094", "score": 2.0}, {"arxiv_id": "1908.10084", "score": 1.0}]))
    found = {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
             "recalled": "BERT: Pre-training of Deep Bidirectional Transformers"}
    resolve = AsyncMock(return_value=found)
    monkeypatch.setattr(named_paper, "resolve", resolve)
    return resolve


async def test_on_puts_the_named_paper_first(search_world, monkeypatch):
    monkeypatch.setenv(named_paper.SWITCH, "on")
    ids, meta = await hs.search("bert paper", return_meta=True)
    assert ids == ["1810.04805", "1902.04094", "1908.10084"]
    assert meta["named_paper"]["arxiv_id"] == "1810.04805"


async def test_a_named_paper_already_found_moves_up_once(search_world, monkeypatch):
    monkeypatch.setenv(named_paper.SWITCH, "on")
    search_world.return_value = {"arxiv_id": "1908.10084", "title": "Sentence-BERT", "recalled": "Sentence-BERT"}
    ids = await hs.search("sbert")
    assert ids == ["1908.10084", "1902.04094"]


async def test_shadow_records_but_changes_nothing(search_world, monkeypatch):
    monkeypatch.setenv(named_paper.SWITCH, "shadow")
    ids, meta = await hs.search("bert paper", return_meta=True)
    assert ids == ["1902.04094", "1908.10084"]
    assert meta["named_paper_shadow"]["arxiv_id"] == "1810.04805"
    assert "named_paper" not in meta


async def test_off_and_long_and_no_rewrite_searches_never_ask(search_world, monkeypatch):
    monkeypatch.setenv(named_paper.SWITCH, "off")
    await hs.search("bert paper")
    monkeypatch.setenv(named_paper.SWITCH, "on")
    await hs.search("which paper introduced masked language model pretraining")
    await hs.search("bert paper", use_rewrite=False)
    search_world.assert_not_awaited()


async def test_a_late_answer_is_not_waited_for(search_world, monkeypatch):
    monkeypatch.setenv(named_paper.SWITCH, "on")
    monkeypatch.setattr(named_paper, "WAIT_S", 0.05)

    async def slow(query):
        await asyncio.sleep(1)
        return {"arxiv_id": "1810.04805", "title": "BERT", "recalled": "BERT"}
    monkeypatch.setattr(named_paper, "resolve", slow)
    ids, meta = await hs.search("bert paper", return_meta=True)
    assert ids == ["1902.04094", "1908.10084"]
    assert "named_paper" not in meta
