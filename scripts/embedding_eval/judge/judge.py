"""Blind LLM grading of every (case, paper) item in cases.json.

    python judge.py openai/gpt-oss-120b            # primary judge
    python judge.py qwen/qwen3.8-27b               # second judge (agreement)
    python judge.py openai/gpt-oss-120b --retest   # 10% of cases again, reshuffled
    python judge.py qwen/qwen3.8-27b --k 5         # only each list's first five (first screen)
    python judge.py openai/gpt-oss-20b --k 5 --only home --short   # cheaper prompt, one surface
    python judge.py qwen/qwen3.8-27b --k 5 --shard 1/3 --key 1     # every 3rd case, 2nd extra key

Groq's free tier caps each model at ~200k tokens/day (about 1,300 graded papers),
so --k 5 grades the first screen first; a later --k 10 run adds ranks 6-10.

The judge never sees which model suggested a paper: each case's candidates are
pooled across models, de-duplicated, shuffled (seeded per case and pass) and
labelled c1..cN. Grades are cached in CACHE/judge/judgments.sqlite keyed by
(case, paper, judge, prompt version), so a re-run, a crash or a newly added
model only pays for items not graded yet. Paper text is public arXiv metadata;
home cases also send the titles of papers the reader saved (public dataset).
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SEED, log  # noqa: E402

OUT = CACHE / "judge"
PROMPT_VERSION = "v1"
CHUNK, TPM_BUDGET = 12, 7200

UPNEXT = """A researcher has just finished reading this paper:
TITLE: {title}
ABSTRACT: {abstract}

Below are candidate papers. Rate each as the NEXT paper for this researcher to read.
3 = they would very likely open it next: same specific problem or method, a direct prerequisite, follow-up or competitor
2 = relevant: same subfield and clearly connected
1 = loosely related: same broad area only
0 = not relevant
Judge relevance to this paper only, not fame or quality. Rate every candidate independently.
Return JSON: {{"r":[{{"c":"c1","g":0,"why":"at most 8 words"}}, ...]}} with one entry per candidate.

{candidates}"""

SEARCH = """A researcher typed this into a scientific paper search engine:
QUERY: {query}

Below are papers the engine could return. Rate each by how well it answers this search.
3 = exactly what they are looking for, or one of the best answers
2 = relevant: clearly about what they asked
1 = loosely related: same broad area only
0 = not relevant
Judge relevance to the query only, not fame or quality. Rate every paper independently.
Return JSON: {{"r":[{{"c":"c1","g":0,"why":"at most 8 words"}}, ...]}} with one entry per paper.

{candidates}"""

HOME = """A researcher's feed should show papers they will want to read. Recently they saved these papers (newest first):
{liked}
They marked these as not interesting:
{disliked}

Below are candidate papers for their feed today. Rate each by how much this researcher would want it.
3 = they would very likely open or save it: squarely within one of their interests
2 = relevant: clearly connected to their interests
1 = loosely related: same broad area only
0 = not relevant to them
Judge fit to this researcher only, not fame or quality. Rate every candidate independently.
Return JSON: {{"r":[{{"c":"c1","g":0,"why":"at most 8 words"}}, ...]}} with one entry per candidate.

{candidates}"""


def db() -> sqlite3.Connection:
    OUT.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(OUT / "judgments.sqlite", timeout=120)   # several workers share the cache
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS j (qid TEXT, paper TEXT, judge TEXT, version TEXT,
                 grade INTEGER, why TEXT, ts REAL, PRIMARY KEY (qid, paper, judge, version))""")
    return c


def short(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


class Budget:
    """Keep estimated tokens per rolling minute under the key's limit."""

    def __init__(self, tpm: int):
        self.tpm, self.log = tpm, deque()

    def wait(self, tokens: int) -> None:
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] > 60:
                self.log.popleft()
            if sum(t for _, t in self.log) + tokens <= self.tpm:
                self.log.append((now, tokens))
                return
            time.sleep(1.0)

    def correct(self, actual: int) -> None:
        if self.log:
            self.log[-1] = (self.log[-1][0], actual)


def main(judge: str, retest: bool, k_top: int = 10, only: str | None = None, short_mode: bool = False,
         shard: tuple[int, int] = (0, 1), key_index: int | None = None, cases_file: str = "cases.json") -> None:
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv("/Users/siddh/Projects/researchit/.env.local")
    key = os.environ["GROQ_API_KEY"]
    if key_index is not None:      # extra accounts, one key per line, kept outside the repo (chmod 600)
        keys = [k.strip() for k in (OUT / ".groq_keys").read_text().splitlines() if k.strip()]
        key = keys[key_index]
    client = Groq(api_key=key)
    data = json.loads((OUT / cases_file).read_text())
    papers = data["papers"]
    cases = data["cases"]
    version = PROMPT_VERSION + ("-short" if short_mode else "") + ("-retest" if retest else "")
    snip, why_words = (150, 5) if short_mode else (280, 8)
    if only:
        cases = [c for c in cases if c["surface"] == only]
    cases = [c for i, c in enumerate(cases) if i % shard[1] == shard[0]]
    if retest:
        cases = [c for i, c in enumerate(cases) if i % 10 == 3]
    con = db()
    budget = Budget(TPM_BUDGET)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in judge else {}
    done_items = calls = 0
    t0 = time.time()
    for case in cases:
        items = sorted({a for L in case["lists"].values() for a in L[:k_top]} | set(case.get("probe", {})))
        have = {r[0] for r in con.execute("SELECT paper FROM j WHERE qid=? AND judge=? AND version=?",
                                          (case["qid"], judge, version))}
        todo = [a for a in items if a not in have]
        if not todo:
            continue
        random.Random(f"{SEED}-{case['qid']}-{version}").shuffle(todo)
        if case["surface"] == "upnext":
            s = papers[case["seed"]]
            head = dict(title=s["title"], abstract=short(s["abstract"], 700))
        elif case["surface"] == "search":
            head = dict(query=case["query"])
        else:
            head = dict(liked="\n".join(f"- {papers[a]['title']}" for a in case["context_liked"]),
                        disliked="\n".join(f"- {papers[a]['title']}" for a in case["context_disliked"]) or "- (none)")
        for k in range(0, len(todo), CHUNK):
            part = todo[k:k + CHUNK]
            cands = "\n".join(f"c{i + 1}: {papers[a]['title']} -- {short(papers[a]['abstract'], snip)}"
                              for i, a in enumerate(part))
            tmpl = {"upnext": UPNEXT, "search": SEARCH}.get(case["surface"], HOME)
            prompt = tmpl.format(candidates=cands, **head)
            if short_mode:
                prompt = prompt.replace("at most 8 words", f"at most {why_words} words")
            est = len(prompt) // 3 + 450
            got: dict[str, tuple[int, str]] = {}
            for attempt in range(6):
                budget.wait(est)
                try:
                    r = client.chat.completions.create(
                        model=judge, messages=[{"role": "user", "content": prompt}], max_tokens=1200,
                        temperature=0, response_format={"type": "json_object"}, **extra)
                    budget.correct(r.usage.total_tokens)
                    calls += 1
                    rows = json.loads(r.choices[0].message.content)
                    rows = rows.get("r", []) if isinstance(rows, dict) else rows
                    for row in rows if isinstance(rows, list) else []:
                        if not isinstance(row, dict):
                            continue
                        m = re.fullmatch(r"c(\d+)", str(row.get("c", "")).strip())
                        g = row.get("g")
                        if m and isinstance(g, (int, float)) and 0 <= int(g) <= 3 and 1 <= int(m.group(1)) <= len(part):
                            got[part[int(m.group(1)) - 1]] = (int(g), str(row.get("why", ""))[:120])
                    if len(got) >= len(part):
                        break
                    log("judge", f"{case['qid']}: {len(got)}/{len(part)} parsed, retrying the rest")
                except Exception as e:  # noqa: BLE001 -- rate limits and transient API errors
                    wait = 20 * (attempt + 1)
                    msg = str(e)
                    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
                    if m:
                        wait = 60 * int(m.group(1) or 0) + float(m.group(2)) + 1
                    if "tokens per day" in msg:
                        log("judge", f"{judge}: daily token cap reached; next room in {wait:.0f}s")
                        if wait > 900:
                            log("judge", f"{judge}: stopping; re-run later to continue (cached grades kept)")
                            return
                    log("judge", f"{case['qid']}: {type(e).__name__} {str(e)[:120]} -- waiting {wait:.0f}s")
                    time.sleep(wait)
            con.executemany("INSERT OR REPLACE INTO j VALUES (?,?,?,?,?,?,?)",
                            [(case["qid"], a, judge, version, g, w, time.time()) for a, (g, w) in got.items()])
            con.commit()
            done_items += len(got)
        if calls and calls % 20 == 0:
            log("judge", f"{judge}: {done_items:,} items, {calls} calls, {time.time() - t0:.0f}s")
    n = con.execute("SELECT COUNT(*) FROM j WHERE judge=? AND version=?", (judge, version)).fetchone()[0]
    log("judge", f"{judge} {version}: done; {n:,} grades stored ({done_items:,} new, {calls} calls, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 10
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    shard = tuple(int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/")) if "--shard" in sys.argv else (0, 1)
    key_index = int(sys.argv[sys.argv.index("--key") + 1]) if "--key" in sys.argv else None
    cases_file = sys.argv[sys.argv.index("--cases") + 1] if "--cases" in sys.argv else "cases.json"
    main(sys.argv[1], "--retest" in sys.argv, k, only, "--short" in sys.argv, shard, key_index, cases_file)
