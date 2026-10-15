"""Synthetic search queries: an LLM writes what a researcher would type to find a paper.

Sources (fixed seed): 100 random corpus papers (every field, natural mix) and 80
AI/ML papers (50 from mid-2024 on, 30 from 2015-2023). Two queries per paper:
  keyword   2-6 words typed quickly, no copying the title
  question  one sentence describing the need, as someone who hasn't read it
The source paper is the known answer. Writes CACHE/search/synthetic_queries.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SEED, log  # noqa: E402

S = CACHE / "search"
PROMPT = """You write realistic queries that researchers type into a scientific paper search engine.
For each paper below write two queries someone would use to find it:
- "keyword": 2-6 words, typed quickly. Use the field's usual terms. Do NOT copy the title: at most two consecutive words from it.
- "question": one plain sentence describing what they are looking for, written by someone who has not read this paper. Describe the problem or approach; avoid the title's distinctive words and any model or dataset names coined by this paper.
Return JSON: {{"q":[{{"p":"p1","keyword":"...","question":"..."}}, ...]}} with one entry per paper.

{papers}"""


def short(t: str, n: int) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."


def main() -> None:
    from groq import Groq
    lib = json.loads((S / "library.json").read_text())
    keys = [k.strip() for k in (CACHE / "judge" / ".groq_keys").read_text().splitlines() if k.strip()]
    client = Groq(api_key=keys[0])
    rnd = random.Random(SEED)
    bg = [a for a in lib["background"] if lib["abstracts"].get(a)]
    recent = [s for s in lib["aiml_seeds"] if lib["aiml_stratum"][s] == "recent"]
    est = [s for s in lib["aiml_seeds"] if lib["aiml_stratum"][s] == "established"]
    src = [(a, "random corpus") for a in rnd.sample(bg, 100)] + [(a, "AI/ML recent") for a in rnd.sample(recent, 50)] \
        + [(a, "AI/ML established") for a in rnd.sample(est, 30)]
    out = []
    for k in range(0, len(src), 5):
        part = src[k:k + 5]
        papers = "\n\n".join(f"p{i + 1}: TITLE: {lib['titles'][a]}\nABSTRACT: {short(lib['abstracts'][a], 900)}"
                             for i, (a, _) in enumerate(part))
        for attempt in range(6):
            try:
                r = client.chat.completions.create(
                    model="openai/gpt-oss-120b", messages=[{"role": "user", "content": PROMPT.format(papers=papers)}],
                    max_tokens=1500, temperature=0.3, response_format={"type": "json_object"}, reasoning_effort="low")
                rows = json.loads(r.choices[0].message.content).get("q", [])
                got = {}
                for row in rows:
                    m = re.fullmatch(r"p(\d+)", str(row.get("p", "")))
                    if m and 1 <= int(m.group(1)) <= len(part) and row.get("keyword") and row.get("question"):
                        got[int(m.group(1)) - 1] = row
                for i, (a, grp) in enumerate(part):
                    if i in got:
                        for style in ("keyword", "question"):
                            out.append({"qid": f"sy{len(out):03d}", "set": f"synthetic {style}", "group": grp,
                                        "query": got[i][style].strip(), "gold": [a]})
                break
            except Exception as e:  # noqa: BLE001
                log("search", f"retry {attempt}: {str(e)[:120]}")
                time.sleep(15 * (attempt + 1))
    (S / "synthetic_queries.json").write_text(json.dumps(out, indent=1))
    log("search", f"wrote {len(out)} synthetic queries from {len(src)} papers")


if __name__ == "__main__":
    main()
