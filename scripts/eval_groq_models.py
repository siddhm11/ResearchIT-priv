"""
Compare Groq models on the three jobs groq_svc gives them.

Run it whenever Groq announces a deprecation (console.groq.com/docs/deprecations)
or before reordering GROQ_MODELS:

    .venv/bin/python scripts/eval_groq_models.py
    .venv/bin/python scripts/eval_groq_models.py --models openai/gpt-oss-120b,qwen/qwen3.8-27b

Each model gets exactly the production prompts and per-family parameters
(groq_svc._request_options), called directly rather than through the failover
chain, so a result belongs to the model named. Measured per job:

  rewrite   latency, empty/truncated answers, the 8-word limit, and whether
            precise terms and paper titles come back unchanged
  overview  latency, truncation, and "unsupported terms"
  explain   latency, truncation, sentence count, and "unsupported terms"

"Unsupported terms" is a cheap grounding proxy: capitalised words, acronyms
and numbers in the output that never occur in the query, titles or abstracts
the model was given. It over-reports (a correct gloss can introduce a new
word), so read the listed terms rather than trusting the count -- but it does
catch the failure that matters, e.g. an overview naming a model or result that
is not in any of the papers.

Paper text comes from Turso, so this needs TURSO_URL/TURSO_DB_TOKEN as well as
GROQ_API_KEY. Free-tier limits are 30 requests and 8K tokens per minute per
model; calls are paced per model to stay under them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, groq_svc, turso_svc  # noqa: E402

# (query, expected) -- expected None means "any rewrite is fine", a string
# means the answer must equal it (precise terms and titles must pass through).
REWRITE_CASES: list[tuple[str, str | None]] = [
    ("when AI makes up fake facts", None),
    ("the llama model by facebook", None),
    ("how do robots learn to walk", None),
    ("making neural nets smaller for phones", None),
    ("teaching computers to see in the dark", None),
    ("why do deep networks generalize", None),
    ("finding planets around other stars", None),
    ("predicting how proteins fold", None),
    ("stopping chatbots from saying harmful things", None),
    ("cheap way to fine tune big models", None),
    ("attention is all you need", "attention is all you need"),
    ("deep residual learning", "deep residual learning"),
    ("graph neural network oversmoothing", "graph neural network oversmoothing"),
    ("Adam optimizer convergence", "Adam optimizer convergence"),
    ("neural tangent kernel", "neural tangent kernel"),
    ("dark matter direct detection", "dark matter direct detection"),
]

OVERVIEW_CASES: list[tuple[str, list[str]]] = [
    ("large language model pretraining",
     ["1706.03762", "2302.13971", "2005.14165", "1810.04805", "2001.08361"]),
    ("diffusion models for image generation",
     ["2006.11239", "2112.10752", "2105.05233", "2010.02502", "2207.12598"]),
    ("graph neural networks",
     ["1609.02907", "1710.10903", "1706.02216", "1810.00826", "1704.01212"]),
    ("parameter efficient fine tuning",
     ["2106.09685", "2305.14314", "1902.00751", "2101.00190", "2104.08691"]),
]

EXPLAIN_IDS = ["1706.03762", "2006.11239", "1609.02907", "2106.09685",
               "1512.03385", "2203.02155", "1412.6980", "2305.14314"]

_TOKENS_PER_MINUTE = 6000       # under the 8K free-tier TPM, with margin
_REQUESTS_PER_MINUTE = 25       # under the 30 RPM free-tier limit


class Pacer:
    """Per-model sliding-window pacing so the eval itself is not rate-limited."""

    def __init__(self) -> None:
        self.events: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def wait(self, model: str, tokens: int) -> None:
        q = self.events[model]
        while True:
            now = time.monotonic()
            while q and now - q[0][0] > 60:
                q.popleft()
            used = sum(t for _, t in q)
            if len(q) < _REQUESTS_PER_MINUTE and used + tokens <= _TOKENS_PER_MINUTE:
                q.append((now, tokens))
                return
            time.sleep(1.0)


def _call(client, pacer: Pacer, model: str, messages: list[dict], *,
          temperature: float, max_tokens: int, est_tokens: int) -> dict:
    import groq
    for attempt in range(4):
        pacer.wait(model, est_tokens)
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                timeout=30.0, **groq_svc._request_options(model, max_tokens))
        except groq.RateLimitError as e:
            time.sleep(groq_svc._retry_after(e))
            continue
        except groq.APIStatusError as e:
            return {"ms": int((time.perf_counter() - t0) * 1000),
                    "text": "", "finish": f"HTTP {e.status_code}", "error": str(e)[:200]}
        c = r.choices[0]
        return {"ms": int((time.perf_counter() - t0) * 1000),
                "text": (c.message.content or "").strip(),
                "finish": c.finish_reason,
                "completion_tokens": getattr(r.usage, "completion_tokens", None)}
    return {"ms": 0, "text": "", "finish": "rate-limited", "error": "gave up after 4 tries"}


_TERM = re.compile(r"\b(?:[A-Z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]*\d[\w.\-%]*)\b")


def unsupported_terms(output: str, source: str) -> list[str]:
    """Capitalised words, acronyms and numbers absent from the source text."""
    src = source.lower()
    terms = []
    for sentence in re.split(r"(?<=[.!?])\s+", output):
        for i, m in enumerate(_TERM.finditer(sentence)):
            word = m.group(0).strip(".-")
            if not word or (i == 0 and m.start() == 0 and word.isalpha() and not word.isupper()):
                continue    # an ordinary sentence-initial capital
            if word.lower() not in src and word.lower().rstrip("s") not in src:
                terms.append(word)
    return sorted(set(terms))


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def evaluate(models: list[str], papers: dict[str, dict]) -> dict:
    client = groq_svc._get_client()
    if client is None:
        raise SystemExit("GROQ_API_KEY not set or groq not installed")
    pacer = Pacer()
    out: dict = {m: {"rewrite": [], "overview": [], "explain": []} for m in models}

    # Interleave models case by case so time-of-day load hits them equally.
    for query, expected in REWRITE_CASES:
        for m in models:
            r = _call(client, pacer, m,
                      [{"role": "system", "content": groq_svc._SYSTEM_PROMPT},
                       {"role": "user", "content": query}],
                      temperature=0.1, max_tokens=60, est_tokens=500)
            text = r["text"].strip().strip('"').strip("'").strip()
            r.update(query=query, expected=expected, answer=text,
                     words=len(text.split()),
                     passthrough_ok=None if expected is None
                     else text.lower() == expected.lower())
            out[m]["rewrite"].append(r)
            print(f"  rewrite  {m:22} {r['ms']:5}ms {r['finish']:>7}  {query!r} -> {text!r}")

    for query, ids in OVERVIEW_CASES:
        group = [papers[i] for i in ids if i in papers]
        if len(group) < 2:
            continue
        prompt = groq_svc._summary_prompt(query, group)
        source = query + " " + " ".join(p["title"] + " " + p["abstract"][:800] for p in group[:5])
        for m in models:
            r = _call(client, pacer, m, [{"role": "user", "content": prompt}],
                      temperature=0.3, max_tokens=150, est_tokens=1800)
            r.update(query=query, unsupported=unsupported_terms(r["text"], source),
                     sentences=_sentences(r["text"]))
            out[m]["overview"].append(r)
            print(f"  overview {m:22} {r['ms']:5}ms {r['finish']:>7}  {query!r} unsupported={r['unsupported']}")

    for aid in EXPLAIN_IDS:
        p = papers.get(aid)
        if not p or len(p.get("abstract") or "") < 120:
            continue
        user = f"<title>{p['title'].strip()}</title>\n<abstract>{p['abstract'].strip()}</abstract>"
        for m in models:
            r = _call(client, pacer, m,
                      [{"role": "system", "content": groq_svc._EXPLAIN_SYSTEM},
                       {"role": "user", "content": user}],
                      temperature=0.2, max_tokens=220, est_tokens=900)
            r.update(arxiv_id=aid, unsupported=unsupported_terms(r["text"], p["title"] + " " + p["abstract"]),
                     sentences=_sentences(r["text"]))
            out[m]["explain"].append(r)
            print(f"  explain  {m:22} {r['ms']:5}ms {r['finish']:>7}  {aid} sentences={r['sentences']} unsupported={r['unsupported']}")
    return out


def summarize(results: dict) -> str:
    rows = ["| model | job | p50 ms | p95 ms | failed/truncated | quality |",
            "|---|---|---|---|---|---|"]
    for m, jobs in results.items():
        for job, rs in jobs.items():
            if not rs:
                continue
            ms = [r["ms"] for r in rs if r["text"]]
            bad = sum(1 for r in rs if not r["text"] or r["finish"] != "stop")
            if job == "rewrite":
                pt = [r["passthrough_ok"] for r in rs if r["passthrough_ok"] is not None]
                over = sum(1 for r in rs if r["words"] > 8)
                quality = f"pass-through {sum(pt)}/{len(pt)}, >8 words {over}"
            else:
                flagged = sum(1 for r in rs if r["unsupported"])
                quality = f"outputs with unsupported terms {flagged}/{len(rs)}"
                if job == "explain":
                    quality += f", exactly 3 sentences {sum(1 for r in rs if r['sentences'] == 3)}/{len(rs)}"
            rows.append(f"| {m} | {job} | {statistics.median(ms) if ms else '-'} | "
                        f"{_pct(ms, 0.95)} | {bad}/{len(rs)} | {quality} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(config.GROQ_MODELS),
                    help="comma-separated Groq model ids (default: GROQ_MODELS)")
    ap.add_argument("--out", default="reports/groq_model_eval.json",
                    help="where to write the full per-call results")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    ids = sorted({i for _, g in OVERVIEW_CASES for i in g} | set(EXPLAIN_IDS))
    papers = asyncio.run(turso_svc.fetch_metadata_batch(ids))
    print(f"[eval_groq] {len(papers)}/{len(ids)} papers fetched; models: {models}")

    results = evaluate(models, papers)
    table = summarize(results)
    print("\n" + table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "models": models, "summary": table, "results": results}, indent=1))
    print(f"\n[eval_groq] full results -> {args.out}")


if __name__ == "__main__":
    main()
