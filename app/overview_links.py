"""Link the papers a search overview names to their pages.

The overview model sees only the top five results (groq_svc._summary_prompt),
so a title it names is almost always one of them, and their ids are already in
hand. Nothing here asks the model for links: it works on the overview after
generate_search_summary has escaped it, finds the known titles, and wraps the
first mention of each in an <a> built from our own id. Model text never
becomes markup, and a title the overview does not name stays unlinked.

Two names per paper:
  the full title         case-insensitive, any whitespace, at least 3 words;
  the name before ":"    "BERT" from "BERT: Pre-training of ...", only when it
                         reads like a name (a capital inside a word, as in
                         BERT or LoRA, or a digit), matched case-sensitively,
                         so "Deep Learning: A Survey" never links the phrase.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote, urlencode

_TAG = re.compile(r"(<[^>]+>)")
_MIN_TITLE_WORDS = 3
_MAX_SHORT_WORDS = 4


def _pattern(escaped: str) -> str:
    return r"\s+".join(re.escape(w) for w in escaped.split())


def _short_name(title: str) -> str | None:
    head, sep, _ = title.partition(":")
    head = head.strip()
    if not sep or not head or len(head.split()) > _MAX_SHORT_WORDS:
        return None
    if not (re.search(r"\d", head) or any(re.search(r"[A-Z]", w[1:]) for w in head.split())):
        return None
    return head


def _names(papers: list[dict]) -> list[tuple[str, int]]:
    """(regex, paper index) for every name, longest first, ambiguous short names dropped."""
    names: list[tuple[str, int, int]] = []
    shorts: dict[str, list[int]] = {}
    for i, p in enumerate(papers):
        title = " ".join((p.get("title") or "").split())
        if len(title.split()) >= _MIN_TITLE_WORDS:
            escaped = html.escape(title, quote=False)
            names.append((f"(?i:{_pattern(escaped)})", len(escaped), i))
        short = _short_name(title)
        if short:
            shorts.setdefault(short, []).append(i)
    for short, owners in shorts.items():
        if len(owners) == 1:
            escaped = html.escape(short, quote=False)
            names.append((_pattern(escaped), len(escaped), owners[0]))
    names.sort(key=lambda n: -n[1])
    return [(rx, i) for rx, _, i in names]


def _href(arxiv_id: str, qid: str, position: int | None) -> str:
    params = {}
    if qid:
        # The same provenance a result card's link carries (§3.11), with the
        # surface named, so overview clicks never pass for result-list clicks.
        params = {"qid": qid, "sf": "search_overview"}
        if position is not None:
            params["pos"] = str(position)
    url = "/p/" + quote(arxiv_id, safe="/.")
    return url + ("?" + urlencode(params) if params else "")


def link_titles(summary_html: str, papers: list[dict], *, qid: str = "",
                positions: dict[str, int] | None = None) -> str:
    """summary_html with the first mention of each paper's title linked to /p/<id>."""
    papers = [p for p in papers if p.get("arxiv_id")]
    names = _names(papers)
    if not summary_html or not names:
        return summary_html
    combined = re.compile(
        "|".join(rf"(?P<g{k}>(?<![\w-]){rx}(?![\w-]))" for k, (rx, _) in enumerate(names)))
    linked: set[int] = set()

    def _link(m: re.Match) -> str:
        i = names[int(m.lastgroup[1:])][1]
        if i in linked:
            return m.group(0)
        linked.add(i)
        aid = papers[i]["arxiv_id"]
        href = html.escape(_href(aid, qid, (positions or {}).get(aid)), quote=True)
        return f'<a class="overview-paper" href="{href}">{m.group(0)}</a>'

    parts = _TAG.split(summary_html)
    for k in range(0, len(parts), 2):          # even parts are text, odd are tags
        parts[k] = combined.sub(_link, parts[k])
    return "".join(parts)
