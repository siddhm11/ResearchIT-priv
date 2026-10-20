#!/usr/bin/env python3
"""
Cut an arXiv HTML page down to a small test fixture.

    python scripts/make_arxiv_fixture.py page.html out.html --keep S1,S4.SS1 \\
        --credit "Jiang et al. 2026, SatNav, CC BY 4.0"

Keeps what the parser needs (the licence infobox, title, abstract) plus the
sections named in --keep, whole, and adds a stub contents nav and bibliography
so the parser's skipping is exercised too. Only ever commit pages under a
licence that allows redistribution (CC BY, CC BY-SA, CC0), with credit.

    --sizes   print each top-level section's size and stop, to choose --keep
"""
from __future__ import annotations

import argparse
import re
import sys


def balanced(html: str, start: int, tag: str) -> str:
    """The element whose opening tag starts at `start`, up to its matching close."""
    depth = 0
    for m in re.finditer(rf"<{tag}\b|</{tag}>", html[start:]):
        depth += -1 if m.group(0).startswith("</") else 1
        if depth == 0:
            return html[start:start + m.end()]
    raise ValueError(f"unclosed <{tag}> at {start}")


def element(html: str, pattern: str, tag: str) -> str:
    m = re.search(pattern, html)
    return balanced(html, m.start(), tag) if m else ""


def section(html: str, anchor: str) -> str:
    return element(html, rf'<section id="{re.escape(anchor)}"', "section")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--keep", default="")
    ap.add_argument("--credit", default="")
    ap.add_argument("--sizes", action="store_true")
    args = ap.parse_args(argv)
    html = open(args.page, encoding="utf-8").read()

    if args.sizes:
        for m in re.finditer(r'<section id="([A-Z]+\d+)" class="ltx_(?:section|appendix)', html):
            print(f"{m.group(1):6} {len(section(html, m.group(1))) // 1024:5} KB")
        return 0

    parts = [element(html, r'<div[^>]*\bid="infobox"', "div"),
             element(html, r'<h1[^>]*\bclass="ltx_title ltx_title_document"', "h1"),
             element(html, r'<div[^>]*\bclass="ltx_abstract"', "div")]
    for anchor in filter(None, args.keep.split(",")):
        s = section(html, anchor.strip())
        if not s:
            raise SystemExit(f"no section {anchor}")
        parts.append(s)
    title = re.sub(r"<[^>]+>", "", parts[1]).strip()
    body = "\n".join(parts)
    page = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n</head>\n<body>\n"
        f"<!-- Test fixture trimmed from arXiv HTML by scripts/make_arxiv_fixture.py. {args.credit} -->\n"
        "<nav class=\"ltx_TOC\"><ol><li><a href=\"#S1\" class=\"ltx_ref\">Contents stub</a></li></ol></nav>\n"
        "<div class=\"ltx_page_main\">\n"
        f"{body}\n"
        "<section id=\"bib\" class=\"ltx_bibliography\"><h2 class=\"ltx_title ltx_title_bibliography\">References</h2>"
        "<ul class=\"ltx_biblist\"><li id=\"bib.bib1\" class=\"ltx_bibitem\">Bibliography stub, never parsed.</li></ul></section>\n"
        "</div>\n</body>\n</html>\n")
    if args.out:
        open(args.out, "w", encoding="utf-8").write(page)
        print(f"{args.out}: {len(page.encode()) // 1024} KB", file=sys.stderr)
    else:
        sys.stdout.write(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
