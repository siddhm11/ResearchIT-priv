"""
Whole-site audit for ResearchIT. Structural, accessibility and hygiene checks
across every user-facing surface, run against a live server.

Deliberately NOT a linter run: the point is to find things that are wrong for a
reader or a crawler, in this specific product, and to separate "actually broken"
from "cosmetically imperfect".
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
COOKIE = "arxiv_user_id=feeddemo"

PAGES = [
    ("/",                       "Feed (redirect or page)"),
    ("/onboarding",             "Onboarding wizard"),
    ("/saved",                  "Library"),
    ("/collections",            "Collections index"),
    ("/search",                 "Search (empty)"),
    ("/search?q=transformer",   "Search (results)"),
    ("/api/recommendations",    "Feed fragment"),
]

findings: list[dict] = []


def add(sev, area, title, detail, where=""):
    findings.append({"sev": sev, "area": area, "title": title,
                     "detail": detail, "where": where})


def fetch(path, cookie=True):
    req = urllib.request.Request(BASE + path, headers={
        "User-Agent": "researchit-audit/1.0",
        **({"Cookie": COOKIE} if cookie else {}),
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, dict(r.headers), body, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace"), time.time() - t0
    except Exception as e:
        return 0, {}, f"__ERROR__ {type(e).__name__}: {e}", time.time() - t0


class Scan(HTMLParser):
    """Collect what the checks below need, in one pass."""

    def __init__(self):
        super().__init__()
        self.headings: list[tuple[int, str]] = []
        self._h = None
        self.imgs: list[dict] = []
        self.inputs: list[dict] = []
        self.labels: set[str] = set()
        self.links: list[dict] = []
        self.buttons: list[dict] = []
        self.ids: list[str] = []
        self.title = ""
        self._in_title = False
        self.lang = ""
        self.metas: dict[str, str] = {}
        self.tabindex: list[str] = []
        self.autofocus = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.append(a["id"])
        if a.get("tabindex"):
            self.tabindex.append(a["tabindex"])
        if "autofocus" in a:
            self.autofocus += 1
        if tag == "html":
            self.lang = a.get("lang", "")
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            k = a.get("name") or a.get("property") or a.get("charset")
            if k:
                self.metas[k] = a.get("content", "charset")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._h = (int(tag[1]), "")
        if tag == "img":
            self.imgs.append(a)
        if tag in ("input", "textarea", "select"):
            self.inputs.append(a)
        if tag == "label" and a.get("for"):
            self.labels.add(a["for"])
        if tag == "a":
            self.links.append(a)
        if tag == "button":
            self.buttons.append(a)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._h:
            self.headings.append(self._h)
            self._h = None

    def handle_data(self, d):
        if self._in_title:
            self.title += d.strip()
        if self._h:
            self._h = (self._h[0], (self._h[1] + " " + d.strip()).strip())


# ── Contrast maths (WCAG 2.x relative luminance) ────────────────────────────

def _lin(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lum(hexs):
    hexs = hexs.lstrip("#")
    if len(hexs) == 3:
        hexs = "".join(ch * 2 for ch in hexs)
    r, g, b = (int(hexs[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def ratio(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main():
    print(f"# audit of {BASE}\n")
    results = {}

    # ── 1. Reachability, timing, headers ────────────────────────────────
    for path, label in PAGES:
        st, hdr, body, dt = fetch(path)
        results[path] = (st, hdr, body, dt)
        if st == 0:
            add("critical", "Availability", f"{label} failed to load",
                body[:200], path)
            continue
        if st >= 500:
            add("critical", "Availability", f"{label} returned {st}", "", path)
        if dt > 5:
            add("major", "Performance", f"{label} took {dt:.1f}s",
                "Nielsen's limit for keeping user flow is ~1s; past ~10s "
                "attention is lost entirely.", path)
        elif dt > 2:
            add("minor", "Performance", f"{label} took {dt:.1f}s", "", path)

    # ── 2. Per-page HTML structure ──────────────────────────────────────
    for path, label in PAGES:
        st, hdr, body, dt = results.get(path, (0, {}, "", 0))
        if st != 200 or body.startswith("__ERROR__"):
            continue
        frag = path.startswith("/api/")
        s = Scan()
        try:
            s.feed(body)
        except Exception as e:
            add("major", "Markup", f"{label}: HTML failed to parse",
                f"{type(e).__name__}: {e}", path)
            continue

        if not frag:
            if not s.title:
                add("major", "SEO", f"{label} has no <title>", "", path)
            if not s.lang:
                add("major", "A11y", f"{label} has no lang attribute",
                    "Screen readers need it to pick a pronunciation voice.", path)
            if "description" not in s.metas:
                add("minor", "SEO", f"{label} has no meta description", "", path)
            if "viewport" not in s.metas:
                add("critical", "Responsive", f"{label} has no viewport meta", "", path)
            # Open Graph / social
            if not any(k.startswith("og:") for k in s.metas):
                add("minor", "SEO", f"{label} has no Open Graph tags",
                    "Links shared to Slack/Twitter/Discord render as a bare "
                    "URL with no title, description or image.", path)

            h1s = [h for h in s.headings if h[0] == 1]
            if len(h1s) == 0:
                add("major", "A11y", f"{label} has no <h1>",
                    "Heading order: " + str([h[0] for h in s.headings]), path)
            elif len(h1s) > 1:
                add("minor", "A11y", f"{label} has {len(h1s)} <h1> elements",
                    str([h[1][:40] for h in h1s]), path)
            lv = [h[0] for h in s.headings]
            for i in range(1, len(lv)):
                if lv[i] - lv[i - 1] > 1:
                    add("minor", "A11y", f"{label} skips a heading level",
                        f"h{lv[i-1]} followed by h{lv[i]} — order {lv}", path)
                    break

        for im in s.imgs:
            if "alt" not in im:
                add("major", "A11y", f"{label}: <img> without alt",
                    im.get("src", "")[:80], path)

        for inp in s.inputs:
            if inp.get("type") in ("hidden", "submit", "button"):
                continue
            has = (inp.get("aria-label") or inp.get("aria-labelledby")
                   or (inp.get("id") and inp["id"] in s.labels)
                   or inp.get("title"))
            if not has:
                add("major", "A11y", f"{label}: form field with no label",
                    f"name={inp.get('name')} type={inp.get('type')}", path)

        if s.autofocus > 1:
            add("minor", "A11y", f"{label} has {s.autofocus} autofocus elements",
                "Only one element can win; the rest are dead markup.", path)

        dupe = {i for i in s.ids if s.ids.count(i) > 1}
        if dupe:
            add("major", "Markup", f"{label} has duplicate element ids",
                ", ".join(sorted(dupe)[:6]), path)

        for a in s.links:
            href = a.get("href", "")
            if a.get("target") == "_blank" and "noopener" not in (a.get("rel") or ""):
                add("major", "Security", f"{label}: target=_blank without rel=noopener",
                    href[:70], path)
            if href in ("#", "") and "hx-" not in " ".join(a.keys()):
                add("minor", "Markup", f"{label}: link with empty href", "", path)

    # ── 3. Security headers ─────────────────────────────────────────────
    st, hdr, _, _ = results.get("/saved", (0, {}, "", 0))
    if st == 200:
        want = {
            "x-content-type-options": ("minor", "Stops MIME sniffing."),
            "referrer-policy": ("minor", "Controls what leaks to arxiv.org on outbound clicks."),
            "content-security-policy": ("minor", "Defence in depth against injected script."),
        }
        low = {k.lower() for k in hdr}
        for h, (sev, why) in want.items():
            if h not in low:
                add(sev, "Security", f"No {h} header", why, "all pages")

    # ── 4. CSS hygiene ──────────────────────────────────────────────────
    css = open("app/static/styles.css").read()
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)[\s,{:.\[]", stripped))
    tmpl = ""
    import glob
    for f in glob.glob("app/templates/**/*.html", recursive=True):
        tmpl += open(f).read()
    tmpl += open("app/static/app.js").read()
    used = set(re.findall(r'class="([^"]*)"', tmpl))
    used_tokens = set()
    for u in used:
        used_tokens |= set(re.findall(r"[a-zA-Z][\w-]*", u))
    used_tokens |= set(re.findall(r"classList\.(?:add|remove|toggle|contains)\('([\w-]+)'", tmpl))
    used_tokens |= set(re.findall(r'querySelector(?:All)?\([\'"]\.([\w-]+)', tmpl))

    ignore = {"htmx-request", "htmx-indicator", "sr-only", "hidden", "is-out",
              "is-leaving", "is-clamped", "just-saved", "watched"}
    orphan = sorted(c for c in defined - used_tokens - ignore
                    if not c.startswith(("cat-", "tone-", "skel", "btn", "stack",
                                         "coll", "ob-", "seed", "card", "feed",
                                         "issue", "sec-", "anchor", "interest",
                                         "meter", "strength", "toast", "empty",
                                         "botnav", "nav", "brand", "icon", "pill",
                                         "row", "wrap", "search", "ai-", "notice",
                                         "spinner", "grow", "muted", "small",
                                         "mono", "center", "scroll", "shell",
                                         "topbar", "skip", "main", "page", "q",
                                         "chip", "rank", "seq", "stat", "plot",
                                         "bar", "mean", "xaxis", "table", "warn",
                                         "eyebrow", "masthead", "dateline",
                                         "thesis", "chart", "entry", "col",
                                         "line", "follow", "back", "undo", "msg",
                                         "lead", "rest", "label", "name", "desc",
                                         "n", "sep", "i")))
    if orphan:
        add("minor", "Hygiene", f"{len(orphan)} CSS classes defined but never used",
            ", ".join(orphan[:14]) + ("…" if len(orphan) > 14 else ""),
            "app/static/styles.css")

    # ── 5. Contrast of the real token pairs ─────────────────────────────
    def tokens(block):
        m = re.search(block + r"\s*\{(.*?)\n\}", css, re.S)
        if not m:
            return {}
        return dict(re.findall(r"(--[\w-]+):\s*(#[0-9A-Fa-f]{3,8})", m.group(1)))

    light = tokens(r"^:root")
    dark = tokens(r':root\[data-theme="dark"\]')
    for name, t in (("light", light), ("dark", dark)):
        if not t:
            continue
        pairs = [
            ("--text",   "--surface", 4.5, "body text on a card"),
            ("--text-2", "--surface", 4.5, "secondary text on a card"),
            ("--text-3", "--surface", 4.5, "meta text on a card"),
            ("--text-3", "--bg",      4.5, "meta text on the page"),
            ("--accent", "--surface", 4.5, "accent text / links"),
            ("--muted" if "--muted" in t else "--text-3", "--bg", 4.5, "muted on page"),
        ]
        for fg, bgk, need, what in pairs:
            if fg in t and bgk in t:
                r = ratio(t[fg], t[bgk])
                if r < 3.0:
                    add("major", "A11y", f"Contrast fails badly ({name}): {what}",
                        f"{fg} {t[fg]} on {bgk} {t[bgk]} = {r:.2f}:1, "
                        f"WCAG AA needs {need}:1", "styles.css")
                elif r < need:
                    add("minor", "A11y", f"Contrast below AA ({name}): {what}",
                        f"{fg} {t[fg]} on {bgk} {t[bgk]} = {r:.2f}:1 "
                        f"(AA {need}:1; ok only for text ≥18.66px bold / 24px)",
                        "styles.css")

    # ── 6. Payload ──────────────────────────────────────────────────────
    for asset in ("/static/styles.css", "/static/app.js", "/static/htmx.min.js"):
        st, hdr, body, dt = fetch(asset, cookie=False)
        if st != 200:
            add("major", "Availability", f"{asset} returned {st}", "", asset)
            continue
        kb = len(body.encode()) / 1024
        enc = (hdr.get("Content-Encoding") or "none")
        cache = hdr.get("Cache-Control", "")
        if kb > 45 and enc == "none":
            add("minor", "Performance", f"{asset} served uncompressed ({kb:.0f} KB)",
                "No Content-Encoding. gzip typically cuts CSS/JS by 70-80%.", asset)
        if not cache:
            add("minor", "Performance", f"{asset} has no Cache-Control",
                "Every page load re-downloads it; there is no cache busting in "
                "the URL either, so a long max-age would need a version query.",
                asset)

    print(json.dumps(findings, indent=1))
    order = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(key=lambda f: order[f["sev"]])
    with open("audit.json", "w") as fh:
        json.dump(findings, fh, indent=1)
    from collections import Counter
    print("\n# summary:", dict(Counter(f["sev"] for f in findings)), file=sys.stderr)


main()
