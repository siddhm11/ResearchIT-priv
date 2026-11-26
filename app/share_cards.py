"""Share cards: the 1200x630 picture a link to a paper, topic or reading list
shows in Slack, X, LinkedIn, Discord, WhatsApp or iMessage.

Before this every page shared one picture (app/static/og.png), or a paper's
bare first figure with no title. A card that names the paper is what makes a
pasted link worth clicking (UI review 2026-11-24, batch 3).

Drawn with Pillow from data the site already holds: no model call, no network
fetch, so a link-preview bot asking for one costs a few milliseconds of CPU and
leaves no trace (CLAUDE.md §3.11, docs/AEO-PLAN.md). The look repeats
scripts/build_og_image.py: graph paper, ink text, the green accent, the book
mark from the top bar.

Fonts are ones the image already holds, so the container needs no font
package: KaTeX's serif (app/static/vendor/katex-*, SIL OFL) for titles, names
and sentences, its math italic for Greek letters, and Pillow's bundled Aileron
for labels. None of them carries accented Latin letters, so a letter no font
has is drawn as its base letter (é as e) rather than as an empty box.
"""
from __future__ import annotations

import io
import re
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
PAPER, INK, MUTED, ACCENT = (242, 238, 230), (22, 20, 15), (95, 88, 76), (44, 86, 56)
ACCENT_BG, RULE = (228, 239, 229), (214, 207, 193)
GRID, GRID_MAJOR = (233, 228, 218), (223, 216, 203)
PAD = 72

_KATEX = next(iter(sorted((Path(__file__).parent / "static" / "vendor").glob("katex-*/fonts"))), None)
_FILES = {
    "serif": "KaTeX_Main-Regular.woff2",
    "bold": "KaTeX_Main-Bold.woff2",
    "greek": "KaTeX_Math-Italic.woff2",
}
# Which fonts a style tries, in order, for each character.
_CHAINS = {"serif": ("serif", "greek", "ui"), "bold": ("bold", "greek", "serif", "ui"),
           "ui": ("ui", "serif", "greek")}
_MISSING = "\U0010fff0"          # a code point no font has: its mask is the "no glyph" box


@lru_cache(maxsize=64)
def _font(face: str, size: int) -> ImageFont.FreeTypeFont:
    if face != "ui" and _KATEX is not None and (_KATEX / _FILES[face]).exists():
        try:
            return ImageFont.truetype(str(_KATEX / _FILES[face]), size)
        except OSError:          # a FreeType built without WOFF2 support
            pass
    return ImageFont.load_default(size)


@lru_cache(maxsize=4096)
def _has(face: str, ch: str) -> bool:
    f = _font(face, 40)
    if ch.isspace():
        return True
    return bytes(f.getmask(ch)) != _missing_mask(face)


@lru_cache(maxsize=8)
def _missing_mask(face: str) -> bytes:
    return bytes(_font(face, 40).getmask(_MISSING))


# LaTeX that turns up in arXiv titles: $\alpha$-divergence, $O(n \log n)$.
_GREEK = {name: chr(cp) for name, cp in (
    ("alpha", 0x3B1), ("beta", 0x3B2), ("gamma", 0x3B3), ("delta", 0x3B4), ("epsilon", 0x3B5),
    ("varepsilon", 0x3B5), ("zeta", 0x3B6), ("eta", 0x3B7), ("theta", 0x3B8), ("kappa", 0x3BA),
    ("lambda", 0x3BB), ("mu", 0x3BC), ("nu", 0x3BD), ("xi", 0x3BE), ("pi", 0x3C0), ("rho", 0x3C1),
    ("sigma", 0x3C3), ("tau", 0x3C4), ("phi", 0x3C6), ("varphi", 0x3C6), ("chi", 0x3C7),
    ("psi", 0x3C8), ("omega", 0x3C9), ("Gamma", 0x393), ("Delta", 0x394), ("Theta", 0x398),
    ("Lambda", 0x39B), ("Pi", 0x3A0), ("Sigma", 0x3A3), ("Phi", 0x3A6), ("Psi", 0x3A8),
    ("Omega", 0x3A9), ("ell", 0x2113), ("infty", 0x221E), ("times", 0xD7), ("to", 0x2192),
    ("rightarrow", 0x2192), ("leq", 0x2264), ("geq", 0x2265), ("pm", 0xB1), ("cdot", 0xB7))}
_COMMAND = re.compile(r"\\([A-Za-z]+)")


def clean(text: str, style: str = "serif") -> str:
    """Text as the card can draw it: LaTeX markup unwrapped, whitespace folded,
    and every character some font in the style's chain has (else its base letter)."""
    text = _COMMAND.sub(lambda m: _GREEK.get(m.group(1), m.group(1) if m.group(1) not in
                                             ("mathrm", "mathbf", "mathcal", "textbf", "emph", "text", "it", "bf")
                                             else ""), text or "")
    text = re.sub(r"[${}^_]", "", text)
    out = []
    for ch in " ".join(text.split()):
        if any(_has(f, ch) for f in _CHAINS[style]):
            out.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append(base if base and all(any(_has(f, c) for f in _CHAINS[style]) for c in base) else "")
    return " ".join("".join(out).split())


def _runs(text: str, style: str) -> list[tuple[str, str]]:
    """(face, text) pieces: each character in the first face of the chain that has it."""
    runs: list[tuple[str, str]] = []
    for ch in text:
        face = next((f for f in _CHAINS[style] if _has(f, ch)), _CHAINS[style][0])
        if runs and (runs[-1][0] == face or ch == " "):
            runs[-1] = (runs[-1][0], runs[-1][1] + ch)
        else:
            runs.append((face, ch))
    return runs


def text_w(text: str, size: int, style: str = "serif") -> float:
    return sum(_font(face, size).getlength(t) for face, t in _runs(text, style))


def draw(d: ImageDraw.ImageDraw, xy, text: str, size: int, fill, style: str = "serif") -> float:
    """Draws already-cleaned text; returns its width."""
    x, y = xy
    for face, t in _runs(text, style):
        d.text((x, y), t, font=_font(face, size), fill=fill)
        x += _font(face, size).getlength(t)
    return x - xy[0]


def wrap(text: str, size: int, width: float, max_lines: int, style: str = "serif") -> list[str]:
    """Greedy word wrap to `width` px, at most `max_lines`; the last line ends
    in an ellipsis when text is left over. A word wider than a line is cut."""
    words = clean(text, style).split(" ")
    lines: list[str] = []
    cur = ""
    i = 0
    while i < len(words) and len(lines) < max_lines:
        w = words[i]
        trial = f"{cur} {w}".strip()
        if text_w(trial, size, style) <= width:
            cur, i = trial, i + 1
            continue
        if not cur:                                   # one word wider than the line
            while w and text_w(w + "…", size, style) > width:
                w = w[:-1]
            lines.append(w + "…")
            i += 1
            continue
        lines.append(cur)
        cur = ""
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if i < len(words) and lines:                      # text left over: ellipsis
        last = lines[-1].rstrip("…")
        while last and text_w(last + "…", size, style) > width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last.rstrip(" ,;:—-") + "…"
    return [line for line in lines if line]


def _base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 24):
        d.line([(x, 0), (x, H)], fill=GRID_MAJOR if x % 120 == 0 else GRID)
    for y in range(0, H, 24):
        d.line([(0, y), (W, y)], fill=GRID_MAJOR if y % 120 == 0 else GRID)
    # The book mark from base.html's top bar, then the name.
    s, ox, oy = 1.9, PAD, 50
    def p(x, y): return (ox + x * s, oy + y * s)
    d.line([p(4, 20.5), p(4, 5.5), p(6.5, 3), p(19, 3), p(19, 18), p(6.5, 18), p(4, 20.5)], fill=ACCENT, width=4, joint="curve")
    d.line([p(6.5, 18), p(19, 18), p(19, 21), p(6.5, 21), p(4, 20.5)], fill=ACCENT, width=4, joint="curve")
    d.line([p(9, 8), p(15, 8)], fill=ACCENT, width=3)
    d.line([p(9, 11.5), p(13, 11.5)], fill=ACCENT, width=3)
    draw(d, (ox + 52, oy + 2), "ResearchIT", 32, INK, "bold")
    d.rectangle([0, H - 12, W, H], fill=ACCENT)
    return img, d


def _chip(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int = 22) -> float:
    text = clean(text, "ui")
    w = text_w(text, size, "ui") + 28
    d.rounded_rectangle([x, y, x + w, y + size + 18], radius=8, fill=ACCENT_BG, outline=(202, 223, 205))
    draw(d, (x + 14, y + 8), text, size, ACCENT, "ui")
    return x + w + 12


def _title_block(d, x: float, y: float, title: str, width: float,
                 sizes=(66, 58, 50, 44), max_lines: int = 4) -> float:
    """Draws the title at the largest size that fits in `max_lines`; returns the y below it."""
    for size in sizes:
        lines = wrap(title, size, width, max_lines, "bold")
        if not (lines and lines[-1].endswith("…")):
            break
    for line in lines:
        draw(d, (x, y), line, size, INK, "bold")
        y += int(size * 1.16)
    return y


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _fit_figure(raw: bytes, box_w: int, box_h: int) -> Image.Image | None:
    try:
        fig = Image.open(io.BytesIO(raw))
        fig.load()
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    fig = fig.convert("RGBA")
    bg = Image.new("RGBA", fig.size, (255, 255, 255, 255))
    fig = Image.alpha_composite(bg, fig).convert("RGB")
    fig.thumbnail((box_w, box_h), Image.LANCZOS)
    return fig


def authors_line(authors: list[str]) -> str:
    if not authors:
        return ""
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{', '.join(authors[:2])} and {len(authors) - 2} others"


def paper_card(*, title: str, authors: list[str], category: str = "", year: str = "",
               cited: int | None = None, arxiv_id: str = "", hook: str = "",
               figure: bytes | None = None) -> bytes:
    """A paper: its area, title, authors and one line on what it does; the
    lead figure on the right when one is already saved."""
    img, d = _base()
    fig = _fit_figure(figure, 420, 400) if figure else None
    text_w_ = W - 2 * PAD - (470 if fig else 0)
    y, x = 128, PAD
    if category:
        x = _chip(d, x, y, category)
    if year:
        _chip(d, x, y, str(year))
    y += 70
    y = _title_block(d, PAD, y, title, text_w_, sizes=(66, 58, 50, 44, 40),
                     max_lines=4 if (fig or not hook) else 3)
    y += 14
    for a in wrap(authors_line(authors), 27, text_w_, 1):
        draw(d, (PAD, y), a, 27, MUTED)
        y += 40
    if hook and y < H - 140:
        y += 8
        for h in wrap(hook, 27, text_w_, 2 if y < H - 180 else 1):
            draw(d, (PAD, y), h, 27, INK)
            y += 37
    foot = " · ".join(p for p in (f"arXiv:{arxiv_id}" if arxiv_id else "",
                                   f"{cited:,} cited" if cited else "") if p)
    if foot:
        draw(d, (PAD, H - 68), foot, 23, MUTED, "ui")
    if fig:
        # A white frame hugging the figure, centred in the right-hand column.
        fx = W - PAD - 420 + (420 - fig.width) // 2
        fy = 130 + (400 - fig.height) // 2
        d.rounded_rectangle([fx - 20, fy - 20, fx + fig.width + 20, fy + fig.height + 20],
                            radius=14, fill=(255, 255, 255), outline=RULE)
        img.paste(fig, (fx, fy))
    return _png(img)


def _spark(d, x: float, y: float, w: float, h: float, values: list[float]) -> None:
    if len(values) < 2:
        return
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = [(x + i * w / (len(values) - 1), y + h - (v - lo) / span * h) for i, v in enumerate(values)]
    d.line(pts, fill=ACCENT, width=5, joint="curve")
    d.ellipse([pts[-1][0] - 8, pts[-1][1] - 8, pts[-1][0] + 8, pts[-1][1] + 8], fill=ACCENT)


def one_in(share: float) -> str:
    return f"1 in {round(1 / share)}" if share > 0 else "none"


def topic_card(*, name: str, label: str, from_share: float, to_share: float,
               rolling: list[float], papers: list[str], since: str = "") -> bytes:
    """A trending topic: its name, the change in its share of Hugging Face's
    Daily Papers, the curve, and its most upvoted papers."""
    img, d = _base()
    x = _chip(d, PAD, 128, label)
    if since:
        draw(d, (x, 136), clean(f"since {since}", "ui"), 22, MUTED, "ui")
    y = _title_block(d, PAD, 196, name, W - 2 * PAD, sizes=(74, 62, 52), max_lines=2)
    draw(d, (PAD, y + 6), f"{from_share * 100:.1f}% → {to_share * 100:.1f}%", 48, ACCENT, "bold")
    draw(d, (PAD, y + 68), f"of Hugging Face Daily Papers: {one_in(from_share)} → {one_in(to_share)}",
         24, MUTED)
    _spark(d, W - PAD - 360, y + 6, 360, 90, rolling)
    yy = y + 130
    for t in papers[:3]:
        if yy > H - 72:
            break
        for line in wrap(t, 27, W - 2 * PAD - 30, 1):
            d.ellipse([PAD, yy + 13, PAD + 9, yy + 22], fill=ACCENT)
            draw(d, (PAD + 26, yy), line, 27, INK)
        yy += 42
    return _png(img)


def list_card(*, title: str, kicker: str, blurb: str, items: list[str]) -> bytes:
    """A list (a reading list, Lasting): its title, a line on it, its first entries."""
    img, d = _base()
    _chip(d, PAD, 128, kicker)
    y = _title_block(d, PAD, 196, title, W - 2 * PAD, sizes=(66, 56, 48), max_lines=2)
    for line in wrap(blurb, 28, W - 2 * PAD, 2):
        draw(d, (PAD, y + 4), line, 28, MUTED)
        y += 40
    y += 22
    for i, t in enumerate(items[:3], 1):
        if y > H - 80:
            break
        draw(d, (PAD, y), str(i), 27, ACCENT, "bold")
        for line in wrap(t, 27, W - 2 * PAD - 44, 1):
            draw(d, (PAD + 40, y), line, 27, INK)
        y += 44
    return _png(img)


class Cache:
    """The last `size` cards drawn, by key. A card is ~40-90 KB."""

    def __init__(self, size: int = 256):
        self.size, self._d = size, OrderedDict()

    def get(self, key: str) -> bytes | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, data: bytes) -> None:
        self._d[key] = data
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)


cache = Cache()
