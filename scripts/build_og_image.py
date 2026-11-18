#!/usr/bin/env python3
"""Draw app/static/og.png, the 1200x630 picture a shared ResearchIT link shows.

    python scripts/build_og_image.py [--font /path/to/serif.ttc]

Paper pages with a saved figure use that figure instead (app/routers/paper.py).
The picture repeats the app's own look: graph paper, ink text, the green accent
and the book mark from the top bar. Run again after changing the wording.
Charter (Bitstream's free licence) ships with macOS; any serif works.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "app" / "static" / "og.png"
W, H = 1200, 630
PAPER, INK, MUTED, ACCENT = (242, 238, 230), (22, 20, 15), (95, 88, 76), (44, 86, 56)
GRID, GRID_MAJOR = (229, 224, 213), (219, 212, 198)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="/System/Library/Fonts/Supplemental/Charter.ttc")
    args = ap.parse_args()
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 24):
        d.line([(x, 0), (x, H)], fill=GRID_MAJOR if x % 120 == 0 else GRID)
    for y in range(0, H, 24):
        d.line([(0, y), (W, y)], fill=GRID_MAJOR if y % 120 == 0 else GRID)

    def font(size: int, index: int = 0):
        return ImageFont.truetype(args.font, size, index=index)

    # The book mark from base.html's top bar, scaled up.
    s, ox, oy = 4.2, 96, 92
    def p(x, y): return (ox + x * s, oy + y * s)
    d.line([p(4, 20.5), p(4, 5.5), p(6.5, 3), p(19, 3), p(19, 18), p(6.5, 18), p(4, 20.5)], fill=ACCENT, width=8, joint="curve")
    d.line([p(6.5, 18), p(19, 18), p(19, 21), p(6.5, 21), p(4, 20.5)], fill=ACCENT, width=8, joint="curve")
    d.line([p(9, 8), p(15, 8)], fill=ACCENT, width=7)
    d.line([p(9, 11.5), p(13, 11.5)], fill=ACCENT, width=7)
    d.text((ox + 26 * s, oy + 14), "ResearchIT", font=font(60, 3), fill=INK)

    d.text((96, 270), "Find the papers that matter", font=font(76, 3), fill=INK)
    d.text((96, 360), "to your work.", font=font(76, 3), fill=INK)
    d.text((96, 478), "Search 2 million arXiv papers in plain words, or get a feed", font=font(34), fill=MUTED)
    d.text((96, 522), "that learns your separate interests from what you save.", font=font(34), fill=MUTED)
    d.rectangle([0, H - 14, W, H], fill=ACCENT)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
