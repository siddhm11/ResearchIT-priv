"""
Turn an arXiv figure image into a small WebP for the feed and paper page.

Originals run to 2.1 MB (September 2026 sample: median 182 KB, 90th
percentile 919 KB), too heavy for a feed. Pure and synchronous: call it through
asyncio.to_thread, since a large PNG takes long enough to stall the event loop.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps

MAX_WIDTH = 800
TARGET_BYTES = 150 * 1024
QUALITIES = (80, 70, 60)
MAX_PIXELS = 50_000_000

# Pillow's own bomb guard warns past this and refuses at twice it; make() also
# refuses anything over it outright, before decoding.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class NotAnImage(ValueError):
    pass


def _flatten(im: Image.Image) -> Image.Image:
    """RGB with any transparency composited onto white (figures sit on paper)."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def make(raw: bytes, max_width: int = MAX_WIDTH) -> bytes:
    """WebP bytes: first frame, upright, flattened, at most `max_width` wide.

    Tries quality 80, then 70, then 60 to land under 150 KB, and returns the
    smallest it made if none does. Never enlarges a small image.
    """
    try:
        im = Image.open(io.BytesIO(raw))
    except Exception as e:                          # PIL raises several types here
        raise NotAnImage(str(e)) from e
    with im:
        if im.width * im.height > MAX_PIXELS:
            raise NotAnImage(f"{im.width}x{im.height} is over {MAX_PIXELS} pixels")
        im.seek(0)
        frame = ImageOps.exif_transpose(im)
        frame = _flatten(frame)
        if frame.width > max_width:
            height = max(1, round(frame.height * max_width / frame.width))
            frame = frame.resize((max_width, height), Image.Resampling.LANCZOS)
        best = b""
        for quality in QUALITIES:
            buf = io.BytesIO()
            frame.save(buf, "WEBP", quality=quality, method=4)
            data = buf.getvalue()
            if not best or len(data) < len(best):
                best = data
            if len(data) <= TARGET_BYTES:
                return data
        return best
