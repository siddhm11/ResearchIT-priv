"""
Clean an SVG figure down to drawing elements, so it can be served from our origin.

arXiv HTML pages carry vector figures two ways: as .svg files that an <object>
points to (matplotlib and Inkscape exports), and as <svg> drawn inline in the
page (LaTeXML's rendering of TikZ and pgfplots, with axis labels as HTML and
MathML inside <foreignObject>). In 50 papers from September 2026, 118 of 333
figure images were one of the two.

SVG is code: it can carry scripts, event handlers and links. Three layers keep
that out, and this module is the first:

  1. clean() and clean_tree() rebuild the document from an allow-list of SVG,
     HTML and MathML elements. They drop every on* attribute, every link that is
     not to an id in the same document (#...), every url(...) that is not
     url(#...), and <image> data other than inline PNG, JPEG, GIF or WebP.
  2. app/routers/figures.py serves the result with a Content-Security-Policy
     that blocks scripts and sandboxes the file if it is opened directly.
  3. Pages show figures only through <img>, where browsers never run scripts.

Inline SVG reaches clean_tree() from html.parser, which lowercases names, so
SVG's camelCase element and attribute names are restored from the tables below
(the lists in the HTML standard's "adjust SVG attributes" step).

Pure and synchronous: call through asyncio.to_thread.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XHTML_NS = "http://www.w3.org/1999/xhtml"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
XML_NS = "http://www.w3.org/XML/1998/namespace"

MAX_BYTES = 5 * 1024 * 1024
MAX_OUT_BYTES = 4 * 1024 * 1024
MAX_ELEMENTS = 100_000
MAX_DEPTH = 200

ET.register_namespace("", SVG_NS)
ET.register_namespace("h", XHTML_NS)
ET.register_namespace("m", MATHML_NS)

SVG_ELEMENTS = {
    "svg", "g", "defs", "symbol", "use", "path", "rect", "circle", "ellipse", "line", "polyline",
    "polygon", "text", "tspan", "textPath", "title", "desc", "clipPath", "mask", "pattern", "marker",
    "linearGradient", "radialGradient", "stop", "image", "style", "switch", "foreignObject", "filter",
    "feBlend", "feColorMatrix", "feComponentTransfer", "feComposite", "feDropShadow", "feFlood",
    "feFuncA", "feFuncB", "feFuncG", "feFuncR", "feGaussianBlur", "feMerge", "feMergeNode",
    "feMorphology", "feOffset",
}
HTML_ELEMENTS = {"span", "div", "p", "br", "b", "i", "u", "s", "em", "strong", "sub", "sup", "small", "code"}
MATHML_ELEMENTS = {
    "math", "semantics", "mrow", "mi", "mo", "mn", "ms", "mtext", "mspace", "msub", "msup", "msubsup",
    "mfrac", "msqrt", "mroot", "mstyle", "mover", "munder", "munderover", "mtable", "mtr", "mtd",
    "mpadded", "mphantom", "menclose", "mmultiscripts", "mprescripts", "none", "merror",
}
SVG_CASE = {name.lower(): name for name in SVG_ELEMENTS}
SVG_ATTR_CASE = {name.lower(): name for name in (
    "attributeName", "baseFrequency", "baseProfile", "calcMode", "clipPathUnits", "diffuseConstant",
    "edgeMode", "filterUnits", "glyphRef", "gradientTransform", "gradientUnits", "kernelMatrix",
    "kernelUnitLength", "keyPoints", "keySplines", "keyTimes", "lengthAdjust", "limitingConeAngle",
    "markerHeight", "markerUnits", "markerWidth", "maskContentUnits", "maskUnits", "numOctaves",
    "pathLength", "patternContentUnits", "patternTransform", "patternUnits", "pointsAtX", "pointsAtY",
    "pointsAtZ", "preserveAlpha", "preserveAspectRatio", "primitiveUnits", "refX", "refY",
    "repeatCount", "repeatDur", "requiredExtensions", "requiredFeatures", "specularConstant",
    "specularExponent", "spreadMethod", "startOffset", "stdDeviation", "stitchTiles", "surfaceScale",
    "systemLanguage", "tableValues", "targetX", "targetY", "textLength", "viewBox", "viewTarget",
    "xChannelSelector", "yChannelSelector", "zoomAndPan")}
# Attributes that load or point at something, whatever element carries them.
BLOCKED_ATTRS = {"src", "srcdoc", "data", "action", "formaction", "background", "lowsrc", "dynsrc", "ping"}

_DATA_IMAGE = re.compile(r"^data:image/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=\s]+$", re.I)
_FRAGMENT = re.compile(r"^#[A-Za-z_][\w.:\-]*$")
_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I | re.S)
_UNSAFE = ("javascript:", "vbscript:", "expression(", "@import", "behavior:", "-moz-binding")
_DOCTYPE = re.compile(r"<!DOCTYPE[^\[>]*>", re.I)
_LENGTH = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(px|pt)?\s*$")


class NotSVG(ValueError):
    pass


def _split(name: str) -> tuple[str, str]:
    """(namespace or prefix, local name) for '{ns}local', 'prefix:local' or 'local'."""
    if name.startswith("{"):
        ns, _, local = name[1:].partition("}")
        return ns, local
    if ":" in name:
        prefix, _, local = name.partition(":")
        return prefix, local
    return "", name


def _safe_value(value: str) -> bool:
    """No script schemes, no CSS escapes, and every url(...) points inside the document.

    A backslash is refused outright: CSS escapes can spell url( or a scheme so
    that no plain-text check sees it (u\\72 l(...)). Figures never need one."""
    if "\\" in value:
        return False
    squashed = re.sub(r"[\s\x00-\x1f]", "", value).lower()
    if any(u in squashed for u in _UNSAFE):
        return False
    return all(m.group(2).strip().startswith("#") for m in _URL.finditer(value))


def _attrs(attrib: dict, ctx: str, name: str) -> dict:
    out = {}
    for raw, value in attrib.items():
        ns, local = _split(raw)
        lname = local.lower()
        value = "" if value is None else str(value)
        if lname.startswith("on"):
            continue
        if lname == "href":                              # href and xlink:href alike
            v = value.strip()
            if _FRAGMENT.match(v) or (name == "image" and _DATA_IMAGE.match(v)):
                out["href"] = v
            continue
        if ns in (XML_NS, "xml") and lname in ("space", "lang"):
            out[f"{{{XML_NS}}}{lname}"] = value
            continue
        if ns or lname in BLOCKED_ATTRS or lname.startswith("xmlns") or not _safe_value(value):
            continue
        out[SVG_ATTR_CASE.get(lname, local) if ctx == "svg" else lname] = value
    return out


def _tag(src_tag: str, ctx: str) -> tuple[str, str, str] | None:
    """(namespaced tag, local name, context for children) or None to drop the element."""
    local = _split(src_tag)[1] if isinstance(src_tag, str) else ""
    lower = local.lower()
    if ctx == "html" and lower == "math":
        ctx = "mathml"
    elif ctx == "html" and lower == "svg":
        ctx = "svg"
    if ctx == "svg":
        name = "g" if lower == "a" else SVG_CASE.get(lower)
        if name is None:
            return None
        return f"{{{SVG_NS}}}{name}", name, ("html" if name == "foreignObject" else "svg")
    if ctx == "html":
        name = "span" if lower == "a" else (lower if lower in HTML_ELEMENTS else None)
        return (f"{{{XHTML_NS}}}{name}", name, "html") if name else None
    if lower in MATHML_ELEMENTS:                         # mathml
        return f"{{{MATHML_NS}}}{lower}", lower, "mathml"
    return None


class _Budget:
    __slots__ = ("left",)

    def __init__(self, n: int):
        self.left = n


def _rebuild(src: ET.Element, ctx: str, depth: int, budget: _Budget) -> ET.Element | None:
    picked = _tag(src.tag, ctx) if depth <= MAX_DEPTH else None
    if picked is None:
        return None
    tag, name, child_ctx = picked
    budget.left -= 1
    if budget.left < 0:
        raise NotSVG(f"over {MAX_ELEMENTS} elements")
    own_ctx = {SVG_NS: "svg", XHTML_NS: "html", MATHML_NS: "mathml"}[_split(tag)[0]]
    el = ET.Element(tag, _attrs(src.attrib, own_ctx, name))
    if name == "style":
        css = "".join(src.itertext())
        if not _safe_value(css):
            return None
        el.text = css
        return el
    el.text = src.text
    last = None
    for child in src:
        kept = _rebuild(child, child_ctx, depth + 1, budget)
        if kept is not None:
            kept.tail = child.tail
            el.append(kept)
            last = kept
        elif child.tail:                                 # keep text that followed a dropped element
            if last is not None:
                last.tail = (last.tail or "") + child.tail
            else:
                el.text = (el.text or "") + child.tail
    return el


def _number(value: str | None) -> str | None:
    m = _LENGTH.match(value or "")
    return m.group(1) if m else None


def clean_tree(root: ET.Element) -> bytes:
    """Cleaned SVG bytes for a parsed <svg> element, from a file or an HTML page."""
    if _split(root.tag)[1].lower() != "svg":
        raise NotSVG("the root is not <svg>")
    out = _rebuild(root, "svg", 0, _Budget(MAX_ELEMENTS))
    if out is None:
        raise NotSVG("nothing left after cleaning")
    # <img> needs a size and a viewBox to scale the drawing instead of cropping it.
    box = out.get("viewBox")
    w, h = _number(out.get("width")), _number(out.get("height"))
    if not box and w and h:
        out.set("viewBox", f"0 0 {w} {h}")
    elif box and not (out.get("width") and out.get("height")):
        parts = box.replace(",", " ").split()
        if len(parts) == 4:
            out.set("width", parts[2])
            out.set("height", parts[3])
    data = ET.tostring(out, encoding="unicode").encode("utf-8")
    if len(data) > MAX_OUT_BYTES:
        raise NotSVG(f"{len(data)} bytes after cleaning")
    return data


def clean(raw: bytes) -> bytes:
    """Cleaned SVG bytes for an .svg file. Raises NotSVG for anything else."""
    if len(raw) > MAX_BYTES:
        raise NotSVG(f"{len(raw)} bytes")
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    if re.search(r"<!ENTITY", text, re.I):
        raise NotSVG("entity declarations")
    # matplotlib writes a DOCTYPE naming the SVG 1.1 DTD; only one with an
    # internal subset could declare anything, and those are refused.
    text = _DOCTYPE.sub("", text)
    if re.search(r"<!DOCTYPE", text, re.I):
        raise NotSVG("a DOCTYPE with an internal subset")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise NotSVG(f"not XML: {e}") from e
    return clean_tree(root)
