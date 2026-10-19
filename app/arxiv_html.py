"""
Parse one arXiv HTML page (LaTeXML output) into what the redesign needs.

Pure: no I/O, no network, no database. app/paper_source_svc.py fetches the page
and stores the result. Run it in a thread (`asyncio.to_thread`): a 1.4 MB page
takes long enough to stall the event loop.

The markup rules, checked on six pages fetched on 2026-10-17 and pinned by the
fixtures in tests/fixtures/arxiv_html/:

    licence      <a id="license-tr">License: CC BY 4.0</a> inside div#infobox
    version      div#watermark-tr: "arXiv:2609.31507v1 [cs.CV] 25 Sep 2026"
    title        h1.ltx_title_document
    abstract     div.ltx_abstract, its p.ltx_p
    sections     <section id=… class="ltx_section|ltx_subsection|…|ltx_appendix">,
                 title from the first h2-h6.ltx_title inside, number from its
                 span.ltx_tag
    paragraphs   div.ltx_para with an id such as S3.SS4.SSS0.Px1.p1
    maths        <math alttext="TeX"> with MathML and a TeX annotation inside;
                 only alttext is kept, or every formula would appear three times
    figures      figure.ltx_figure (sub-panels nest inside); label from
                 span.ltx_tag_figure; images are img.ltx_graphics with a path
                 relative to https://arxiv.org/html/ such as
                 "2609.31507v1/overview-final.png"; a missing image has src=""
                 and class ltx_missing_image; alt text is always "Refer to
                 caption", so captions serve as alt text instead
    vector       <object type="image/svg+xml" data="2508.12461v3/x.svg"> for
                 an SVG file (kind "svg"), or <svg class="ltx_picture"> drawn
                 inline (kind "svg-inline", no src: inline_svgs() lifts the
                 drawing out of the page when the figure is first wanted)
    tables       figure.ltx_table, kept only as citation targets
    skipped      nav (contents), section.ltx_bibliography, header, footer,
                 scripts and styles, footnote bodies, images under /static/
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser

PARSER_VERSION = 2
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_PARAGRAPHS = 600
MAX_FIGURES = 80
MAX_CAPTION_CHARS = 1000

_VOID = {"img", "br", "hr", "meta", "link", "input", "source", "col", "area", "wbr", "base", "embed"}
_SECTION_KINDS = ("ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_paragraph",
                  "ltx_appendix", "ltx_chapter", "ltx_subparagraph")
_IMAGE_PATH = re.compile(r"^[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*$")
_WATERMARK = re.compile(r"arXiv:\s*(\S+?)(v\d+)\b")
_LABEL_PREFIX = re.compile(r"^\s*(?:Figure|Fig\.|Table)\s*[\w.]*\s*[:.]?\s*", re.I)


@dataclass
class Section:
    anchor: str
    number: str
    title: str
    level: int
    kind: str


@dataclass
class Paragraph:
    anchor: str
    section: str
    text: str
    refs: list[str] = field(default_factory=list)


@dataclass
class Image:
    src: str
    width: int | None
    height: int | None
    panel: str = ""
    missing: bool = False
    kind: str = "img"                  # img, svg (a file) or svg-inline (drawn in the page)


@dataclass
class Figure:
    anchor: str
    label: str
    caption: str
    images: list[Image] = field(default_factory=list)

    def first_image(self) -> Image | None:
        return next((i for i in self.images if not i.missing and (i.src or i.kind == "svg-inline")), None)


@dataclass
class TableRef:
    anchor: str
    label: str
    caption: str


@dataclass
class ParsedPaper:
    arxiv_id: str
    status: str = "ok"
    error: str = ""
    version: str = ""
    licence_text: str = ""
    title: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[TableRef] = field(default_factory=list)
    parser_version: int = PARSER_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value else None
    except ValueError:
        return None


class _Collector:
    """Accumulates the text of one element."""
    __slots__ = ("parts",)

    def __init__(self):
        self.parts: list[str] = []

    def add(self, text: str) -> None:
        self.parts.append(text)

    def text(self) -> str:
        return _clean("".join(self.parts))


class _Parser(HTMLParser):
    def __init__(self, paper: ParsedPaper):
        super().__init__(convert_charrefs=True)
        self.p = paper
        self.stack: list[tuple[str, str]] = []     # (tag, role)
        self.collectors: list[_Collector] = []
        self.skip_depth = 0                        # inside a skipped subtree
        self.math_depth = 0
        self.sections: list[Section] = []          # open sections, innermost last
        self.figure: Figure | None = None
        self.table: TableRef | None = None
        self.panel = ""
        self.para: Paragraph | None = None
        self.para_text: _Collector | None = None
        self.title_for: Section | None = None
        self.title_text: _Collector | None = None
        self.tag_text: _Collector | None = None
        self.caption: _Collector | None = None
        self.label: _Collector | None = None
        self.licence: _Collector | None = None
        self.watermark: _Collector | None = None
        self.doc_title: _Collector | None = None
        self.abstract: _Collector | None = None
        self.in_abstract_title = False
        self.full = False
        self.svg_depth = 0                         # inside an inline <svg>

    # ── helpers ──
    def _push(self, tag: str, role: str = "") -> None:
        self.stack.append((tag, role))

    def _start_collector(self) -> _Collector:
        c = _Collector()
        self.collectors.append(c)
        return c

    def _stop_collector(self, c: _Collector | None) -> None:
        if c is not None and c in self.collectors:
            self.collectors.remove(c)

    # ── HTMLParser hooks ──
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if self.skip_depth:
            if tag not in _VOID:
                self.skip_depth += 1
            return
        if self.math_depth:
            if tag == "math":
                self.math_depth += 1
            return
        if tag == "math":
            alt = a.get("alttext") or ""
            if alt:
                for c in self.collectors:
                    c.add(f" ${alt}$ ")
            self.math_depth = 1
            return
        if tag in ("script", "style", "nav", "header", "footer", "button") or \
                (tag == "section" and "ltx_bibliography" in classes) or \
                "ltx_note_outer" in classes:
            self.skip_depth = 1
            return
        if tag == "img":
            self._image(a, classes)
            return
        if tag in _VOID:
            return
        if tag == "object" and ((a.get("type") or "").lower() == "image/svg+xml"
                                or (a.get("data") or "").lower().endswith(".svg")):
            self._image({**a, "src": a.get("data")}, classes)
        elif tag == "svg":
            if self.svg_depth == 0 and self.figure is not None and self.table is None:
                self.figure.images.append(Image(src="", width=_int(a.get("width")), height=_int(a.get("height")),
                                                panel=self.panel, kind="svg-inline"))
            self.svg_depth += 1

        role = ""
        ident = a.get("id") or ""
        if tag == "a" and ident == "license-tr":
            role, self.licence = "licence", self._start_collector()
        elif ident == "watermark-tr":
            role, self.watermark = "watermark", self._start_collector()
        elif tag == "h1" and "ltx_title_document" in classes and not self.p.title:
            role, self.doc_title = "doc_title", self._start_collector()
        elif tag == "div" and "ltx_abstract" in classes:
            role, self.abstract = "abstract", self._start_collector()
        elif tag == "h6" and "ltx_title_abstract" in classes and self.abstract is not None:
            role, self.in_abstract_title = "abstract_title", True
            self._stop_collector(self.abstract)
        elif tag == "section" and ident and any(k in classes for k in _SECTION_KINDS):
            kind = next(k for k in _SECTION_KINDS if k in classes)[4:]
            sec = Section(anchor=ident, number="", title="", level=0, kind=kind)
            self.sections.append(sec)
            self.p.sections.append(sec)
            role = "section"
        elif re.fullmatch(r"h[2-6]", tag) and "ltx_title" in classes and self.sections \
                and self.sections[-1].title == "" and self.title_for is None:
            self.title_for = self.sections[-1]
            self.title_for.level = int(tag[1])
            role, self.title_text = "section_title", self._start_collector()
        elif tag == "span" and "ltx_tag" in classes and (self.title_text is not None or self.caption is not None):
            role, self.tag_text = "tag", _Collector()
            self.collectors.append(self.tag_text)
            if self.title_text is not None:
                self._stop_collector(self.title_text)
        elif tag == "figure" and "ltx_table" in classes and self.figure is None and self.table is None:
            self.table = TableRef(anchor=ident, label="", caption="")
            role = "table"
        elif tag == "figure" and "ltx_figure" in classes and self.table is None:
            if self.figure is None:
                if len(self.p.figures) < MAX_FIGURES:
                    self.figure = Figure(anchor=ident, label="", caption="")
                    role = "figure"
            elif "ltx_figure_panel" in classes:
                self.panel, role = ident, "panel"
        elif tag == "figcaption" and (self.figure is not None or self.table is not None) and \
                self.caption is None and not self.panel:
            role, self.caption = "caption", self._start_collector()
        elif tag == "div" and "ltx_para" in classes and ident and self.para is None \
                and self.abstract is None and not self.full:
            self.para = Paragraph(anchor=ident, section=self.sections[-1].anchor if self.sections else "", text="")
            role, self.para_text = "para", self._start_collector()
        elif tag == "a" and self.para is not None and "ltx_ref" in classes:
            href = a.get("href") or ""
            # Figure, table and section anchors only: citations (#bib.bib12)
            # say nothing about which figure a paragraph explains.
            if href.startswith("#") and len(self.para.refs) < 20:
                ref = href[1:]
                if ref and not ref.startswith("bib") and ref not in self.para.refs:
                    self.para.refs.append(ref)
        self._push(tag, role)

    def handle_startendtag(self, tag, attrs):
        if tag == "img" and not self.skip_depth and not self.math_depth:
            a = dict(attrs)
            self._image(a, (a.get("class") or "").split())
        elif tag == "math" and not self.skip_depth and not self.math_depth:
            alt = dict(attrs).get("alttext") or ""
            if alt:
                for c in self.collectors:
                    c.add(f" ${alt}$ ")

    def handle_endtag(self, tag):
        if self.skip_depth:
            if tag not in _VOID:
                self.skip_depth -= 1
            return
        if self.math_depth:
            if tag == "math":
                self.math_depth -= 1
            return
        if tag in _VOID:
            return
        if tag == "svg" and self.svg_depth:
            self.svg_depth -= 1
        # Pop to the matching open tag; a stray end tag with no opener is ignored.
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                while len(self.stack) > i:
                    self._close(*self.stack.pop())
                return

    def handle_data(self, data):
        if self.skip_depth or self.math_depth:
            return
        for c in self.collectors:
            c.add(data)

    # ── element ends ──
    def _close(self, tag: str, role: str) -> None:
        if role == "licence":
            self.p.licence_text = re.sub(r"^License:\s*", "", self.licence.text(), flags=re.I)
            self._stop_collector(self.licence)
        elif role == "watermark":
            m = _WATERMARK.search(self.watermark.text())
            if m:
                self.p.version = m.group(2)
            self._stop_collector(self.watermark)
        elif role == "doc_title":
            self.p.title = self.doc_title.text()
            self._stop_collector(self.doc_title)
        elif role == "abstract_title":
            self.in_abstract_title = False
            if self.abstract is not None:
                self.collectors.append(self.abstract)
        elif role == "abstract":
            self.p.abstract = self.abstract.text()
            self._stop_collector(self.abstract)
            self.abstract = None
        elif role == "tag":
            self._stop_collector(self.tag_text)
            text = self.tag_text.text()
            if self.title_text is not None and self.title_for is not None:
                self.title_for.number = text
                self.collectors.append(self.title_text)
            elif self.caption is not None and self.label is None:
                self.label = self.tag_text
            self.tag_text = None
        elif role == "section_title":
            self.title_for.title = self.title_text.text()
            self._stop_collector(self.title_text)
            self.title_text, self.title_for = None, None
        elif role == "section":
            if self.sections:
                self.sections.pop()
        elif role == "caption":
            label = re.sub(r"[:.]\s*$", "", self.label.text()) if self.label else ""
            text = self.caption.text()
            if label and text.startswith(label):
                text = text[len(label):].lstrip(" :.")
            elif label == "":
                m = _LABEL_PREFIX.match(text)
                if m:
                    label = text[:m.end()].strip(" :.")
                    text = text[m.end():]
            target = self.figure if self.figure is not None else self.table
            if target is not None:
                target.label = label
                target.caption = text[:MAX_CAPTION_CHARS]
            self._stop_collector(self.caption)
            self.caption, self.label = None, None
        elif role == "panel":
            self.panel = ""
        elif role == "figure":
            if self.figure is not None:
                self.p.figures.append(self.figure)
            self.figure = None
        elif role == "table":
            if self.table is not None and self.table.anchor:
                self.p.tables.append(self.table)
            self.table = None
        elif role == "para":
            self.para.text = self.para_text.text()
            if self.para.text:
                self.p.paragraphs.append(self.para)
                if len(self.p.paragraphs) >= MAX_PARAGRAPHS:
                    self.full = True
            self._stop_collector(self.para_text)
            self.para, self.para_text = None, None

    def _image(self, a: dict, classes: list[str]) -> None:
        if self.figure is None or self.table is not None:
            return
        src = (a.get("src") or "").strip()
        if src.startswith("/") or src.startswith("http:") or src.startswith("https:") or src.startswith("data:"):
            return
        missing = (not src) or "ltx_missing_image" in classes or ".." in src or not _IMAGE_PATH.match(src)
        self.figure.images.append(Image(src="" if missing else src, width=_int(a.get("width")),
                                        height=_int(a.get("height")), panel=self.panel, missing=missing,
                                        kind="svg" if src.lower().endswith(".svg") else "img"))


def parse(html: str | bytes, arxiv_id: str) -> ParsedPaper:
    """Everything the redesign keeps from one arXiv HTML page. Never raises."""
    paper = ParsedPaper(arxiv_id=arxiv_id)
    if isinstance(html, bytes):
        if len(html) > MAX_HTML_BYTES:
            paper.status, paper.error = "error", "page over 8 MB"
            return paper
        html = html.decode("utf-8", errors="replace")
    elif len(html.encode("utf-8", errors="replace")) > MAX_HTML_BYTES:
        paper.status, paper.error = "error", "page over 8 MB"
        return paper
    parser = _Parser(paper)
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:                      # html.parser is lenient; this is a backstop
        paper.status, paper.error = "error", f"{type(e).__name__}: {str(e)[:200]}"
        return paper
    if parser.figure is not None:               # a page cut off inside a figure
        paper.figures.append(parser.figure)
    paper.sections = [s for s in paper.sections if s.title or s.number]
    if not (paper.title or paper.abstract or paper.paragraphs):
        paper.status, paper.error = "error", "no paper content found"
    return paper


# ── inline SVG figures ───────────────────────────────────────────────────────

MAX_SVG_ELEMENTS = 100_000
_HTML_VOID = _VOID | {"track", "param", "keygen"}


class _InlineSvgs(HTMLParser):
    """The first <svg> inside each outermost figure.ltx_figure, as an element tree.

    Built from html.parser events rather than cut out as text: the page is
    HTML, so a label may hold <br> with no end tag, which XML would refuse.
    Names arrive lowercased; app/svg_clean.py restores SVG's camelCase.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found: dict[str, ET.Element | None] = {}
        self.stack: list[str] = []
        self.figure_at: int | None = None          # stack depth of the open outermost figure
        self.anchor = ""
        self.builder: ET.TreeBuilder | None = None
        self.open: list[str] = []                  # open tags inside the drawing
        self.count = 0

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if self.builder is not None:
            self._start(tag, a, void=tag in _HTML_VOID)
        elif tag == "figure" and self.figure_at is None and "ltx_figure" in (a.get("class") or "").split():
            self.figure_at, self.anchor = len(self.stack), a.get("id") or ""
        elif tag == "svg" and self.figure_at is not None and self.anchor and self.anchor not in self.found:
            self.builder, self.open, self.count = ET.TreeBuilder(), [], 0
            self._start(tag, a, void=False)
        if tag not in _HTML_VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if self.builder is not None:
            self._start(tag, {k: v or "" for k, v in attrs}, void=True)

    def handle_data(self, data):
        if self.builder is not None:
            self.builder.data(data)

    def handle_endtag(self, tag):
        if self.builder is not None and tag in self.open:
            while self.open:
                name = self.open.pop()
                self.builder.end(name)
                if name == tag:
                    break
            if not self.open:
                self.found[self.anchor] = self.builder.close()
                self.builder = None
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass
            if self.figure_at is not None and len(self.stack) <= self.figure_at:
                if self.builder is not None:           # the figure closed before its drawing did
                    self.found[self.anchor], self.builder = None, None
                self.figure_at, self.anchor = None, ""

    def _start(self, tag: str, attrs: dict, *, void: bool) -> None:
        self.count += 1
        if self.count > MAX_SVG_ELEMENTS:
            self.found[self.anchor], self.builder = None, None
            return
        self.builder.start(tag, attrs)
        if void:
            self.builder.end(tag)
        else:
            self.open.append(tag)


def inline_svgs(html: str | bytes) -> dict[str, ET.Element]:
    """{figure anchor: its first inline <svg>} for every figure drawn in the page."""
    if isinstance(html, bytes):
        html = html[:MAX_HTML_BYTES].decode("utf-8", errors="replace")
    parser = _InlineSvgs()
    try:
        parser.feed(html)
        parser.close()
    except Exception:                               # html.parser is lenient; this is a backstop
        pass
    return {k: v for k, v in parser.found.items() if v is not None}
