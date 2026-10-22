"""
app/arxiv_html.py: what the redesign keeps from an arXiv HTML page.

Golden tests run the parser over six real pages (CC BY 4.0, trimmed; see
tests/fixtures/arxiv_html/README.md) and compare a summary with the checked-in
expected JSON. The targeted tests below pin each markup rule on its own, so a
golden diff can be read against them.
"""
import json
import random
from pathlib import Path

import pytest

from app import arxiv_html

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"
IDS = ["2609.20888", "2609.27176", "2609.29861", "2609.30356", "2609.30670", "2609.31507"]
VERSIONS = {"2609.20888": "v2", "2609.27176": "v2", "2609.29861": "v2",
            "2609.30356": "v1", "2609.30670": "v1", "2609.31507": "v1"}


def load(aid):
    return arxiv_html.parse((FIXTURES / f"{aid}.html").read_bytes(), aid)


def summary(p):
    return {
        "status": p.status, "version": p.version, "licence_text": p.licence_text, "title": p.title,
        "abstract_start": p.abstract[:160],
        "sections": [[s.anchor, s.number, s.title, s.level, s.kind] for s in p.sections],
        "paragraphs": [[x.anchor, x.section, x.text[:80], x.refs] for x in p.paragraphs],
        "figures": [[f.anchor, f.label, f.caption[:80],
                     [[i.src, i.width, i.height, i.panel, i.missing, i.kind] for i in f.images]] for f in p.figures],
        "tables": [[t.anchor, t.label, t.caption[:80]] for t in p.tables],
    }


@pytest.mark.parametrize("aid", IDS)
def test_golden(aid, update_golden):
    got = summary(load(aid))
    path = FIXTURES / f"{aid}.expected.json"
    if update_golden or not path.exists():
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n")
    assert got == json.loads(path.read_text())


@pytest.mark.parametrize("aid", IDS)
def test_licence_and_version(aid):
    p = load(aid)
    assert p.status == "ok"
    assert p.licence_text == "CC BY 4.0"
    assert p.version == VERSIONS[aid]


def test_satnav_sections_carry_their_numbers_and_titles():
    sections = {s.anchor: s for s in load("2609.31507").sections}
    assert (sections["S1"].number, sections["S1"].title, sections["S1"].level) == ("1", "Introduction", 2)
    assert (sections["S3.SS2"].number, sections["S3.SS2"].title) == ("3.2", "Geospatial Data Resources")
    assert sections["S3.SS2"].kind == "subsection"


def test_paragraphs_know_their_section_and_the_figures_they_cite():
    paras = {x.anchor: x for x in load("2609.27176").paragraphs}
    assert paras["S1.p4"].section == "S1"
    assert "S1.F1" in paras["S1.p4"].refs
    assert not any(r.startswith("bib") for x in paras.values() for r in x.refs)


def test_captions_lose_their_label_prefix():
    figures = {f.anchor: f for f in load("2609.31507").figures}
    assert figures["S1.F1"].label == "Figure 1"
    assert figures["S1.F1"].caption.startswith("Overview of SatNav")


def test_sub_panels_belong_to_their_figure():
    figures = {f.anchor: f for f in load("2609.31507").figures}
    panels = {i.panel for i in figures["S3.F3"].images}
    assert {"S3.F3.sf2", "S3.F3.sf3"} <= panels
    assert "S3.F3.sf1" not in {f.anchor for f in load("2609.31507").figures}


def test_missing_images_are_marked_not_dropped():
    figures = {f.anchor: f for f in load("2609.20888").figures}
    images = figures["A3.F5"].images
    assert images and all(i.missing and i.src == "" for i in images)
    assert figures["A3.F5"].first_image() is None


def test_image_paths_are_kept_relative_with_their_size():
    f = {f.anchor: f for f in load("2609.31507").figures}["S1.F1"]
    img = f.first_image()
    assert (img.src, img.width, img.height) == ("2609.31507v1/overview-final.png", 476, 235)


def test_tables_are_citation_targets_not_figures():
    p = load("2609.27176")
    assert {"S4.T1", "S5.T2"} <= {t.anchor for t in p.tables}
    assert not any(f.anchor.split(".")[-1].startswith("T") for f in p.figures)


def test_the_contents_and_bibliography_are_skipped():
    p = load("2609.31507")
    text = " ".join(x.text for x in p.paragraphs)
    assert "Bibliography stub" not in text and "Contents stub" not in text


def test_each_formula_appears_once():
    html = ('<div class="ltx_para" id="S1.p1"><p class="ltx_p">Let <math alttext="x^{2}" class="ltx_Math">'
            '<semantics><msup><mi>x</mi><mn>2</mn></msup><annotation encoding="application/x-tex">x^{2}'
            '</annotation></semantics></math> be small.</p></div>')
    p = arxiv_html.parse(html, "2609.00001")
    assert p.paragraphs[0].text == "Let $x^{2}$ be small."


def test_real_formulas_are_not_doubled():
    for aid in IDS:
        for x in load(aid).paragraphs:
            for formula in x.text.split("$")[1::2]:
                if len(formula) > 6:
                    assert formula + formula not in x.text.replace("$", "").replace(" ", "")


def test_unsafe_image_paths_are_treated_as_missing():
    html = ('<figure id="S1.F1" class="ltx_figure"><img src="../../etc/passwd.png" class="ltx_graphics">'
            '<img src="https://evil.example/a.png" class="ltx_graphics">'
            '<img src="2609.00001v1/ok.png" class="ltx_graphics" width="10" height="5">'
            '<figcaption><span class="ltx_tag ltx_tag_figure">Figure 1: </span>Caption.</figcaption></figure>'
            '<div class="ltx_para" id="S1.p1">Text.</div>')
    f = arxiv_html.parse(html, "2609.00001").figures[0]
    assert [(i.src, i.missing) for i in f.images] == [("", True), ("2609.00001v1/ok.png", False)]
    assert f.caption == "Caption."


def test_paragraph_and_figure_caps():
    paras = "".join(f'<div class="ltx_para" id="S1.p{i}">Paragraph {i}.</div>' for i in range(700))
    figs = "".join(f'<figure id="S1.F{i}" class="ltx_figure"><figcaption>Figure {i}: c</figcaption></figure>'
                   for i in range(100))
    p = arxiv_html.parse(paras + figs, "2609.00001")
    assert len(p.paragraphs) == arxiv_html.MAX_PARAGRAPHS
    assert len(p.figures) == arxiv_html.MAX_FIGURES


def test_a_page_over_8_mb_is_refused():
    p = arxiv_html.parse(b"<p>" + b"x" * (arxiv_html.MAX_HTML_BYTES + 1), "2609.00001")
    assert p.status == "error" and "8 MB" in p.error


def test_garbage_is_an_error_result_not_an_exception():
    assert arxiv_html.parse("<<<>>> not html at all", "2609.00001").status == "error"
    assert arxiv_html.parse("", "2609.00001").status == "error"


def test_bytes_that_are_not_utf8_still_parse():
    html = (FIXTURES / "2609.29861.html").read_bytes().replace(b"Introduction", b"Introduc\xff\xfetion", 1)
    assert arxiv_html.parse(html, "2609.29861").status == "ok"


@pytest.mark.parametrize("aid", IDS)
def test_cutting_a_page_anywhere_never_raises(aid):
    html = (FIXTURES / f"{aid}.html").read_text()
    rng = random.Random(aid)
    for cut in sorted(rng.sample(range(1, len(html)), 50)):
        arxiv_html.parse(html[:cut], aid)


@pytest.mark.parametrize("aid", IDS)
def test_every_fixture_has_a_title_an_abstract_and_paragraphs(aid):
    """A trimming mistake once dropped every abstract and the golden files
    recorded the gap faithfully; this catches it directly."""
    p = load(aid)
    assert p.title and len(p.abstract) > 200 and len(p.paragraphs) >= 5


# ── vector figures ───────────────────────────────────────────────────────────

FIG = """<html><body><div id="watermark-tr">arXiv:2508.12461v3 [cs.CL] 1 Sep 2025</div>
<h1 class="ltx_title ltx_title_document">T</h1>
<figure id="S1.F1" class="ltx_figure">{body}<figcaption class="ltx_caption">
<span class="ltx_tag ltx_tag_figure">Fig. 1: </span>A chart.</figcaption></figure>
<div class="ltx_para" id="S1.p1"><p class="ltx_p">Text.</p></div></body></html>"""


def fig(body):
    return arxiv_html.parse(FIG.format(body=body), "2508.12461").figures[0]


def test_an_svg_file_in_an_object_is_a_figure_image():
    f = fig('<object type="image/svg+xml" data="2508.12461v3/figure_6_radar_chart.svg" '
            'class="ltx_graphics" width="476" height="380"></object>')
    assert f.first_image() == arxiv_html.Image(src="2508.12461v3/figure_6_radar_chart.svg", width=476,
                                               height=380, kind="svg")


def test_an_svg_file_in_an_img_is_marked_svg():
    assert fig('<img src="2508.12461v3/x.svg" class="ltx_graphics" width="10" height="5">').first_image().kind == "svg"


def test_an_inline_drawing_is_a_figure_image_without_a_path():
    f = fig('<div class="ltx_flex_figure"><span class="ltx_inline-block"><svg id="S1.F1.pic1" '
            'class="ltx_picture" width="259.33" height="182.79" viewBox="0 0 259.33 182.79">'
            '<g><svg class="ltx_nestedsvg"><path d="M0 0"/></svg></g></svg></span></div>')
    assert f.first_image() == arxiv_html.Image(src="", width=259, height=182, kind="svg-inline")
    assert len(f.images) == 1                         # the nested <svg> is part of the same drawing


def test_inline_svgs_lifts_the_first_drawing_of_each_figure():
    page = """<figure id="S1.F1" class="ltx_figure"><span><svg class="ltx_picture" viewBox="0 0 9 9">
    <foreignObject width="9" height="9"><span>label<br>two lines</span></foreignObject>
    <clipPath id="c"><rect width="1"/></clipPath></svg></span>
    <svg class="ltx_picture"><rect id="second"/></svg><figcaption>c</figcaption></figure>
    <p><svg class="icon"><path d="M1 1"/></svg></p>
    <figure id="S2.F2" class="ltx_figure"><img src="2508.12461v3/a.png" class="ltx_graphics"></figure>"""
    got = arxiv_html.inline_svgs(page)
    assert list(got) == ["S1.F1"]                     # icons outside figures and picture figures are not drawings
    svg = got["S1.F1"]
    assert svg.tag == "svg" and svg.find("foreignobject/span").text == "label"
    assert svg.find("foreignobject/span/br").tail == "two lines"
    assert svg.find(".//rect[@id='second']") is None


def test_inline_svgs_gives_up_on_a_drawing_with_too_many_elements(monkeypatch):
    monkeypatch.setattr(arxiv_html, "MAX_SVG_ELEMENTS", 5)
    page = '<figure id="S1.F1" class="ltx_figure"><svg>' + "<rect/>" * 10 + "</svg></figure>"
    assert arxiv_html.inline_svgs(page) == {}


def test_inline_svgs_never_raises_on_a_cut_page():
    page = '<figure id="S1.F1" class="ltx_figure"><svg><g><path d="M0'
    assert arxiv_html.inline_svgs(page) == {}

