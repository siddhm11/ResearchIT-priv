"""
app/svg_clean.py: vector figures cleaned down to drawing elements.

Each attack test feeds one known way of getting script, a link or an outside
load into an SVG and checks that nothing of it survives. The output is always
re-parsed as XML so a test can't pass on a string match alone.
"""
import xml.etree.ElementTree as ET

import pytest

from app import svg_clean
from app.svg_clean import MATHML_NS, SVG_NS, XHTML_NS

S = f'xmlns="{SVG_NS}" xmlns:xlink="http://www.w3.org/1999/xlink"'


def clean(body: str, root_attrs: str = 'viewBox="0 0 10 10"') -> str:
    out = svg_clean.clean(f"<svg {S} {root_attrs}>{body}</svg>".encode())
    ET.fromstring(out)                                    # always well-formed
    return out.decode()


def tags(svg: str) -> set[str]:
    return {el.tag.split("}")[-1] for el in ET.fromstring(svg).iter()}


# ── what a figure needs survives ─────────────────────────────────────────────

def test_drawing_elements_and_local_references_survive():
    out = clean('<defs><clipPath id="c"><rect width="5" height="5"/></clipPath>'
                '<linearGradient id="g"><stop offset="0"/></linearGradient></defs>'
                '<g clip-path="url(#c)"><path d="M0 0L5 5" fill="url(#g)" stroke-width="0.5"/></g>'
                '<use xlink:href="#c"/><text x="1" y="2">Accuracy <tspan>(%)</tspan></text>')
    assert {"clipPath", "linearGradient", "path", "use", "text", "tspan"} <= tags(out)
    assert 'clip-path="url(#c)"' in out and 'fill="url(#g)"' in out and 'href="#c"' in out
    assert "Accuracy" in out and "(%)" in out


def test_a_matplotlib_file_with_its_doctype_is_accepted():
    raw = ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
           '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
           f'<svg {S} width="460.8pt" height="345.6pt" viewBox="0 0 460.8 345.6" version="1.1">'
           '<metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/></metadata>'
           '<defs><style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style></defs>'
           '<g id="figure_1"><path d="M 0 345.6 L 460.8 345.6" style="fill: #ffffff"/></g></svg>')
    out = svg_clean.clean(raw.encode()).decode()
    assert "metadata" not in out and "stroke-linejoin" in out and 'viewBox="0 0 460.8 345.6"' in out


def test_inline_png_data_in_an_image_is_kept():
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert f'href="{png}"' in clean(f'<image width="1" height="1" xlink:href="{png}"/>')


def test_a_size_is_filled_in_from_the_viewbox_and_the_reverse():
    assert 'width="10"' in clean("<rect/>") and 'height="10"' in clean("<rect/>")
    assert 'viewBox="0 0 40 30"' in clean("<rect/>", 'width="40" height="30"')


def test_html_labels_and_maths_keep_their_namespaces_and_the_tex_source_goes():
    out = clean('<foreignObject width="9" height="9"><span class="lbl">x <b>axis</b></span>'
                '<math><semantics><mn>0</mn><annotation encoding="application/x-tex">0</annotation>'
                '</semantics></math></foreignObject>')
    root = ET.fromstring(out)
    assert root.find(f".//{{{XHTML_NS}}}span") is not None
    assert root.find(f".//{{{MATHML_NS}}}mn").text == "0"
    assert "annotation" not in out and "application/x-tex" not in out


# ── attacks ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("body", [
    "<script>alert(1)</script>",
    '<script xlink:href="https://evil.example/x.js"/>',
    "<g><script><![CDATA[alert(1)]]></script></g>",
    '<foreignObject><iframe src="https://evil.example"/><span>ok</span></foreignObject>',
    '<foreignObject><span><script>alert(1)</script>ok</span></foreignObject>',
])
def test_scripts_and_frames_are_removed(body):
    out = clean(body)
    assert "script" not in out.lower() and "iframe" not in out and "alert" not in out and "evil" not in out


@pytest.mark.parametrize("attr", ['onload="alert(1)"', 'ONCLICK="alert(1)"', 'onmouseover="alert(1)"',
                                  'onbegin="alert(1)"'])
def test_event_handlers_are_removed(attr):
    assert "alert" not in clean(f"<rect {attr}/>", f'viewBox="0 0 1 1" {attr}')


@pytest.mark.parametrize("body", [
    '<a xlink:href="javascript:alert(1)"><text>x</text></a>',
    '<a href="java&#10;script:alert(1)"><text>x</text></a>',
    '<use href="https://evil.example/sprite.svg#a"/>',
    '<image href="https://evil.example/track.png"/>',
    '<image href="data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="/>',
    '<foreignObject><a href="https://evil.example">x</a></foreignObject>',
])
def test_links_go_unless_they_point_inside_the_document(body):
    out = clean(body)
    assert "javascript" not in out and "evil" not in out and "svg+xml" not in out


def test_a_link_element_becomes_a_group_and_keeps_its_drawing():
    out = clean('<a xlink:href="https://evil.example"><rect width="2"/></a>')
    assert "rect" in tags(out) and "a" not in tags(out)


@pytest.mark.parametrize("body", [
    '<rect style="fill: url(https://evil.example/x)"/>',
    '<rect fill="url(\'https://evil.example/x\')"/>',
    '<rect style="background: u\\72 l(https://evil.example/x)"/>',
    '<rect style="width: expression(alert(1))"/>',
    '<style>@import url(https://evil.example/x.css);</style>',
    '<style>rect { fill: url(https://evil.example/x) }</style>',
    '<style>rect { behavior: url(x.htc) }</style>',
])
def test_outside_loads_and_old_css_tricks_are_removed(body):
    out = clean(body)
    assert "evil" not in out and "expression" not in out and "behavior" not in out and "@import" not in out


@pytest.mark.parametrize("body", [
    '<animate attributeName="href" to="javascript:alert(1)"/>',
    '<set attributeName="onload" to="alert(1)"/>',
    '<a><animate attributeName="href" values="javascript:alert(1)"/><text>x</text></a>',
])
def test_animation_that_could_rewrite_attributes_is_removed(body):
    out = clean(body)
    assert "animate" not in out and "set" not in tags(out) and "javascript" not in out


def test_blocked_attributes_go_whatever_the_element():
    out = clean('<rect src="https://evil.example" data="x" ping="https://evil.example"/>')
    assert "evil" not in out and 'data="x"' not in out


# ── refusals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    b'<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY lol "lol">]><svg xmlns="http://www.w3.org/2000/svg">&lol;</svg>',
    b'<!DOCTYPE svg [<!ELEMENT svg ANY>]><svg xmlns="http://www.w3.org/2000/svg"/>',
    b'<html><body>not a drawing</body></html>',
    b'<svg xmlns="http://www.w3.org/2000/svg"><rect></svg>',
    b"\x89PNG\r\n\x1a\n",
    b"",
])
def test_entities_dtds_other_documents_and_broken_xml_are_refused(raw):
    with pytest.raises(svg_clean.NotSVG):
        svg_clean.clean(raw)


def test_a_file_too_large_or_with_too_many_elements_is_refused(monkeypatch):
    monkeypatch.setattr(svg_clean, "MAX_BYTES", 100)
    with pytest.raises(svg_clean.NotSVG):
        svg_clean.clean(f"<svg {S}>{'<rect/>' * 50}</svg>".encode())
    monkeypatch.setattr(svg_clean, "MAX_BYTES", 10_000)
    monkeypatch.setattr(svg_clean, "MAX_ELEMENTS", 20)
    with pytest.raises(svg_clean.NotSVG):
        svg_clean.clean(f"<svg {S}>{'<rect/>' * 50}</svg>".encode())


# ── inline SVG from an HTML page (lowercased names, no namespaces) ──────────

def test_lowercased_inline_svg_gets_its_camel_case_and_namespaces_back():
    root = ET.Element("svg", {"viewbox": "0 0 20 10", "width": "20", "height": "10"})
    clip = ET.SubElement(ET.SubElement(root, "defs"), "clippath", {"id": "p", "clippathunits": "userSpaceOnUse"})
    ET.SubElement(clip, "rect", {"width": "5"})
    fo = ET.SubElement(root, "foreignobject", {"width": "9", "height": "9"})
    ET.SubElement(fo, "span").text = "label"
    out = svg_clean.clean_tree(root).decode()
    tree = ET.fromstring(out)
    assert tree.tag == f"{{{SVG_NS}}}svg" and tree.get("viewBox") == "0 0 20 10"
    assert tree.find(f".//{{{SVG_NS}}}clipPath").get("clipPathUnits") == "userSpaceOnUse"
    assert tree.find(f".//{{{SVG_NS}}}foreignObject/{{{XHTML_NS}}}span").text == "label"
