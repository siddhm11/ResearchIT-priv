"""Write app/templates/partials/area_art.html: one line drawing per research area.

    python scripts/build_area_art.py

Edit a drawing here and re-run; the template is generated. The plotted ones
(elliptic curve, helix, flow round a cylinder, spectrogram, galaxy, filtered
signal) are computed so they are what they claim to be.

viewBox 0 0 160 96. Classes (styled in styles.css, section 15):
  s   ink stroke            h   annotation stroke (--note)
  p   paper-filled shape    hf  annotation shape with a pale fill
  d   ink dot               dh  annotation dot
  t   small mono label      th  annotation label   tk  strong label
  tb  larger label          tl  label on an annotation fill
  draw  redrawn on hover (needs pathLength="1")   faint   lower contrast
"""
import math
import random
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "app/templates/partials/area_art.html"


def f(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def poly(points):
    return "M" + " L".join(f"{f(x)} {f(y)}" for x, y in points)


def smooth(points):
    """Catmull-Rom through points, as cubic beziers."""
    d = [f"M{f(points[0][0])} {f(points[0][1])}"]
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{f(c1[0])} {f(c1[1])} {f(c2[0])} {f(c2[1])} {f(p2[0])} {f(p2[1])}")
    return " ".join(d)


def tri(cx, cy, dx, dy, size=3.2):
    """A small filled arrowhead centred on (cx, cy), pointing along (dx, dy)."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    tip = (cx + ux * size, cy + uy * size)
    b1 = (cx - ux * size + px * size * .8, cy - uy * size + py * size * .8)
    b2 = (cx - ux * size - px * size * .8, cy - uy * size - py * size * .8)
    return f'<path class="dk" d="M{f(tip[0])} {f(tip[1])} L{f(b1[0])} {f(b1[1])} L{f(b2[0])} {f(b2[1])}Z"/>'


art = {}

# ── Natural Language Processing: next-word prediction with attention ─────────
art["nlp"] = """
<text class="t faint" x="8" y="17">next word?</text>
<g class="s"><rect x="6" y="56" width="26" height="17" rx="4"/><rect x="35" y="56" width="26" height="17" rx="4"/><rect x="64" y="56" width="26" height="17" rx="4"/><rect x="93" y="56" width="22" height="17" rx="4"/></g>
<rect class="h dash" x="119" y="56" width="35" height="17" rx="4"/>
<g class="t" text-anchor="middle"><text x="19" y="67.5">The</text><text x="48" y="67.5">cat</text><text x="77" y="67.5">sat</text><text x="104" y="67.5">on</text></g>
<text class="th" x="136.5" y="67.5" text-anchor="middle">mat</text>
<path class="h draw" pathLength="1" d="M136 54 Q92 8 48 54"/>
<path class="s draw faint" pathLength="1" d="M136 54 Q106 26 77 54"/>
<path class="s draw faint" pathLength="1" d="M136 54 Q120 40 104 54"/>
<text class="th" x="136.5" y="86" text-anchor="middle">p = .81</text>
"""

# ── Computer Vision: a detector's boxes over a cat and a ball ────────────────
art["cv"] = """
<rect class="s" x="10" y="8" width="140" height="80" rx="4"/>
<path class="s faint" d="M10 77 H150"/>
<g class="s"><ellipse cx="66" cy="65" rx="19" ry="11"/><circle class="p" cx="46" cy="50" r="9.5"/><path d="M39.6 43.4 L39 34 L45.6 40.6 M47.4 40.6 L53 34 L53.8 43.6"/><path d="M84.5 61 Q98 55 93 41"/><path d="M55 75 v3 M77 75 v3"/></g>
<circle class="d" cx="43" cy="49.5" r="1.1"/><circle class="d" cx="49.5" cy="49.5" r="1.1"/>
<path class="h draw" pathLength="1" d="M32 30 H98 V79 H32Z"/>
<rect class="hfill" x="32" y="21" width="41" height="9" rx="1"/>
<text class="tl" x="35" y="27.8">cat 0.97</text>
<circle class="s" cx="122" cy="70" r="7"/>
<path class="s faint" d="M117 66 Q122 70 127 66"/>
<rect class="h dash" x="112" y="60" width="20" height="20" rx="1"/>
<text class="th" x="111" y="56">ball .88</text>
"""

# ── Machine Learning: points, a fitted line, a falling loss ─────────────────
pts = [(22, 74), (28, 75), (34, 66), (42, 67), (48, 57), (56, 59), (62, 48), (70, 50),
       (78, 39), (86, 41), (94, 30), (100, 33)]
dots = "".join(f'<circle cx="{x}" cy="{y}" r="2"/>' for x, y in pts)
art["ml"] = f"""
<path class="s" d="M14 10 V84 H108"/>
<g class="d">{dots}</g>
<path class="h draw" pathLength="1" d="M16 80 L104 26"/>
<text class="th" x="20" y="18">y = wx + b</text>
<rect class="p" x="116" y="12" width="36" height="32" rx="3"/>
<path class="s draw" pathLength="1" d="M120 16 C124 34 132 38 148 39"/>
<text class="t" x="116" y="56">loss</text>
<path class="s" d="M140 50 V58 M137 55 L140 58 L143 55"/>
"""

# ── Artificial Intelligence: a question answered by reasoning step by step ──
art["ai"] = """
<rect class="p" x="80" y="8" width="72" height="17" rx="8.5"/>
<text class="tk" x="116" y="19.5" text-anchor="middle">is 97 prime?</text>
<path class="dh" d="M16 35 Q17 42.5 24 43.5 Q17 44.5 16 52 Q15 44.5 8 43.5 Q15 42.5 16 35Z"/>
<rect class="p" x="28" y="32" width="100" height="56" rx="8"/>
<text class="t seq" style="--i:0" x="36" y="45">√97 &lt; 10</text>
<text class="t seq" style="--i:1" x="36" y="56">try 2, 3, 5, 7</text>
<text class="t seq" style="--i:2" x="36" y="67">none divide 97</text>
<text class="th seq" style="--i:3" x="36" y="80">→ yes, prime ✓</text>
"""

# ── Robotics: a two-link arm carrying a block toward its target ─────────────
art["robotics"] = """
<path class="s faint" d="M8 86 H152"/>
<path class="s lk" d="M35 70 L68 36 L114 46"/>
<rect class="p" x="18" y="76" width="34" height="10" rx="2"/>
<path class="s" d="M35 70 V76"/>
<circle class="p" cx="35" cy="70" r="4.5"/><circle class="p" cx="68" cy="36" r="4"/>
<path class="s" d="M114 46 V52 M106 52 H122 M106 52 V60 M122 52 V60"/>
<rect class="hf" x="108" y="57" width="12" height="12" rx="1"/>
<path class="s faint" d="M47 70 A12 12 0 0 0 43.5 61.5"/>
<text class="t" x="49" y="62">θ</text>
<path class="h dash" d="M122 66 Q142 62 142 80"/>
<path class="h" d="M138 82 L146 78 M138 78 L146 82"/>
"""

# ── High Energy Physics: e⁻e⁺ → γ → μ⁻μ⁺, as a Feynman diagram ──────────────
wave = "M48 48 q4 -7 8 0" + " t8 0" * 7
art["hep"] = f"""
<path class="s draw" pathLength="1" d="M14 12 L48 48"/>
<path class="s draw" pathLength="1" d="M14 84 L48 48"/>
<path class="h" d="{wave}"/>
<path class="s draw" pathLength="1" d="M112 48 L146 12"/>
<path class="s draw" pathLength="1" d="M112 48 L146 84"/>
{tri(31, 30, 1, 1)}{tri(31, 66, -1, 1)}{tri(129, 30, 1, -1)}{tri(129, 66, -1, -1)}
<circle class="d" cx="48" cy="48" r="2.2"/><circle class="d" cx="112" cy="48" r="2.2"/>
<g class="t"><text x="8" y="26">e⁻</text><text x="8" y="76">e⁺</text><text x="140" y="32">μ⁻</text><text x="140" y="70">μ⁺</text></g>
<text class="th" x="80" y="36" text-anchor="middle">γ</text>
"""

# ── Astrophysics: a two-armed spiral galaxy ──────────────────────────────────
random.seed(7)
cx, cy = 78, 47
arm_paths, arm_dots = [], []
for a in (0, math.pi):
    line = []
    th = 0.0
    while th <= 3.25 * math.pi:
        r = 5.2 * math.exp(0.22 * th)
        x = cx + r * math.cos(th + a)
        y = cy + 0.52 * r * math.sin(th + a)
        line.append((x, y))
        if th > 0.6:
            for _ in range(1):
                jx, jy = random.gauss(0, 1.6), random.gauss(0, 1.0)
                rad = max(0.7, 1.7 - th / 8)
                arm_dots.append(f'<circle cx="{f(x + jx)}" cy="{f(y + jy)}" r="{f(rad)}"/>')
        th += 0.3
    arm_paths.append(smooth(line))
stars = [(14, 14), (150, 16), (12, 70), (22, 88), (140, 76), (128, 8)]
sparkle = "".join(
    f'<path class="d" d="M{x} {y - 3} Q{x + .5} {y - .5} {x + 3} {y} Q{x + .5} {y + .5} {x} {y + 3} Q{x - .5} {y + .5} {x - 3} {y} Q{x - .5} {y - .5} {x} {y - 3}Z"/>'
    for x, y in stars[:3]) + "".join(f'<circle class="d" cx="{x}" cy="{y}" r=".9"/>' for x, y in stars[3:])
art["astro"] = f"""
<path class="s" d="{arm_paths[0]}"/><path class="s" d="{arm_paths[1]}"/>
<g class="d">{''.join(arm_dots)}</g>
<ellipse class="dh soft" cx="{cx}" cy="{cy}" rx="14" ry="7.5"/>
<ellipse class="dh" cx="{cx}" cy="{cy}" rx="6" ry="3.6"/>
{sparkle}
<text class="th" x="148" y="92" text-anchor="end">z = 1.2</text>
"""

# ── Quantum Computing: a Bell-pair circuit ──────────────────────────────────
art["quant_ph"] = """
<path class="s" d="M30 34 H110 M30 66 H110"/>
<g class="t"><text x="8" y="37">|0⟩</text><text x="8" y="69">|0⟩</text></g>
<rect class="p" x="42" y="25" width="18" height="18" rx="2"/>
<text class="tk" x="51" y="37.4" text-anchor="middle">H</text>
<path class="h" d="M82 34 V73.5"/>
<circle class="dh" cx="82" cy="34" r="3.2"/>
<circle class="hf" cx="82" cy="66" r="7.5"/>
<path class="h" d="M74.5 66 H89.5"/>
<rect class="p" x="110" y="25" width="24" height="18" rx="2"/><path class="s" d="M115 39 A7 7 0 0 1 129 39 M122 39 L127 30"/>
<rect class="p" x="110" y="57" width="24" height="18" rx="2"/><path class="s" d="M115 71 A7 7 0 0 1 129 71 M122 71 L127 62"/>
<text class="th" x="146" y="92" text-anchor="end">|00⟩ + |11⟩</text>
"""

# ── Mathematics: the chord rule on an elliptic curve ─────────────────────────
def ex(x):
    return 18 + (x + 1.45) * 26


def ey(y):
    return 48 - y * 18.5


root = -1.324718
upper, lower = [], []
xs = [root + (1.75 - root) * (i / 18) ** 1.6 for i in range(19)]
for x in xs:
    y = math.sqrt(max(x ** 3 - x + 1, 0))
    upper.append((ex(x), ey(y)))
    lower.append((ex(x), ey(-y)))
curve = smooth(list(reversed(upper)) + lower[1:])
P, Q, R = (-1.2, 0.687), (1.2, -1.236), (0.642, -0.788)
S = (0.642, 0.788)
art["math"] = f"""
<path class="s faint" d="M10 48 H112 M{f(ex(0))} 8 V88"/>
<path class="s draw" pathLength="1" d="{curve}"/>
<path class="h" d="M{f(ex(P[0] - .12))} {f(ey(P[1] + .1))} L{f(ex(Q[0] + .1))} {f(ey(Q[1] - .08))}"/>
<path class="h dash" d="M{f(ex(R[0]))} {f(ey(R[1]))} V{f(ey(S[1]))}"/>
<g class="d"><circle cx="{f(ex(P[0]))}" cy="{f(ey(P[1]))}" r="2.2"/><circle cx="{f(ex(Q[0]))}" cy="{f(ey(Q[1]))}" r="2.2"/><circle cx="{f(ex(R[0]))}" cy="{f(ey(R[1]))}" r="2.2"/></g>
<circle class="dh" cx="{f(ex(S[0]))}" cy="{f(ey(S[1]))}" r="2.6"/>
<g class="t"><text x="{f(ex(P[0]) - 4)}" y="{f(ey(P[1]) - 5)}">P</text><text x="{f(ex(Q[0]) + 4)}" y="{f(ey(Q[1]) + 3)}">Q</text></g>
<text class="th" x="{f(ex(S[0]) + 4)}" y="{f(ey(S[1]) - 4)}">P+Q</text>
<text class="t" x="152" y="20" text-anchor="end">y²=x³−x+1</text>
"""

# ── Computational Biology: a DNA double helix ────────────────────────────────
a_pts, b_pts, rungs = [], [], []
for i in range(0, 133):
    x = 14 + i
    phase = 2 * math.pi * i / 66
    a_pts.append((x, 48 + 22 * math.sin(phase)))
    b_pts.append((x, 48 - 22 * math.sin(phase)))
for k, i in enumerate(range(3, 133, 6)):
    x = 14 + i
    phase = 2 * math.pi * i / 66
    ya, yb = 48 + 22 * math.sin(phase), 48 - 22 * math.sin(phase)
    if abs(ya - yb) > 5:
        cls = "h" if k % 3 == 0 else "s faint"
        rungs.append(f'<path class="{cls}" d="M{x} {f(min(ya, yb) + 2)} V{f(max(ya, yb) - 2)}"/>')
art["bio"] = f"""
{''.join(rungs)}
<path class="s lk2" d="{poly(a_pts[::3] + [a_pts[-1]])}"/>
<path class="s lk2" d="{poly(b_pts[::3] + [b_pts[-1]])}"/>
<g class="th"><text x="16" y="90">A–T</text><text x="132" y="90">G–C</text></g>
"""

# ── Neuroscience: a neuron and the spikes it fires ──────────────────────────
spike = "l2 -14 l2.5 20 l2 -6"
art["neuro"] = f"""
<path class="s" d="M31 31 L18 18 M24 24 L13 26 M31 41 L16 53 M22 47.5 L12 44 M36 28.5 L32 11 M34 20 L26 13 M41.5 28.6 L47 15"/>
<path class="s" d="M46 36 H118"/>
<g class="p"><rect x="56" y="32.5" width="11" height="7" rx="3.5"/><rect x="72" y="32.5" width="11" height="7" rx="3.5"/><rect x="88" y="32.5" width="11" height="7" rx="3.5"/><rect x="104" y="32.5" width="11" height="7" rx="3.5"/></g>
<circle class="p" cx="38" cy="36" r="8"/>
<path class="s" d="M118 36 L132 26 M118 36 L134 37 M118 36 L130 48"/>
<g class="dh"><circle cx="134" cy="24.5" r="2"/><circle cx="136.5" cy="37" r="2"/><circle cx="132" cy="49.5" r="2"/></g>
<path class="h draw" pathLength="1" d="M12 80 H34 {spike} H66 {spike} H80 {spike} H118 {spike} H150"/>
"""

# ── Economics & Game Theory: the prisoner's dilemma ─────────────────────────
art["econ"] = """
<rect class="p" x="46" y="20" width="100" height="62" rx="2"/>
<path class="s" d="M96 20 V82 M46 51 H146"/>
<g class="t" text-anchor="middle"><text x="71" y="15">C</text><text x="121" y="15">D</text><text x="36" y="39">C</text><text x="36" y="70">D</text></g>
<g class="tb" text-anchor="middle"><text x="71" y="39.5">3, 3</text><text x="121" y="39.5">0, 5</text><text x="71" y="70.5">5, 0</text></g>
<path class="h draw" pathLength="1" d="M99 54 H143 V79 H99Z"/>
<text class="tb thc" x="121" y="70.5" text-anchor="middle">1, 1</text>
<text class="th" x="146" y="93" text-anchor="end">Nash</text>
"""

# ── Cryptography & Security: plaintext in, ciphertext out ────────────────────
art["crypto"] = """
<text class="tb" x="10" y="52">hello</text>
<path class="s" d="M44 48 H64 M90 48 H104 M100 45 L104 48 L100 51"/>
<path class="s" d="M71 43 V37 a6 6 0 0 1 12 0 V43"/>
<rect class="p" x="67" y="43" width="20" height="15" rx="2.5"/>
<circle class="d" cx="77" cy="49" r="1.6"/><path class="s" d="M77 50 V54"/>
<g class="th"><text x="109" y="45">9f 2c a7</text><text x="109" y="56">1e 0b d4</text></g>
<circle class="s" cx="66" cy="75" r="4.5"/><path class="s" d="M70.5 75 H90 M85 75 V79 M89 75 V78"/>
<text class="t" x="95" y="78">key</text>
<text class="t faint" x="10" y="20">e3 4a 9c 01 7f b2 58 c6 0d</text>
"""

# ── Systems & Networking: servers passing messages ──────────────────────────
nodes = [(26, 24), (80, 14), (134, 24), (44, 72), (116, 72)]
edges = [(0, 1), (1, 2), (0, 3), (2, 4), (3, 4), (1, 3), (1, 4)]
edge_d = " ".join(f"M{nodes[a][0]} {nodes[a][1]} L{nodes[b][0]} {nodes[b][1]}" for a, b in edges)
servers = "".join(
    f'<rect class="p" x="{x - 11}" y="{y - 7.5}" width="22" height="15" rx="2.5"/>'
    f'<path class="s faint" d="M{x - 7} {y - 2} H{x + 3} M{x - 7} {y + 2.5} H{x + 3}"/>'
    f'<circle class="dh" cx="{x + 6.5}" cy="{y}" r="1.3"/>' for x, y in nodes)


def along(a, b, t):
    return nodes[a][0] + (nodes[b][0] - nodes[a][0]) * t, nodes[a][1] + (nodes[b][1] - nodes[a][1]) * t


packets = "".join(tri(*along(a, b, t), nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1], 3)
                  .replace('class="dk"', 'class="dh"') for a, b, t in ((0, 1, .5), (1, 4, .55), (3, 4, .5)))
art["systems"] = f"""
<path class="s faint" d="{edge_d}"/>
{packets}
{servers}
<text class="th" x="80" y="92" text-anchor="middle">12 ms</text>
"""

# ── Human-Computer Interaction: a click on an interface ─────────────────────
art["hci"] = """
<rect class="p" x="14" y="10" width="132" height="76" rx="5"/>
<path class="s" d="M14 22 H146"/>
<g class="d"><circle cx="21" cy="16" r="1.5"/><circle cx="27" cy="16" r="1.5"/><circle cx="33" cy="16" r="1.5"/></g>
<path class="s lk2" d="M24 33 H76"/>
<path class="s faint" d="M24 42 H112 M24 49 H98"/>
<text class="tb" x="120" y="47">Aa</text>
<rect class="hf" x="24" y="59" width="44" height="15" rx="7.5"/>
<text class="th" x="42" y="69.4" text-anchor="middle">Sign up</text>
<rect class="s" x="74" y="59" width="34" height="15" rx="7.5"/>
<text class="t" x="91" y="69.4" text-anchor="middle">Later</text>
<circle class="h faint" cx="64" cy="69" r="5"/>
<circle class="h dash faint" cx="64" cy="69" r="9.5"/>
<path class="p" d="M64 69 l0 14 l3.6 -3.4 l2.6 6 l2.6 -1.1 l-2.6 -6 l5 -.4z"/>
"""

# ── Speech & Audio: a spoken word and its spectrogram ────────────────────────
wave_pts = []
for i in range(0, 133, 2):
    x = 14 + i
    t = i / 132
    env = math.exp(-((t - .27) / .11) ** 2) + .8 * math.exp(-((t - .68) / .13) ** 2) + .04
    y = 30 - 21 * env * math.sin(2 * math.pi * 11 * t) * math.cos(2 * math.pi * 3.1 * t + .4)
    wave_pts.append((x, y))
cells = []
for col in range(22):
    t = (col + .5) / 22
    env = math.exp(-((t - .27) / .12) ** 2) + .8 * math.exp(-((t - .68) / .14) ** 2)
    for row in range(5):
        formant = (1.0, .75, .5, .62, .3)[row] if t < .47 else (.85, .55, .7, .35, .25)[row]
        v = min(.92, .06 + env * formant)
        if v > .1:
            cells.append(f'<rect x="{f(14 + col * 6)}" y="{f(82 - row * 5.2)}" width="5.2" height="4.4" rx=".6" style="opacity:{v:.2f}"/>')
art["audio"] = f"""
<path class="s draw" pathLength="1" d="{smooth(wave_pts)}"/>
<path class="s faint" d="M12 30 H148"/>
<g class="dh">{''.join(cells)}</g>
<g class="th"><text x="42" y="57" text-anchor="middle">hel</text><text x="104" y="57" text-anchor="middle">lo</text></g>
"""

# ── Physics — General: flow around a cylinder, with vortices shed behind ────
cxp, cyp, Rc = 56, 48, 11


def stream_y(x, psi):
    lo = math.sqrt(max(Rc * Rc - x * x, 0)) + 1e-6
    hi = 80.0

    def g(y):
        return y * (1 - Rc * Rc / (x * x + y * y)) - psi
    for _ in range(60):
        mid = (lo + hi) / 2
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


lines = []
for psi in (2.5, 8, 15, 23, 32):
    for sign in (1, -1):
        p = []
        for i in range(0, 141, 10):
            x = 10 + i - cxp
            y = stream_y(x, psi)
            p.append((cxp + x, cyp - sign * y))
        lines.append(f'<path class="s faint" d="{smooth(p)}"/>')


def spiral(x0, y0, turns, r1, sign):
    p = []
    n = 26
    for i in range(n + 1):
        th = turns * 2 * math.pi * i / n
        r = r1 * (1 - i / (n + 6))
        p.append((x0 + r * math.cos(th) * sign, y0 + r * math.sin(th)))
    return f'<path class="h draw" pathLength="1" d="{smooth(p)}"/>'


art["pde"] = f"""
{''.join(lines)}
<circle class="p" cx="{cxp}" cy="{cyp}" r="{Rc}"/>
{spiral(88, 42, 1.4, 6.5, 1)}{spiral(108, 55, 1.4, 6.5, -1)}{spiral(130, 42, 1.4, 6.5, 1)}
"""

# ── Condensed Matter: an antiferromagnet, one spin flipped ──────────────────
atoms, links, spins = [], [], []
for j in range(3):
    for i in range(6):
        x, y = 22 + 23 * i, 22 + 26 * j
        if i < 5:
            links.append(f"M{x} {y} H{x + 23}")
        if j < 2:
            links.append(f"M{x} {y} V{y + 26}")
        up = (i + j) % 2 == 0
        flipped = (i, j) == (3, 1)
        if flipped:
            up = not up
        cls = "hf" if flipped else "p"
        atoms.append(f'<circle class="{cls}" cx="{x}" cy="{y}" r="7"/>')
        y0, y1 = (y + 4.2, y - 4.2) if up else (y - 4.2, y + 4.2)
        head = f"M{x - 2.4} {f(y1 + (2.6 if up else -2.6))} L{x} {f(y1)} L{x + 2.4} {f(y1 + (2.6 if up else -2.6))}"
        spins.append(f'<path class="{"h" if flipped else "s"}" d="M{x} {f(y0)} L{x} {f(y1)} {head}"/>')
art["cond_mat"] = f"""
<path class="s faint" d="{' '.join(links)}"/>
{''.join(atoms)}{''.join(spins)}
"""

# ── Software Engineering: a function and its passing test ───────────────────
art["se"] = """
<rect class="p" x="12" y="8" width="136" height="80" rx="5"/>
<path class="s faint" d="M28 8 V88"/>
<g class="t faint" text-anchor="middle"><text x="20" y="24">1</text><text x="20" y="36">2</text><text x="20" y="48">3</text><text x="20" y="60">4</text></g>
<text class="t" x="34" y="24"><tspan class="tk">def</tspan> add(a, b):</text>
<text class="t" x="34" y="36">  <tspan class="tk">return</tspan> a + b</text>
<text class="t" x="34" y="60"><tspan class="tk">assert</tspan> add(2, 2) == 4</text>
<rect class="hf" x="34" y="68" width="58" height="13" rx="6.5"/>
<text class="th" x="40" y="77.3">✓ 1 passed</text>
<rect class="dh blink" x="104" y="30" width="1.6" height="8"/>
"""

# ── Signal Processing: a noisy signal, filtered, and its spectrum ───────────
random.seed(3)
noisy, clean = [], []
for i in range(0, 133, 2):
    x = 14 + i
    s = 13 * math.sin(2 * math.pi * 2.5 * i / 132)
    clean.append((x, 32 - s))
    noisy.append((x, 32 - s - random.gauss(0, 4.2)))
bars = []
heights = [3, 4, 6, 9, 26, 10, 5, 4, 3, 7, 4, 3, 2, 3, 2, 2]
for k, h in enumerate(heights):
    cls = "hf" if k == 4 else "p"
    bars.append(f'<rect class="{cls}" x="{58 + k * 5.6}" y="{86 - h}" width="4" height="{h}" rx=".8"/>')
art["acl_acl"] = f"""
<path class="s faint" d="{poly(noisy[::2])}"/>
<path class="h draw" pathLength="1" d="{smooth(clean[::3] + [clean[-1]])}"/>
<path class="s faint" d="M54 86 H150"/>
{''.join(bars)}
<text class="th" x="16" y="84">FFT</text>
<path class="s" d="M36 81 H50 M46 78 L50 81 L46 84"/>
"""

# Fallback for an area without a drawing: a paper.
fallback = """
<rect class="p" x="58" y="12" width="44" height="72" rx="3"/>
<path class="s lk2" d="M66 26 H94"/>
<path class="s faint" d="M66 36 H94 M66 43 H94 M66 50 H90 M66 57 H94 M66 64 H86"/>
"""

order = ["nlp", "cv", "ml", "ai", "robotics", "hep", "astro", "quant_ph", "math", "bio",
         "neuro", "econ", "crypto", "systems", "hci", "audio", "pde", "cond_mat", "se", "acl_acl"]

parts = ["""{#
  One small drawing per research area, for the area pickers (onboarding and
  /interests). An area's name alone often means nothing to someone outside it,
  so each shows the field doing its typical thing: a detector's box round a cat,
  a line fitted to points, a Feynman diagram.

  GENERATED by scripts/build_area_art.py — edit the drawing there and re-run.

  Inline SVG, not images: it takes the page's colours in both themes, scales
  without blur, and Hugging Face refuses binary files in pushed history. Purely
  decorative (aria-hidden) — the area's name and description carry the meaning.
  Classes are styled in styles.css (section 15, "Area drawings").
#}
{% macro area_art(key) -%}
<svg class="cat-art" viewBox="0 0 160 96" aria-hidden="true" focusable="false">"""]
for i, key in enumerate(order):
    kw = "if" if i == 0 else "elif"
    body = "\n".join(line for line in art[key].strip().splitlines())
    parts.append(f'{{%- {kw} key == "{key}" %}}\n{body}')
parts.append("{%- else %}\n" + fallback.strip())
parts.append("{%- endif %}\n</svg>\n{%- endmacro %}\n")
open(OUT, "w").write("\n".join(parts))
print("wrote", OUT, sum(len(v) for v in art.values()), "bytes of drawing")
