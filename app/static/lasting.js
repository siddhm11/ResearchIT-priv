/*
  /trending/lasting: the chart of every lasting paper, and the filter menus.

  The chart (figure.hn-strata) fetches /trending/lasting/strata.json once the
  page has drawn: [first month, last month, [months with someone new], rank,
  title, arxiv id] per paper, months counted from January 2007. Each paper is
  a line from its first HN link to its latest, stacked by its first link; the
  dots are months in which someone new linked it. Pointing at a line names the
  paper; clicking (or Enter on a focused chart) opens it. Colours come from
  the stylesheet's tokens and follow the theme. If the fetch fails the figure
  is removed: the list below is the page, the chart is a way into it.

  The menus (details.hn-menu) close when another opens, on Escape, and on a
  click outside them.
*/
(function () {
  "use strict";

  // ── Filter menus ──────────────────────────────────────────────────────────
  const menus = Array.from(document.querySelectorAll("details.hn-menu"));
  menus.forEach(function (m) {
    m.addEventListener("toggle", function () {
      if (m.open) menus.forEach(function (o) { if (o !== m) o.open = false; });
    });
  });
  document.addEventListener("click", function (e) {
    menus.forEach(function (m) { if (m.open && !m.contains(e.target)) m.open = false; });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    menus.forEach(function (m) {
      if (m.open) { m.open = false; m.querySelector("summary").focus(); }
    });
  });

  // ── The chart ─────────────────────────────────────────────────────────────
  const fig = document.querySelector("figure.hn-strata");
  if (!fig) return;
  const canvas = fig.querySelector("canvas");
  const plot = fig.querySelector(".hn-strata-plot");
  const tip = fig.querySelector(".hn-strata-tip");
  const axis = fig.querySelector(".hn-strata-axis");
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const label = function (m) { return MON[m % 12] + " " + (2007 + Math.floor(m / 12)); };
  let data = null, hot = -1, geo = null;

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function draw() {
    if (!data) return;
    const r = canvas.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, r.width, r.height);
    const lines = data.lines, N = lines.length, M = data.months, pad = 3;
    const H = r.height - pad * 2, W = r.width;
    const x = function (m) { return (m + 0.5) / M * W; };
    const y = function (i) { return pad + (i + 0.5) / N * H; };
    geo = { x: x, y: y, N: N, H: H, pad: pad };

    // years, faint, every five
    g.strokeStyle = token("--border");
    g.lineWidth = 1;
    g.setLineDash([2, 4]);
    for (let yr = 2010; yr <= 2007 + Math.floor((M - 1) / 12); yr += 5) {
      const xx = Math.round(x((yr - 2007) * 12)) + 0.5;
      g.beginPath(); g.moveTo(xx, 0); g.lineTo(xx, r.height); g.stroke();
    }
    g.setLineDash([]);

    g.strokeStyle = token("--hn-line");
    g.lineWidth = Math.max(0.5, Math.min(1, H / N));
    g.beginPath();
    for (let i = 0; i < N; i++) {
      const l = lines[i], yy = y(i);
      g.moveTo(x(l[0]), yy); g.lineTo(x(l[1]) + 1, yy);
    }
    g.stroke();

    g.fillStyle = token("--hn-dot");
    for (let i = 0; i < N; i++) {
      const yy = y(i), ms = lines[i][2];
      for (let k = 0; k < ms.length; k++) g.fillRect(x(ms[k]) - 0.9, yy - 0.9, 1.8, 1.8);
    }

    if (hot >= 0) {
      const l = lines[hot], yy = y(hot);
      g.strokeStyle = token("--hn-hot");
      g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(x(l[0]), yy); g.lineTo(x(l[1]) + 1, yy); g.stroke();
      g.fillStyle = token("--hn-hot");
      for (let k = 0; k < l[2].length; k++) { g.beginPath(); g.arc(x(l[2][k]), yy, 3, 0, 7); g.fill(); }
    }

    const last = 2007 + Math.floor((M - 1) / 12);
    axis.innerHTML = "";
    [2007, 2012, 2017, 2022, last].forEach(function (yr, i, all) {
      if (i < all.length - 1 && yr >= last) return;
      const s = document.createElement("span");
      s.textContent = yr;
      s.style.left = (x((yr - 2007) * 12) / W * 100) + "%";
      if (i === 0) s.style.transform = "none";
      if (i === all.length - 1) { s.style.left = "100%"; s.style.transform = "translateX(-100%)"; }
      axis.appendChild(s);
    });
  }

  function show(i, px) {
    if (i === hot) return;
    hot = i;
    draw();
    if (i < 0) { tip.hidden = true; return; }
    const l = data.lines[i];
    tip.textContent = "";
    const b = document.createElement("b");
    b.textContent = l[4];
    const m = document.createElement("span");
    m.textContent = "#" + l[3] + " · " + l[2].length + " months with someone new · " + label(l[0]) + " to " + label(l[1]);
    tip.append(b, m);
    tip.hidden = false;
    const pr = plot.getBoundingClientRect();
    const left = Math.max(0, Math.min((px == null ? geo.x(l[0]) : px) + 12, pr.width - tip.offsetWidth));
    let top = geo.y(i) + 10;
    if (top + tip.offsetHeight > pr.height) top = geo.y(i) - tip.offsetHeight - 10;
    tip.style.left = left + "px";
    tip.style.top = Math.max(0, top) + "px";
  }

  function at(ev) {
    const r = canvas.getBoundingClientRect();
    const i = Math.round((ev.clientY - r.top - geo.pad) / geo.H * geo.N - 0.5);
    return Math.max(0, Math.min(geo.N - 1, i));
  }

  function open(i) {
    if (i >= 0) window.location.href = "/p/" + encodeURIComponent(data.lines[i][5]);
  }

  canvas.addEventListener("pointermove", function (ev) {
    if (!geo) return;
    show(at(ev), ev.clientX - canvas.getBoundingClientRect().left);
  });
  canvas.addEventListener("pointerleave", function (ev) {
    if (ev.pointerType === "mouse") show(-1);
  });
  canvas.addEventListener("click", function (ev) {
    if (!geo) return;
    const i = at(ev);
    // A tap first names the paper; a second tap on the same line opens it.
    if (ev.pointerType !== "mouse" && i !== hot) { show(i, ev.clientX - canvas.getBoundingClientRect().left); return; }
    open(i);
  });
  canvas.tabIndex = 0;
  canvas.addEventListener("keydown", function (ev) {
    if (!data) return;
    const N = data.lines.length;
    const step = { ArrowDown: 1, ArrowUp: -1, PageDown: 25, PageUp: -25 }[ev.key];
    if (step) { ev.preventDefault(); show(Math.max(0, Math.min(N - 1, (hot < 0 ? 0 : hot + step)))); }
    else if (ev.key === "Enter" && hot >= 0) open(hot);
    else if (ev.key === "Escape") show(-1);
  });
  canvas.addEventListener("blur", function () { show(-1); });

  let pending = 0;
  function redraw() {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(draw);
  }
  if ("ResizeObserver" in window) new ResizeObserver(redraw).observe(canvas);
  else window.addEventListener("resize", redraw);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

  fetch(fig.dataset.src, { credentials: "same-origin" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (d) {
      if (!d.lines || !d.lines.length) throw new Error("empty");
      data = d;
      fig.classList.add("is-ready");
      draw();
    })
    .catch(function () { fig.remove(); });
})();
