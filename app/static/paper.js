/* ═══════════════════════════════════════════════════════════════════════════
   Paper page (redesign): summary tabs, figure lightbox, outline, maths.

   Loaded only by paper_v2.html, deferred after KaTeX and its auto-render, so
   both exist by the time this runs. Never writes HTML from data: captions and
   labels go in with textContent, and KaTeX renders from text content.
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  var SUMTAB_KEY = 'researchit-sumtab';

  function stored(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode */ } }

  /* ── Maths ─────────────────────────────────────────────────────────────
     Renders $…$ and $$…$$ inside .tex elements once each. trust:false keeps
     \\href and \\url from making links; throwOnError:false shows bad TeX as
     text instead of an error. */

  function renderMath(root) {
    if (typeof window.renderMathInElement !== 'function') return;
    var scope = root && root.querySelectorAll ? root : document;
    var els = Array.prototype.slice.call(scope.querySelectorAll('.tex:not([data-tex])'));
    if (scope.classList && scope.classList.contains('tex') && !scope.hasAttribute('data-tex')) els.push(scope);
    els.forEach(function (el) {
      el.setAttribute('data-tex', '1');
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false, trust: false, strict: 'ignore', maxExpand: 1000
        });
      } catch (e) { /* leave the raw text */ }
    });
  }

  renderMath(document);
  document.body.addEventListener('htmx:afterSettle', function () { renderMath(document); });

  /* ── Summary tabs: In plain terms | Abstract ─────────────────────────── */

  var box = document.querySelector('[data-sumbox]');

  function selectTab(name, remember) {
    if (!box) return;
    box.querySelectorAll('[role="tab"]').forEach(function (t) {
      var on = t.getAttribute('data-tab') === name;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      var panel = document.getElementById(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !on;
    });
    if (remember) store(SUMTAB_KEY, name);
  }

  function plainUnavailable() {
    var tab = document.getElementById('tab-plain');
    if (tab) tab.hidden = true;
    selectTab('abs', false);
  }

  if (box) {
    box.addEventListener('click', function (e) {
      var t = e.target.closest('[role="tab"]');
      if (t && !t.hidden) selectTab(t.getAttribute('data-tab'), true);
    });
    box.addEventListener('keydown', function (e) {
      var tabs = Array.prototype.filter.call(box.querySelectorAll('[role="tab"]'),
                                             function (t) { return !t.hidden; });
      var i = tabs.indexOf(document.activeElement);
      if (i < 0) return;
      var next = null;
      if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
      else if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
      else if (e.key === 'Home') next = tabs[0];
      else if (e.key === 'End') next = tabs[tabs.length - 1];
      if (next) { e.preventDefault(); next.focus(); selectTab(next.getAttribute('data-tab'), true); }
    });
    if (stored(SUMTAB_KEY) === 'abs') selectTab('abs', false);
  }

  /* The plain summary arrives after paint. Nothing back (no honest summary,
     no key, a timeout) or a failed request means the abstract is all there is. */
  document.body.addEventListener('htmx:afterSwap', function (e) {
    var t = e.target;
    if (t && t.id === 'panel-plain' && !t.textContent.trim()) plainUnavailable();
  });
  ['htmx:responseError', 'htmx:sendError'].forEach(function (name) {
    document.body.addEventListener(name, function (e) {
      var t = e.detail && e.detail.target;
      if (t && t.id === 'panel-plain') { t.innerHTML = ''; plainUnavailable(); }
    });
  });

  /* ── Figure lightbox ─────────────────────────────────────────────────── */

  var lb = document.getElementById('lb');
  var opener = null, openLabel = '';

  function openFigure(btn) {
    opener = btn;
    openLabel = btn.getAttribute('data-label') || '';
    var caption = btn.getAttribute('data-caption') || '';
    var text = openLabel ? openLabel + ': ' + caption : caption;
    var img = document.getElementById('lb-img');
    img.src = btn.getAttribute('data-src');
    img.alt = text;
    var cap = document.getElementById('lb-cap');
    cap.textContent = text;
    cap.removeAttribute('data-tex');
    document.getElementById('lb-credit').textContent = btn.getAttribute('data-credit') || '';
    renderMath(lb);
    if (typeof lb.showModal === 'function') lb.showModal(); else lb.setAttribute('open', '');
  }

  function closeFigure() {
    if (typeof lb.close === 'function') lb.close(); else lb.removeAttribute('open');
  }

  if (lb) {
    document.addEventListener('click', function (e) {
      var btn = e.target.closest && e.target.closest('[data-fig]');
      if (btn) { e.preventDefault(); openFigure(btn); }
    });
    document.getElementById('lb-close').addEventListener('click', closeFigure);
    lb.addEventListener('click', function (e) { if (e.target === lb) closeFigure(); });
    lb.addEventListener('close', function () { if (opener) opener.focus(); });
    var ask = document.getElementById('lb-ask');
    if (ask) {
      ask.addEventListener('click', function () {
        var question = openLabel ? 'Explain ' + openLabel.toLowerCase() : 'Explain this figure';
        closeFigure();
        document.dispatchEvent(new CustomEvent('researchit:ask', { detail: { question: question } }));
      });
    }
  }

  /* ── Outline: mark the section in view ───────────────────────────────── */

  var links = document.querySelectorAll('.outline nav a[data-sec]');
  if (links.length && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        links.forEach(function (a) {
          var on = a.getAttribute('data-sec') === en.target.id;
          a.classList.toggle('is-active', on);
          if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
        });
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    function watch() {
      links.forEach(function (a) {
        var el = document.getElementById(a.getAttribute('data-sec'));
        if (el && !el.hasAttribute('data-watched')) { el.setAttribute('data-watched', '1'); observer.observe(el); }
      });
    }
    watch();
    document.body.addEventListener('htmx:afterSettle', watch);   /* briefing sections arrive later */
  }
})();
