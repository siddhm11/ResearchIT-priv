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

  /* ── Ask this paper ──────────────────────────────────────────────────────
     The question is echoed into the log (textContent) as it is sent; the
     server answers with the bot bubble only. A panel from 900 px; below that
     a bottom sheet with focus kept inside it until it closes. */

  var ask = document.querySelector('[data-ask]');
  if (ask) {
    var form = document.getElementById('chat-form');
    var input = document.getElementById('chat-input');
    var log = document.getElementById('chat-log');
    var keyField = form.querySelector('input[name="key"]');
    var backdrop = document.querySelector('.sheet-backdrop');
    var fab = document.querySelector('[data-ask-open]');
    var wide = window.matchMedia ? window.matchMedia('(min-width: 900px)') : { matches: true };

    var autosize = function () {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 96) + 'px';
    };
    var echo = function (text) {
      var d = document.createElement('div');
      d.className = 'msg msg-user';
      d.textContent = text;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
    };
    var send = function (text, key) {
      if (!text || form.classList.contains('htmx-request')) return;
      input.value = text;
      keyField.value = key || '';
      if (form.requestSubmit) form.requestSubmit(); else window.htmx.trigger(form, 'submit');
    };

    form.addEventListener('htmx:beforeRequest', function () {
      var text = input.value.trim();
      if (text) echo(text);
      input.value = '';
      autosize();
    });
    form.addEventListener('htmx:afterRequest', function () {
      keyField.value = '';
      log.scrollTop = log.scrollHeight;
      renderMath(log);
    });
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input.value.trim(), ''); }
    });
    var sugg = document.getElementById('ask-sugg');
    if (sugg) {
      sugg.addEventListener('click', function (e) {
        var b = e.target.closest('[data-ask-q]');
        if (!b) return;
        send(b.getAttribute('data-ask-q'), b.getAttribute('data-ask-key'));
        b.remove();
      });
    }

    var openSheet = function () {
      if (wide.matches) { input.focus(); return; }
      ask.classList.add('is-sheet');
      if (backdrop) backdrop.hidden = false;
      if (fab) fab.hidden = true;
      input.focus();
    };
    var closeSheet = function () {
      ask.classList.remove('is-sheet');
      if (backdrop) backdrop.hidden = true;
      if (fab) { fab.hidden = false; fab.focus(); }
    };
    var trap = function (e) {
      var f = Array.prototype.filter.call(ask.querySelectorAll('button, textarea, a[href]'),
                                          function (x) { return x.offsetParent !== null; });
      if (!f.length) return;
      if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
    };
    document.addEventListener('click', function (e) {
      if (e.target.closest('[data-ask-open]')) openSheet();
      else if (e.target.closest('[data-ask-close]')) closeSheet();
    });
    document.addEventListener('keydown', function (e) {
      if (!ask.classList.contains('is-sheet')) return;
      if (e.key === 'Escape') closeSheet();
      else if (e.key === 'Tab') trap(e);
    });
    if (wide.addEventListener) {
      wide.addEventListener('change', function () { if (wide.matches && ask.classList.contains('is-sheet')) closeSheet(); });
    }
    /* "Ask about this figure" in the lightbox. */
    document.addEventListener('researchit:ask', function (e) {
      openSheet();
      send(e.detail && e.detail.question, '');
    });
  }

  /* ── Outline: mark the section in view ───────────────────────────────── */

  var links = document.querySelectorAll('.outline nav a[data-sec]');

  function markActive(id) {
    links.forEach(function (a) {
      var on = a.getAttribute('data-sec') === id;
      a.classList.toggle('is-active', on);
      if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
  }

  function atBottom() {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
  }
  function lastVisibleLink() {
    var shown = Array.prototype.filter.call(links, function (a) { return !a.hidden; });
    return shown[shown.length - 1];
  }

  if (links.length && 'IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      /* At the bottom the last section wins, whatever crosses the band. */
      if (atBottom()) { markActive(lastVisibleLink().getAttribute('data-sec')); return; }
      entries.forEach(function (en) { if (en.isIntersecting) markActive(en.target.id); });
    }, { rootMargin: '-20% 0px -70% 0px' });
    /* A short last section never reaches that band, so at the bottom of the
       page the last link is the one in view. */
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        ticking = false;
        if (atBottom()) markActive(lastVisibleLink().getAttribute('data-sec'));
      });
    }, { passive: true });
    function watch() {
      links.forEach(function (a) {
        var el = document.getElementById(a.getAttribute('data-sec'));
        /* Briefing sections exist only once their fragment arrives, and not at
           all when there is no briefing; their links stay hidden until then. */
        a.hidden = !el;
        if (el && !el.hasAttribute('data-watched')) { el.setAttribute('data-watched', '1'); observer.observe(el); }
      });
    }
    watch();
    document.body.addEventListener('htmx:afterSettle', watch);   /* briefing sections arrive later */
  }
})();
