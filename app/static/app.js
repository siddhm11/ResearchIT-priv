/* ═══════════════════════════════════════════════════════════════════════════
   ResearchIT — client script

   Deliberately small. Everything that htmx can express in markup is left in
   markup; this file covers only the three things it cannot:

     1. theme persistence
     2. progressive disclosure (abstract, "why this?")
     3. DEFERRED DISMISSAL, which is the interesting one — see below

   Removed from the previous version: the rotating "Encoding your query with
   BGE-M3…" progress messages. They narrated the implementation to someone who
   only wanted papers, and the skeleton already communicates "loading".
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  /* How long Undo stays available. Long enough to catch a misfire, short
     enough that the pending queue never grows meaningfully. */
  var UNDO_MS = 5000;

  /* ── Theme ─────────────────────────────────────────────────────────────
     Only ever writes an explicit choice. With nothing stored the document
     stays unstamped and follows prefers-color-scheme, so a user who has
     never touched the toggle tracks their OS. Icon swapping is pure CSS. */

  function currentTheme() {
    var explicit = document.documentElement.getAttribute('data-theme');
    if (explicit) return explicit;
    return window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  window.toggleTheme = function () {
    var next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('researchit-theme', next); } catch (e) { /* private mode */ }
  };

  /* ── Progressive disclosure ────────────────────────────────────────── */

  window.toggleAbstract = function (btn) {
    var p = document.getElementById(btn.getAttribute('aria-controls'));
    if (!p) return;
    var clamped = p.classList.toggle('is-clamped');
    btn.setAttribute('aria-expanded', String(!clamped));
    btn.textContent = clamped ? 'Read more' : 'Show less';
  };

  window.toggleWhy = function (btn, id) {
    var p = document.getElementById('why-' + id);
    if (!p) return;
    var hidden = p.classList.toggle('hidden');
    btn.setAttribute('aria-expanded', String(!hidden));
  };

  /* ── Deferred dismissal + undo ─────────────────────────────────────────
     The card is hidden immediately, but the POST is held for UNDO_MS. Undo
     cancels the request rather than compensating for it.

     Why not commit-then-reverse: a dismissal is folded into the user's
     negative EWMA profile, and an EWMA is a lossy running average with no
     exact inverse. A reversed dismissal would leave a permanent smudge on the
     profile. Deferring keeps it out entirely.

     NN/g's rule for repetitive destructive actions is undo over confirmation,
     since a confirm dialog charges everyone who meant it. */

  var pending = new Map();   // arxivId -> { timer, card, vals }

  function endpointFor(id) {
    return '/api/papers/' + encodeURIComponent(id) + '/not-interested';
  }

  function payloadFor(vals) {
    var body = new FormData();
    Object.keys(vals || {}).forEach(function (k) {
      if (vals[k] !== null && vals[k] !== undefined) body.append(k, vals[k]);
    });
    return body;
  }

  function commitDismiss(id) {
    var p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    fetch(endpointFor(id), { method: 'POST', body: payloadFor(p.vals) })
      .catch(function () { /* the card is already gone; a lost dismissal is
                              recoverable, an error dialog here is not worth it */ });
    if (p.card && p.card.parentNode) p.card.remove();
    topUpFeed();
  }

  function undoDismiss(id) {
    var p = pending.get(id);
    if (!p) return;
    clearTimeout(p.timer);
    pending.delete(id);
    if (p.card) p.card.classList.remove('is-leaving');
  }

  function dismissPaper(btn) {
    var id = btn.getAttribute('data-paper-id');
    if (!id || pending.has(id)) return;
    var card = btn.closest('.card');
    if (!card) return;

    var vals = {};
    try { vals = JSON.parse(btn.getAttribute('data-vals') || '{}'); } catch (e) { vals = {}; }

    card.classList.add('is-leaving');
    pending.set(id, {
      card: card,
      vals: vals,
      timer: setTimeout(function () { commitDismiss(id); }, UNDO_MS)
    });
    showUndoToast(id);
  }

  /* Leaving the page with dismissals still pending would silently drop them.
     sendBeacon survives unload, unlike fetch. */
  function flushPending() {
    pending.forEach(function (p, id) {
      clearTimeout(p.timer);
      if (navigator.sendBeacon) navigator.sendBeacon(endpointFor(id), payloadFor(p.vals));
    });
    pending.clear();
  }
  window.addEventListener('pagehide', flushPending);

  /* ── Toasts ────────────────────────────────────────────────────────── */

  function closeToast(el) {
    if (!el || el.classList.contains('is-out')) return;
    el.classList.add('is-out');
    setTimeout(function () { if (el.parentNode) el.remove(); }, 200);
  }

  function makeToast(message) {
    var host = document.getElementById('toasts');
    if (!host) return null;
    var t = document.createElement('div');
    t.className = 'toast';
    var msg = document.createElement('span');
    msg.className = 'msg';
    msg.textContent = message;          // textContent, never innerHTML
    t.appendChild(msg);
    host.appendChild(t);
    return t;
  }

  function showUndoToast(id) {
    var t = makeToast('Removed from your feed');
    if (!t) return;
    var undo = document.createElement('button');
    undo.type = 'button';
    undo.className = 'undo';
    undo.textContent = 'Undo';
    undo.addEventListener('click', function () {
      undoDismiss(id);
      closeToast(t);
    });
    t.appendChild(undo);
    setTimeout(function () { closeToast(t); }, UNDO_MS);
  }

  function showError(message) {
    var t = makeToast(message);
    if (t) setTimeout(function () { closeToast(t); }, 4000);
  }

  /* ── Infinite scroll ───────────────────────────────────────────────────
     The loader button carries only hx-trigger="click"; this observer supplies
     the scroll half by clicking it when it comes near the viewport.

     Both htmx built-ins were tried against the deployed build first and are
     documented in rec_page.html: `revealed` cannot fire when combined with a
     second trigger (htmx's poller matches the attribute exactly), and
     `intersect once` wired its handler but never fired on real scrolling.
     Owning the observer is deterministic and testable.

     rootMargin pre-loads a screen early so the next page is usually already
     in place by the time the user reaches the bottom. */

  var feedObserver = null;

  function watchLoader() {
    var more = document.querySelector('.feed-more');
    if (!more) return;
    if (!feedObserver) {
      feedObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          var el = e.target;
          feedObserver.unobserve(el);           // one shot per loader
          if (!el.classList.contains('htmx-request')) el.click();
        });
      }, { rootMargin: '800px 0px' });
    }
    if (more.dataset.watched !== '1') {
      more.dataset.watched = '1';
      feedObserver.observe(more);
    }
  }

  /* Each appended page brings its own loader, so re-arm after every swap. */
  document.body.addEventListener('htmx:afterSwap', watchLoader);
  document.addEventListener('DOMContentLoaded', watchLoader);
  watchLoader();

  /* ── Keep the feed populated ───────────────────────────────────────────
     Dismissing shrinks the list. Without this the feed drains toward empty
     as the user triages, which is the opposite of what a feed should do. */

  function topUpFeed() {
    var feed = document.querySelector('.feed');
    if (!feed) return;
    if (feed.querySelectorAll('.card:not(.is-leaving)').length > 3) return;
    var more = document.querySelector('.feed-more');
    if (more && !more.classList.contains('htmx-request')) more.click();
  }

  /* ── Wiring ────────────────────────────────────────────────────────────
     Delegated so htmx-appended pages need no re-binding. */

  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('[data-dismiss]');
    if (btn) { e.preventDefault(); dismissPaper(btn); }
  });

  document.body.addEventListener('htmx:responseError', function () {
    showError('Something went wrong. Please try again.');
  });
  document.body.addEventListener('htmx:sendError', function () {
    showError('Connection lost. Check your network.');
  });
})();
