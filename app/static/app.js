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

  /* Un-saving is not disliking. "Remove" on an already-saved card used to post
     to /not-interested — the only unwind path that existed — so correcting a
     misclick was recorded as a dislike, added to the negative deque, and folded
     into the negative EWMA profile that the ranker subtracts at 0.15. The
     button now names the act it performs via data-action. */
  function endpointFor(id, action) {
    return '/api/papers/' + encodeURIComponent(id) + '/' +
           (action === 'unsave' ? 'unsave' : 'not-interested');
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
    fetch(endpointFor(id, p.action), { method: 'POST', body: payloadFor(p.vals) })
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
    var action = btn.getAttribute('data-action') || 'not-interested';

    card.classList.add('is-leaving');
    pending.set(id, {
      card: card,
      vals: vals,
      action: action,
      timer: setTimeout(function () { commitDismiss(id); }, UNDO_MS)
    });
    showUndoToast(id, action);
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

  function showUndoToast(id, action) {
    var t = makeToast(action === 'unsave'
      ? 'Removed from your library'
      : 'Removed from your feed');
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

  /* ── Save feedback ─────────────────────────────────────────────────────
     Saving used to change a label and nothing else, which understates it: a
     save is folded into the long-term EWMA profile and, past the clustering
     threshold, into the Ward clusters the whole feed is built from. The
     collection Follow button already says what it changed ("N papers added —
     your feed will reflect this now"); this applies the same courtesy to the
     single most common action in the product.

     Driven from the swap rather than from CSS on .btn-saved, so it fires on
     the ACT of saving. A CSS-only animation would replay on every already-
     saved card the Library renders. */

  function onSaved(target) {
    var btn = target.querySelector('.btn-saved');
    if (btn) {
      btn.classList.add('just-saved');
      setTimeout(function () { btn.classList.remove('just-saved'); }, 400);
    }
    var t = makeToast('Saved — your feed will lean toward this.');
    if (t) setTimeout(function () { closeToast(t); }, 2600);
  }

  document.body.addEventListener('htmx:afterSwap', function (e) {
    /* Only the card's own action row. The seed picker swaps a whole row and
       runs its own counter, and the Library renders saved cards on load. */
    var t = e.target;
    if (t && t.classList && t.classList.contains('card-actions')) onSaved(t);
  });

  document.body.addEventListener('htmx:responseError', function () {
    showError('Something went wrong. Please try again.');
  });
  document.body.addEventListener('htmx:sendError', function () {
    showError('Connection lost. Check your network.');
  });

  /* ── Search progress ───────────────────────────────────────────────────
     Two jobs, both about telling the truth about time.

     1. Tick real elapsed seconds. The shimmer is identical at 2s and at 40s,
        so a waiting user cannot tell "nearly there" from "stuck" and reloads —
        which abandons the in-flight request and starts the pipeline over. The
        measured warm p50 is ~2.8s and the first search after a restart has been
        seen at 40s, so this span genuinely needs a readout.

     2. Mark the previous results stale. They stay in the DOM until the swap
        lands, so without this the page shows skeletons above full-strength
        cards for the OLD query with nothing indicating which is live.

     Deliberately no fake stage messages ("Ranking results…" on a timer). They
     would be calibrated to the median and would therefore claim progress they
     cannot observe for the whole of a 40s cold start — misleading exactly when
     the user most needs the truth. */
  var SLOW_HINT_AFTER_MS = 8000;   /* just past the measured p90 of ~4.6s */
  var searchTimer = null;

  function searchProgressStop() {
    if (searchTimer) { clearInterval(searchTimer); searchTimer = null; }
    var hint = document.querySelector('[data-search-hint]');
    if (hint) hint.classList.remove('is-shown');
    var results = document.getElementById('search-results');
    if (results) results.classList.remove('is-stale');
  }

  function searchProgressStart() {
    var out = document.querySelector('[data-search-elapsed]');
    if (!out) return;
    var results = document.getElementById('search-results');
    if (results) results.classList.add('is-stale');

    var started = Date.now();
    out.textContent = '0.0s';
    if (searchTimer) clearInterval(searchTimer);
    searchTimer = setInterval(function () {
      var ms = Date.now() - started;
      out.textContent = (ms / 1000).toFixed(1) + 's';
      if (ms > SLOW_HINT_AFTER_MS) {
        var hint = document.querySelector('[data-search-hint]');
        if (hint) hint.classList.add('is-shown');
      }
    }, 100);
  }

  function isSearchRequest(e) {
    var el = e.detail && e.detail.elt;
    return !!(el && el.closest && el.closest('form.searchbar') &&
              el.getAttribute('hx-target') === '#search-results');
  }

  document.body.addEventListener('htmx:beforeRequest', function (e) {
    if (isSearchRequest(e)) searchProgressStart();
  });
  /* afterRequest covers success, error and abort alike — afterSwap alone would
     leave the counter running forever on a failed search. */
  document.body.addEventListener('htmx:afterRequest', function (e) {
    if (isSearchRequest(e)) searchProgressStop();
  });
})();
