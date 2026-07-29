/* ═══════════════════════════════════════════════════════════════════════════
   ResearchIT — Client-Side JavaScript
   Theme toggle, toasts, progressive loading messages, optimistic UI,
   abstract expand/collapse, HTMX event handlers.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Theme Toggle ─────────────────────────────────────────────────────────────

function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('researchit-theme', next);
  updateThemeIcons();
}

function updateThemeIcons() {
  const theme = document.documentElement.getAttribute('data-theme');
  // Desktop icon
  document.querySelectorAll('.theme-icon-sun').forEach(el => {
    el.classList.toggle('hidden', theme !== 'dark');
  });
  document.querySelectorAll('.theme-icon-moon').forEach(el => {
    el.classList.toggle('hidden', theme !== 'light');
  });
  // Mobile text
  document.querySelectorAll('.theme-text-dark').forEach(el => {
    el.classList.toggle('hidden', theme !== 'dark');
  });
  document.querySelectorAll('.theme-text-light').forEach(el => {
    el.classList.toggle('hidden', theme !== 'light');
  });
}

// ── Toast Notification System ────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 3000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const iconMap = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
  };

  const toast = document.createElement('div');
  toast.className = `rit-toast rit-toast-${type}`;
  toast.innerHTML = `
    <span class="rit-toast-icon">${iconMap[type] || 'ℹ'}</span>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-exit');
    toast.addEventListener('animationend', () => toast.remove());
  }, duration);
}

// ── Progressive Loading Messages ─────────────────────────────────────────────

class ProgressMessages {
  constructor(containerId, messages) {
    this.container = document.getElementById(containerId);
    this.messages = messages || [
      'Encoding your query with BGE-M3…',
      'Searching across 1.6M papers…',
      'Ranking the best matches for you…',
      'Almost there — curating results…',
    ];
    this.index = 0;
    this.timer = null;
  }

  start(intervalMs = 2500) {
    if (!this.container) return;
    this.index = 0;
    this._show();
    this.timer = setInterval(() => {
      this.index = Math.min(this.index + 1, this.messages.length - 1);
      this._show();
    }, intervalMs);
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  _show() {
    if (!this.container) return;
    this.container.innerHTML = `
      <span class="rit-progress-msg">${this.messages[this.index]}
        <span class="rit-progress-dots"><span></span><span></span><span></span></span>
      </span>
    `;
  }
}

// Global progress message instances
let searchProgress = null;
let recProgress = null;

// ── Abstract Expand / Collapse ───────────────────────────────────────────────

function toggleAbstract(btn) {
  const card = btn.closest('.paper-card');
  if (!card) return;
  const abstract = card.querySelector('.abstract-text');
  if (!abstract) return;

  const isExpanded = abstract.classList.contains('expanded');
  abstract.classList.toggle('expanded');
  btn.textContent = isExpanded ? '↓ Show more' : '↑ Show less';
}

// ── Coming Soon Click Handler ────────────────────────────────────────────────

function handleComingSoon(e) {
  e.preventDefault();
  e.stopPropagation();
  showToast('This feature is coming soon!', 'info', 2500);
}

// ── Search Loading State ─────────────────────────────────────────────────────

function initSearchHandlers() {
  const searchForm = document.getElementById('search-form');
  if (!searchForm) return;

  searchProgress = new ProgressMessages('search-progress-msgs', [
    'Rewriting query with Groq AI…',
    'Encoding with BGE-M3 embeddings…',
    'Searching Qdrant + Zilliz vectors…',
    'Fusing results with RRF…',
    'Reranking the best matches…',
    'Almost there…',
  ]);
}

// ── Recommendation Loading State ─────────────────────────────────────────────

function initRecHandlers() {
  recProgress = new ProgressMessages('rec-progress-msgs', [
    'Analyzing your interest clusters…',
    'Searching across 1.6M papers…',
    'Running LightGBM reranker…',
    'Diversifying with MMR…',
    'Curating your personal feed…',
  ]);
}

// ── HTMX Global Event Handlers ───────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
  updateThemeIcons();
  initSearchHandlers();
  initRecHandlers();

  // ── Before HTMX Request ──────────────────────────────────────────────────
  document.body.addEventListener('htmx:beforeRequest', function (evt) {
    const target = evt.detail.target;
    if (!target) return;

    // Search results loading
    if (target.id === 'search-results') {
      const loadingEl = document.getElementById('search-loading');
      const resultsEl = document.getElementById('search-results');
      if (loadingEl) loadingEl.classList.remove('hidden');
      if (resultsEl) resultsEl.classList.add('opacity-20', 'pointer-events-none');
      // Hide recs
      const recWrapper = document.getElementById('rec-wrapper');
      if (recWrapper) recWrapper.classList.add('hidden');
      // Swap button text
      document.querySelectorAll('.search-btn-text').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.search-btn-loading').forEach(el => el.classList.remove('hidden'));
      // Start progress messages
      if (searchProgress) searchProgress.start(2000);
    }

    // Recommendations loading
    if (target.id === 'rec-section') {
      if (recProgress) recProgress.start(2000);
    }
  });

  // ── After HTMX Request (success or error) ────────────────────────────────
  document.body.addEventListener('htmx:afterRequest', function (evt) {
    const target = evt.detail.target;
    if (!target) return;

    // Search results loaded
    if (target.id === 'search-results') {
      const loadingEl = document.getElementById('search-loading');
      const resultsEl = document.getElementById('search-results');
      if (loadingEl) loadingEl.classList.add('hidden');
      if (resultsEl) resultsEl.classList.remove('opacity-20', 'pointer-events-none');
      // Restore button
      document.querySelectorAll('.search-btn-text').forEach(el => el.classList.remove('hidden'));
      document.querySelectorAll('.search-btn-loading').forEach(el => el.classList.add('hidden'));
      // Stop progress messages
      if (searchProgress) searchProgress.stop();
    }

    // Recommendations loaded
    if (target.id === 'rec-section') {
      if (recProgress) recProgress.stop();
    }
  });

  // ── HTMX Error Handler ─────────────────────────────────────────────────
  document.body.addEventListener('htmx:responseError', function (evt) {
    const target = evt.detail.target;
    if (target) {
      // Show friendly error for search/rec failures
      if (target.id === 'search-results' || target.id === 'rec-section') {
        target.innerHTML = `
          <div class="rit-error-state">
            <div class="text-3xl mb-3 opacity-70">⚡</div>
            <p class="font-medium mb-1">Our servers are thinking extra hard</p>
            <p class="text-sm opacity-60 mb-4">The request took longer than expected. Please try again.</p>
            <button class="btn btn-primary btn-sm" onclick="location.reload()">
              ↻ Try again
            </button>
          </div>
        `;
      }
      // Stop any progress messages
      if (searchProgress) searchProgress.stop();
      if (recProgress) recProgress.stop();
    }
  });

  // ── Optimistic Save Feedback ─────────────────────────────────────────────
  // After a save action completes, show a toast
  document.body.addEventListener('htmx:afterSwap', function (evt) {
    const target = evt.detail.target;
    if (!target) return;

    // Check if this was a save action (the target id starts with 'actions-')
    if (target.id && target.id.startsWith('actions-')) {
      // Check if it now contains a "saved" state
      const savedBtn = target.querySelector('.rit-btn-save.saved, .btn-success');
      if (savedBtn) {
        savedBtn.classList.add('save-success');
        showToast('Paper saved to your library', 'success', 2500);
      }
    }

    // Card dismissed (card removed from DOM)
    if (target.id && target.id.startsWith('paper-') && target.innerHTML.trim() === '') {
      showToast('Paper dismissed', 'info', 2000);
    }
  });

  // ── Mobile Menu Toggle ─────────────────────────────────────────────────
  const menuToggle = document.querySelector('.mobile-menu-toggle');
  if (menuToggle) {
    menuToggle.addEventListener('change', function () {
      const menu = this.closest('.flex-none').querySelector('.mobile-menu');
      if (menu) {
        menu.style.display = this.checked ? 'flex' : 'none';
      }
    });
  }

  // ── Set Active Nav Link ────────────────────────────────────────────────
  const path = window.location.pathname;
  document.querySelectorAll('.rit-nav-link, .rit-bottom-nav-item').forEach(link => {
    const href = link.getAttribute('href');
    if (href === path || (href === '/' && path === '/')) {
      link.classList.add('active');
    } else if (href !== '/' && path.startsWith(href)) {
      link.classList.add('active');
    }
  });
});
