# Phase 5: Cold-Start Onboarding + UI Redesign

> **Status**: 📋 PLANNING  
> **Estimated effort**: ~2 weeks  
> **Depends on**: Phase 4.5 ✅ COMPLETE  
> **Research backing**: Doc 02 §4, Doc 05, Doc 06 §1-3/§5, Doc 07 §C/§D  

---

## 1. Problem Statement

### 1.1 — The Cold-Start Trap

Right now, when a new user lands on ResearchIT, they see this:

```
┌──────────────────────────────────────────┐
│  📄 ArXiv Rec              Search  Saved │
├──────────────────────────────────────────┤
│  Find Research Papers                    │
│  [__________________________] [Search]   │
│                                          │
│  Recommended for You                     │
│  📚 No recommendations yet              │
│  Save 1 or more papers to unlock recs.   │
│                                          │
│                                          │
│              (empty)                     │
│                                          │
└──────────────────────────────────────────┘
```

The user must:
1. Know what to search for
2. Search manually
3. Save papers from results
4. Wait for EWMA profiles to update
5. Refresh to see recommendations

**This is Semantic Scholar's documented drop-off trap** — requiring saves before any recommendation appears. Doc 06 calls this out explicitly:

> "Build the onboarding pipeline that will still be used at month 12: category multi-select, seed-paper import, popularity fallback. This replaces Doc 03's implied 'start empty, wait for saves' plan."

Scholar Inbox, Spotify, Netflix, and Pinterest all solve this with **active onboarding** — the user gives signal in the first 30 seconds, never sees an empty feed.

### 1.2 — The UI Problem

The current UI is a bare DaisyUI light-theme scaffold. It's functional but not presentable:
- No dark mode (research-heavy users overwhelmingly prefer dark mode)
- Browser-default typography (no personality)
- No animations or transitions (feels static/dead)
- Paper cards are plain white boxes with no visual hierarchy
- No branding beyond an emoji + text navbar

Since Phase 5 requires building new screens (onboarding wizard), this is the natural inflection point to overhaul the visual design. Bolting a new onboarding flow onto the existing bare UI would feel disjointed.

---

## 2. Architecture Decisions

### ADR 5.1 — Onboarding Strategy: Category Filter + Seed Papers (Hybrid)

**Decision**: Two-step onboarding — (1) category multi-select, then (2) seed paper search.

**Rationale** (Doc 06):
- Fixed categories alone are too coarse (Doc 05's criticism is valid)
- Pure behavioral (skip onboarding) creates the empty-feed trap
- The hybrid approach delivers both a fast coarse signal AND fine-grained embeddings
- Categories become LightGBM features in Phase 6, so they're kept permanently
- 5 seed papers immediately bootstrap EWMA profiles + Ward clusters

**What the research says**:
| Source | Finding |
|---|---|
| Rashid 2002 (IUI) | Popularity × entropy item selection beats both pure popularity and pure entropy |
| McNee 2003 (UM) | User-chosen seeds → higher retention than system-chosen seeds |
| Spotify 2025 | Onboarding features contribute 4-12% lift even when behavior is rich |
| Scholar Inbox 2025 | Hybrid of author-name search + map exploration + active-learning ratings |
| Semantic Scholar | Requires 5 library saves before recs — documented retention killer |

**Implementation**: Categories are **pool filters** (not user vectors). They restrict the Qdrant search space for the first 1-3 sessions, then fade to a LightGBM categorical feature. Seed papers are immediately treated as "saves" — entering EWMA profiles and triggering Ward clustering.

### ADR 5.2 — Category Taxonomy: Curated Groups, Not Raw arXiv

**Decision**: Present ~20-25 curated research area groups, not the raw 150+ arXiv categories.

**Rationale**:
- Raw arXiv categories like `cs.CL`, `cs.CV`, `hep-th` are opaque to most users
- Grouping into human-readable areas ("Natural Language Processing", "Computer Vision", "High Energy Physics") is what Scholar Inbox, Google Scholar, and Spotify all do
- Each group maps to 2-5 arXiv primary categories internally
- The user sees friendly names; the system stores and queries with real arXiv category codes

### ADR 5.3 — UI Framework: Keep DaisyUI + Tailwind CDN, Add Dark Theme

**Decision**: Stay on the existing DaisyUI 4 + Tailwind CDN stack. Add `data-theme="dark"` as default with a toggle.

**Rationale**:
- The stack already works and is served from CDN (no build step)
- DaisyUI has excellent dark mode support via `data-theme`
- No migration risk — purely additive CSS changes
- Adding a build step (Vite, Next.js) is unnecessary for a server-rendered HTMX app

### ADR 5.4 — Popularity Fallback: Category-Filtered Trending

**Decision**: If a user skips ALL onboarding (or selects categories but no seed papers), serve trending papers filtered by their selected categories. If they skip categories too, serve globally trending recent papers.

**Source**: Turso DB already has `citation_count` and `published` date. "Trending" = recently published + high citation velocity, queryable with a simple SQL sort.

### ADR 5.5 — Onboarding State: Persisted in SQLite, Not Just Cookie

**Decision**: Store onboarding completion state and selected categories in a new `user_onboarding` table.

**Rationale**:
- Cookie only stores `user_id` (UUID) — categories can't fit in a cookie cleanly
- SQLite is already the interaction store; adding a table is trivial
- Categories must survive server restarts (HF Spaces ephemeral filesystem writes to `/tmp`)
- The `user_onboarding` table is also the data source for Phase 6 LightGBM features

---

## 3. Proposed Changes

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 5 File Map                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NEW FILES:                                                     │
│    app/templates/onboarding.html       ← wizard UI              │
│    app/templates/partials/category_selector.html                │
│    app/templates/partials/seed_search.html                      │
│    app/routers/onboarding.py           ← wizard backend         │
│    app/static/styles.css               ← custom CSS overrides   │
│                                                                 │
│  MODIFIED FILES:                                                │
│    app/db.py                           ← user_onboarding table  │
│    app/main.py                         ← onboarding redirect    │
│    app/config.py                       ← category taxonomy      │
│    app/templates/base.html             ← dark mode, typography  │
│    app/templates/index.html            ← redesigned home        │
│    app/templates/saved.html            ← visual polish          │
│    app/templates/search.html           ← visual polish          │
│    app/templates/partials/paper_card.html ← enhanced cards      │
│    app/templates/partials/empty_recs.html ← contextual CTA      │
│    app/routers/recommendations.py      ← popularity fallback    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5.0 — UI Foundation (base.html + styles.css)

**Goal**: Transform the visual identity from "prototype" to "premium tool".

#### [MODIFY] base.html

Current state: Light-only theme, no typography, no transitions, minimal navbar.

Changes:
- Default to `data-theme="dark"` — researchers prefer dark mode
- Add theme toggle button (sun/moon icon) in navbar, persisted via `localStorage`
- Import Google Fonts: **Inter** for body, **JetBrains Mono** for code/IDs
- Add `<link>` to custom `styles.css` for animations and overrides
- Add meta viewport + SEO tags
- Redesigned navbar with glassmorphism effect and proper branding

#### [NEW] app/static/styles.css

Custom CSS for:
- Smooth page transitions (HTMX swap animations)
- Paper card hover lift effect (`transform: translateY(-2px)`)
- Gradient accent on navbar
- Category pill styling for onboarding
- Animated save button (brief pulse on click)
- Skeleton loading states
- Glassmorphism card backgrounds
- Custom scrollbar styling for dark mode

---

### 5.1 — Onboarding Flow

#### [NEW] app/routers/onboarding.py

```
GET  /onboarding          → render full wizard page
POST /api/onboarding/categories  → save selected categories (HTMX)
POST /api/onboarding/skip        → skip and go to home
GET  /api/onboarding/seed-search → search for seed papers (HTMX partial)
POST /api/onboarding/complete    → mark onboarding done, redirect to home
```

Logic:
1. `GET /onboarding` — checks if user has completed onboarding (via `user_onboarding` table). If already done, redirect to `/`. If not, render the wizard.
2. The wizard is a **2-step flow** rendered as a single page with HTMX step transitions:
   - **Step 1**: Category multi-select (grid of clickable tiles with icons)
   - **Step 2**: Seed paper search (search bar + save buttons, "Skip" option)
3. Categories are posted as JSON array and stored in `user_onboarding`.
4. Seed paper saves go through the existing `/api/papers/{id}/save` endpoint.
5. On completion, set `onboarding_completed = true` in DB.

#### [NEW] app/templates/onboarding.html

Full-page wizard (does NOT extend base.html — has its own minimal layout for focus):
- Centered layout, no navbar distractions
- Step 1: Category tiles in a responsive grid (3-4 columns desktop, 2 mobile)
  - Each tile: icon + name + description + checkmark when selected
  - Min 1, max 8 selections
  - "Select at least 1 area to continue" validation
- Step 2: Embedded search bar (reuses hybrid search backend)
  - Shows seed paper results as smaller cards with Save buttons
  - Live counter: "3/5 seed papers saved"
  - "Skip — show me trending papers" button always visible
- Progress indicator (step dots)
- Smooth transitions between steps (HTMX hx-swap with CSS transitions)

#### [NEW] app/templates/partials/category_selector.html

The category tile grid, also usable standalone (e.g., in a future settings page).

Curated category groups (20-25):

```python
CATEGORY_GROUPS = {
    "nlp": {
        "name": "Natural Language Processing",
        "icon": "💬",
        "arxiv_categories": ["cs.CL", "cs.IR"],
        "description": "Language models, text generation, information retrieval",
    },
    "cv": {
        "name": "Computer Vision",
        "icon": "👁️",
        "arxiv_categories": ["cs.CV"],
        "description": "Image recognition, object detection, video understanding",
    },
    "ml": {
        "name": "Machine Learning",
        "icon": "🧠",
        "arxiv_categories": ["cs.LG", "stat.ML"],
        "description": "Learning theory, optimization, generalization",
    },
    "ai": {
        "name": "Artificial Intelligence",
        "icon": "🤖",
        "arxiv_categories": ["cs.AI"],
        "description": "Reasoning, planning, knowledge representation",
    },
    "rl": {
        "name": "Reinforcement Learning",
        "icon": "🎮",
        "arxiv_categories": ["cs.LG"],  # RL papers are under cs.LG
        "description": "Decision making, policy optimization, multi-agent systems",
    },
    "robotics": {
        "name": "Robotics",
        "icon": "🦾",
        "arxiv_categories": ["cs.RO"],
        "description": "Control, manipulation, autonomous systems",
    },
    "hep": {
        "name": "High Energy Physics",
        "icon": "⚛️",
        "arxiv_categories": ["hep-ph", "hep-th", "hep-ex", "hep-lat"],
        "description": "Particle physics, quantum field theory, collider experiments",
    },
    "astro": {
        "name": "Astrophysics",
        "icon": "🔭",
        "arxiv_categories": ["astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE"],
        "description": "Galaxies, cosmology, stellar physics",
    },
    "quant-ph": {
        "name": "Quantum Computing",
        "icon": "💠",
        "arxiv_categories": ["quant-ph"],
        "description": "Quantum algorithms, error correction, quantum information",
    },
    "math": {
        "name": "Mathematics",
        "icon": "📐",
        "arxiv_categories": ["math.CO", "math.AG", "math.NT", "math.PR"],
        "description": "Pure and applied mathematics",
    },
    "bio": {
        "name": "Computational Biology",
        "icon": "🧬",
        "arxiv_categories": ["q-bio.BM", "q-bio.GN", "q-bio.QM"],
        "description": "Bioinformatics, genomics, protein structure",
    },
    "neuro": {
        "name": "Neuroscience",
        "icon": "🧪",
        "arxiv_categories": ["q-bio.NC"],
        "description": "Computational neuroscience, brain modeling",
    },
    "econ": {
        "name": "Economics & Game Theory",
        "icon": "📊",
        "arxiv_categories": ["econ.TH", "cs.GT"],
        "description": "Mechanism design, auctions, market models",
    },
    "crypto": {
        "name": "Cryptography & Security",
        "icon": "🔐",
        "arxiv_categories": ["cs.CR"],
        "description": "Encryption, protocols, privacy, blockchain",
    },
    "systems": {
        "name": "Systems & Networking",
        "icon": "🌐",
        "arxiv_categories": ["cs.DC", "cs.NI", "cs.OS"],
        "description": "Distributed systems, networks, operating systems",
    },
    "hci": {
        "name": "Human-Computer Interaction",
        "icon": "🖱️",
        "arxiv_categories": ["cs.HC"],
        "description": "Interface design, accessibility, user studies",
    },
    "graphs": {
        "name": "Graph Neural Networks",
        "icon": "🕸️",
        "arxiv_categories": ["cs.LG"],  # GNN papers are under cs.LG
        "description": "Graph learning, knowledge graphs, network analysis",
    },
    "audio": {
        "name": "Speech & Audio",
        "icon": "🎵",
        "arxiv_categories": ["cs.SD", "eess.AS"],
        "description": "Speech recognition, audio generation, music AI",
    },
    "acl_other": {
        "name": "Programming Languages & SE",
        "icon": "💻",
        "arxiv_categories": ["cs.PL", "cs.SE"],
        "description": "Compilers, verification, software engineering",
    },
    "cond_mat": {
        "name": "Condensed Matter",
        "icon": "🧊",
        "arxiv_categories": ["cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.str-el"],
        "description": "Materials science, superconductivity, mesoscale physics",
    },
}
```

#### [NEW] app/templates/partials/seed_search.html

An embedded search panel for the onboarding wizard — Step 2.
- Search input with HTMX `hx-get="/api/onboarding/seed-search?q=..."`
- Results rendered as compact paper cards (title + category + Save button)
- Save button calls existing `/api/papers/{id}/save` with `source="onboarding"`
- Live counter updates via HTMX out-of-band swap

---

### 5.2 — Database Changes

#### [MODIFY] app/db.py

New table:

```sql
CREATE TABLE IF NOT EXISTS user_onboarding (
    user_id              TEXT PRIMARY KEY,
    selected_categories  TEXT,        -- JSON array of group keys: ["nlp", "cv", "ml"]
    onboarding_completed INTEGER DEFAULT 0,  -- 0 = in progress, 1 = done
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
```

New functions:
- `save_onboarding_categories(user_id, categories: list[str])` — upsert categories
- `get_onboarding_state(user_id) -> dict | None` — fetch onboarding data
- `complete_onboarding(user_id)` — set `onboarding_completed = 1`
- `get_user_category_filter(user_id) -> set[str]` — returns all arXiv category codes for the user's selected groups (used by recommendation pipeline)

---

### 5.3 — Recommendation Pipeline Changes

#### [MODIFY] app/routers/recommendations.py

Add **Tier 0: Popularity Fallback** — used when user has no saves but has completed onboarding with category selections:

```python
# ── Tier 0: Category-filtered trending (for onboarded users with 0 saves) ──
if not state.has_enough_for_recs():
    category_filter = await db.get_user_category_filter(user_id)
    if category_filter:
        rec_arxiv_ids = await _trending_by_categories(category_filter, limit=REC_LIMIT)
        for aid in rec_arxiv_ids:
            paper_tags[aid] = {
                "ranker_version": _RANKER_VERSION,
                "candidate_source": "trending_category_fallback",
                "cluster_id": "",
            }
```

New function `_trending_by_categories(categories, limit)`:
- Queries Turso for recently published papers in the selected categories
- Sorted by `citation_count DESC, published DESC`
- Simple SQL query — no ML required
- Falls back to `empty_recs.html` if no categories selected

#### [MODIFY] app/main.py

Add onboarding redirect logic:

```python
@app.get("/", response_class=HTMLResponse)
async def home(request, user_id=...):
    # Check if user needs onboarding
    onboarding_state = await db.get_onboarding_state(user_id)
    if onboarding_state is None or not onboarding_state["onboarding_completed"]:
        return RedirectResponse("/onboarding")
    # ... existing logic
```

---

### 5.4 — Visual Polish (Paper Cards, Saved, Search)

#### [MODIFY] partials/paper_card.html
- Add hover lift effect (`hover:shadow-lg hover:-translate-y-0.5 transition-all`)
- Category badge with color mapping (CS = blue, Physics = purple, Math = green, etc.)
- Better abstract truncation with "Show more" toggle
- Save button with brief success animation (checkmark pulse)

#### [MODIFY] saved.html
- Stats row: total saved, category breakdown pills
- Empty state with onboarding CTA (not just "Start Searching")

#### [MODIFY] search.html
- Clean up layout, add consistent spacing
- Add "Recommended for You" section styling consistency

#### [MODIFY] empty_recs.html
- Contextual messaging based on onboarding state:
  - No onboarding → "Complete your onboarding to see recommendations"
  - Onboarding done but 0 saves → "Here are trending papers in your areas"
  - Has saves but <5 → "Save X more papers to unlock personalized recommendations"

---

## 4. User Flow Diagrams

### 4.1 — New User Flow

```
User lands on /
       │
       ▼
┌─ Has onboarding? ──┐
│                     │
│  NO                YES
│                     │
▼                     ▼
Redirect to        Show home with
/onboarding        recs (normal)
       │
       ▼
┌──────────────────────────┐
│  Step 1: Category Select │
│  "What do you research?" │
│  [NLP] [CV] [ML] [RL]   │
│  [Physics] [Bio] [Math]  │
│         [Continue →]     │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Step 2: Seed Papers     │
│  "Save a few papers"     │
│  [Search____________]    │
│  ┌─result card──Save─┐   │
│  └───────────────────┘   │
│  3/5 seeds saved         │
│  [Skip] [Done →]         │
└──────────────────────────┘
       │
       ▼
Redirect to /
       │
       ▼
┌──────────────────────────┐
│  Home with recommendations│
│  Tier 0 (trending) if    │
│    0 saves + categories  │
│  Tier 1-3 if enough saves│
└──────────────────────────┘
```

### 4.2 — Returning User (No Disruption)

Existing users (those with any interaction history) are **auto-marked as onboarded** — they never see the wizard. This is handled by checking if the user has any interactions in the `interactions` table before redirecting.

---

## 5. Implementation Order

| Step | Component | Files | Effort |
|------|-----------|-------|--------|
| **1** | UI Foundation | `base.html`, `styles.css`, `config.py` | 1 day |
| **2** | DB schema + functions | `db.py`, `test_db.py` | 0.5 days |
| **3** | Onboarding router + templates | `onboarding.py`, `onboarding.html`, `category_selector.html`, `seed_search.html` | 2-3 days |
| **4** | Main.py redirect + auto-mark existing users | `main.py` | 0.5 days |
| **5** | Popularity fallback (Tier 0) | `recommendations.py`, `turso_svc.py` | 1 day |
| **6** | Paper card + page polish | `paper_card.html`, `saved.html`, `search.html`, `empty_recs.html` | 1-2 days |
| **7** | Testing | `test_onboarding.py`, integration tests | 1-2 days |
| **8** | Deploy + verify on HF Spaces | Push, check live | 0.5 days |

**Total: ~8-10 working days**

---

## 6. Verification Plan

### Automated Tests
- `test_onboarding.py`: DB roundtrip for categories, onboarding state, category filter expansion
- `test_db.py`: New `user_onboarding` table tests
- Integration: Full onboarding flow → home redirect → trending recs visible
- Existing 176 tests: regression check

### Browser Verification
- New user flow: onboarding wizard → category select → seed search → home with recs
- Returning user: no redirect, normal flow
- Skip onboarding: trending papers shown
- Dark/light mode toggle persists across pages
- Mobile responsive: onboarding tiles reflow correctly

### Manual Verification  
- Deploy to HF Spaces
- Test with fresh browser (no cookie) — full onboarding flow
- Test with existing cookie — no interruption
- Verify trending fallback serves relevant papers for selected categories

---

## 7. What This Does NOT Cover (Deferred)

| Item | Deferred To | Reason |
|---|---|---|
| ORCID / Semantic Scholar import | Phase 5.5 (stretch) | API integration complexity |
| Active learning (2×3 grid at signup) | Phase 9+ | Needs ≥50 signups/week (Doc 07) |
| Free-text interests box | Phase 8c | Needs LLM-consumed preferences (Doc 06 cites Sanner 2023) |
| LightGBM category feature | Phase 6 | Category data stored now, used later |
| "Stay Current" vs "Lit Review" toggle | Phase 8c | Doc 07 use-case design |
