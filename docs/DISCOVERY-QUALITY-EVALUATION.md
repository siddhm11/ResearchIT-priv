# Emerging research discovery: design and evaluation

Research date: 2026-09-25. Status: proposed experiment, not implemented ranking,
not a completed benchmark, and not a production relevance claim.

## What success means

A reader interested in model evaluation should discover a useful new evaluation
method without already knowing its name. A robotics reader should receive a
different feed. Freshness must mean recent source observations and new eligible
content, not merely rearranging an old index.

The 583 passing local tests establish mechanics and regressions. They do not
establish discovery quality. Existing scripts/eval_recs_quality.py uses mostly
canonical older seeds and probes pipeline behavior. scripts/eval_search_quality.py
includes human inspection, but its dated recency cases cannot certify current
emerging-topic coverage. Do not run the former against the developer DB: it
creates and deletes synthetic user records using the configured DB_PATH.

## Market observations and implications

Sources inspected on the research date; product pages establish public behavior,
not independently verified performance or access rights to their data.

| Public example | Observed emphasis | Implication for ResearchIT |
|---|---|---|
| [Hugging Face Papers](https://huggingface.co/papers/trending) | Community discovery with daily/weekly/monthly views | External attention can supply candidates before ResearchIT has traffic |
| [Semantic Scholar FAQ](https://webflow.semanticscholar.org/faq) | Library-based research feeds and relevance feedback | Build separate interest histories and give readers control |
| [ResearchRabbit guide](https://learn.researchrabbit.ai/en/articles/12454528-how-to-search-in-researchrabbit) | Iterative discovery from selected papers and connections | Evaluate useful adjacent discoveries, not only exact topic matches |
| [alphaXiv](https://www.alphaxiv.org/) | Current papers alongside researchers and organizations | Explain provenance and support following research trajectories |

An observed candidate is [JEV-as-a-Judge](https://www.alphaxiv.org/abs/2609.26550),
listed as submitted September 22. This may relate to the user's earlier “Jev”
example, but the later “GEP” name remains unconfirmed. Treat it as a candidate
case for evaluation interests, not proof of universal popularity or relevance.
No paper benchmark claim has been independently reproduced in this review.

## Code gaps that matter

- app/discovery_svc.py balances category popularity candidates.
- app/local_meta.py fetch_trending sorts citation candidates with a window
  anchored to the corpus's newest update, not necessarily the current date.
- app/turso_svc.py's remote path filters citation_count > 0 and ranks by
  citations. New zero-citation work can be excluded before personalization.
- scripts/ingest_arxiv.py exists, but its historical snapshot measurements do
  not establish today's deployed coverage. No scheduled ingestion workflow was
  found among this checkout's GitHub workflows. External scheduling is unknown.
- Refresh suppression fixes repeated exposure; it cannot retrieve missing work.

## Proposed first experiment

Keep the existing recommendation baseline and put the experiment behind a
feature flag. Build a small fresh-candidate store before retraining anything.

1. Collect new papers from primary publication sources, model/repository release
   pages, and permitted community feeds. Store canonical arXiv/DOI/model IDs,
   aliases, original publication date, revision date, source URL, observed_at,
   fetched_at, retrieval readiness, and provenance. Separate a model release
   with no paper from a research paper. Resolve duplicates across versions.
2. Snapshot available attention signals periodically, subject to source access,
   quotas, and retention rules. Without two observations, label an item new or
   currently popular; do not claim measured acceleration. Never reconstruct
   historical popularity using today's counts.
3. Normalize growth within source, topic, age, and (for videos) creator scale.
   Use minimum evidence and smoothing so a tiny denominator cannot create a
   huge trend score. Cap any one source's contribution. Reposts from the same
   origin are not independent corroboration. Keep a discovery route for good
   zero-attention work so community feeds do not become a mandatory gate.
4. Retrieve against explicit topics plus each saved-paper interest cluster.
   Candidates need usable metadata, topic fit, and clear provenance. Missing
   embeddings can use a labelled lexical/category route while encoding catches
   up; do not silently mix incompatible embedding spaces.
5. Trial a ten-card composition: six strong interest matches, three emerging
   matches, one adjacent discovery. These are experimental allocations, not
   current behavior or validated optimum. Preserve representation across
   interests; return to ordinary relevant candidates if emerging supply is weak.
6. Explain cards with source-backed statements: “New in model evaluation,”
   “Related to your saved calibration papers,” or “Attention increased this
   week.” Show source and timestamp; no fabricated trend explanations.

Trend is an additional feature after relevance eligibility, not permission for
an unrelated viral item to displace a useful paper. Preserve the current quota
and MMR baseline for comparison; do not silently change the contributor invariants.

## YouTube pilot: six to eight weeks, evidence-driven exit

Use videos as an optional explanation attached to a paper, initially with manual
review. Allow related-topic videos as a distinct label when no exact link exists.
The [YouTube Data API](https://developers.google.com/youtube/v3/docs/search/list)
supports discovery filters; its [video resource](https://developers.google.com/youtube/v3/docs/videos)
exposes metadata and public statistics. Public view counts do not establish
watch completion, factual accuracy, or research quality.

Verify exact links through arXiv/DOI/project identifiers in the description or
an authoritative project page. A shared title alone is insufficient. Classify
verified association separately from explanation quality: an exact paper link
can still accompany misleading commentary. Review accuracy, limitations,
clarity, level, language, and duration. Let users prefer quick introductions,
technical walkthroughs, or no video. Avoid autoplay and do not treat opening a
video link as finishing a paper.

Use a small curated creator set plus an audited route for smaller creators.
Compare recent attention with the creator's normal performance rather than
raw total views. Respect current API policies, deletion/refresh requirements,
and quotas before retaining observations; public access is not blanket storage
permission. No scraping of private watch histories or transcript assumptions.

During low traffic, use explicit interests, content similarity, editorial review,
and external signals. Later blend local save/helpful feedback when holdout data
supports it. Two months alone does not justify switching to collaborative models.

## Evaluation protocol

### Time-aware dataset

Start with six reader briefs: evaluation, agents/RAG, vision/world models,
robotics, medical imaging, and a deliberate two-interest profile. Include
beginners and specialists and zero-, few-, and many-save states. Recruit real
readers to validate briefs; synthetic personas alone cannot establish utility.

Collect 28 consecutive daily source snapshots prospectively. For each date,
freeze the corpus, source observations, user history, and candidate readiness.
Evaluate day T using only information observed by T. Use the first 14 days for
tuning, the next seven for validation, and the last seven as an untouched test.
Group versions and near-duplicate papers to prevent leakage. This is an initial
study size, not a claim of sufficient statistical power. Repeat longer if uncertain.

Pool candidates from the current baseline, newest-in-topic, community popularity,
content-only, and the proposed combined system, plus a random sample of eligible
new papers. Blind the system identity and ordering. Two qualified reviewers rate
relevance 0–3, usefulness, novelty to the profile, and credibility separately.
Resolve disagreements and retain both raw labels. LLM judgments can triage;
they must not be the only ground truth. Report unjudged coverage explicitly.

### Hard cases

- Important zero-citation paper, new alias, acronym collision, and ambiguous GEP.
- No-paper model release and an old paper with a new revision date.
- Viral off-topic video, inflated engagement, copied videos, misleading explainer.
- Small creator with an excellent technical walkthrough and few initial views.
- Minority interest, changing interest, niche topic with little social activity.
- Missing embeddings/metadata, stale source, API outage, deleted/private video.
- Repeated refresh, exhausted pool, dismissal, unsave, and recent-item recovery.

### Measurements and proposed gates

Targets below are pilot hypotheses, not measured results or market standards.

| Question | Measurement | Initial gate |
|---|---|---|
| Can we discover new work? | Eligible source items searchable/recommendable within 24h of first observation; publication-to-observation measured separately | >=95% readiness in supported sources |
| Is the first page useful? | Precision@10 (rating >=2); NDCG@10 on pooled graded judgments | >=8 useful cards on average; no material loss versus baseline in any reader group |
| Does emerging work reach readers? | Relevant must-surface items entering top 10 within 48h of source observation | >=80%, with denominator fixed by independent judges and profile-specific briefs |
| Is refresh fresh? | Unseen fraction on next page, conditioned on >=10 eligible unseen candidates | >=80%; zero dismissed items |
| Are interests represented? | Per-interest relevant coverage versus requested allocation | Report worst interest as well as aggregate |
| Are links trustworthy? | Audited exact paper-video match precision; explanation accuracy separately | >=98% link precision; report sample size and interval |
| Is novelty useful? | Relevant unfamiliar discoveries selected for reading, versus random exploration | Improvement without a relevance drop |
| Is it operationally usable? | p50/p95 feed latency, ingestion lag, failures, cost/eligible item | Agree latency/cost budget after measuring real model baseline |

Report confidence intervals clustered by reader/date; repeated cards are not
independent observations. Missing denominators and empty cohorts produce “not
evaluable,” not 100%. Treat 98% observed precision from a tiny sample as uncertain.

Run ablations removing freshness, community signals, and videos separately.
Compare all systems on the same snapshots, latency conditions, and reader briefs.
Require an actual lift over content-only recommendations before crediting trends.

### Human pilot and rollout

After offline and browser gates, invite 10–20 consenting readers for a formative
pilot. Counterbalance blind feed comparisons. Ask “Which would you read?” and
“Did this teach you something useful?” after actual use. Track saves followed
by returns, explicit helpfulness, hides, and weekly successful discoveries.
Keep delivery, viewport exposure, click, return, and self-reported read distinct.

A small pilot finds problems; it is not a powered A/B result. Estimate baseline
variance and minimum useful effect before a larger user-randomized experiment.
Do not optimize watch time or CTR as the primary outcome. Run in shadow mode
first, keep the baseline fallback, and release only after stale-source alerts,
scratch-store restore, browser checks, and relevance gates pass.

## Immediate order of work

1. Audit live corpus coverage/readiness and isolate evaluation storage.
2. Implement permitted source snapshots and provenance; collect prospective data.
3. Build the blinded judgment dataset and reproducible report alongside ingestion.
4. Trial fresh candidate retrieval in shadow mode; preserve the serving baseline.
5. Add reviewed video attachments, then run the human pilot.

No model training or live ranking change was performed by writing this plan.
