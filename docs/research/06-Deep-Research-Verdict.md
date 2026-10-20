# ResearchIT architecture: verdict, evidence, and phased plan

**Bottom line up front.** The pure-behavioral position in Doc 03/05 is directionally right but structurally incomplete — the empirical record from every ablation I could find (Rashid 2002, Zhou 2011, Sun 2017, Nguyen 2024, Spotify 2025, Scholar Inbox 2025) says **item-level seeds + adaptive refinement beats both fixed-category questionnaires and pure-behavior-from-zero**, and onboarding cues remain a 4–37% lift even once behavioral data exists. The correct architecture for ResearchIT is a **three-layer hybrid**: a coarse arXiv-category multiselect for *filtering and prior*, seed-paper/ORCID import for the *initial behavioral profile*, and Ward clustering + medoid retrieval taking over as soon as the user crosses ~10 saved papers. Doc 05's "centroid-in-nowhere" argument is well-founded but overstated; the right takeaway is *never average across distant clusters*, not *never use subject vectors at all*. Doc 03's multi-medoid retrieval with RRF fusion has a real architectural fault — RRF lets the dominant cluster swamp minor interests, which is the exact failure mode Amin flagged — and should be replaced with **importance-weighted quota allocation with a minimum floor**, which is what Pinterest (PinnerSage "sample 3 medoids by importance"), Taobao (ULIM, RecSys 2025), and Pinterest again (Bucketized-ANN, SIGIR 2023) actually deploy. Below I resolve each contradiction across the five docs, flag the specific faults in Doc 03, give a worked example, and propose a restructured phase plan in which Phase 1 (zero-ML recommender) and Phase 2 (hybrid search) are repositioned as load-bearing infrastructure for the endgame rather than disposable scaffolding.

## The core verdict: hybrid, with behavioral doing the heavy lifting

Among the serious *persistent* academic recommenders, behavioral onboarding dominates: Semantic Scholar's Research Feeds require 5 library saves before recs appear, Google Scholar seeds from authored papers, arxiv-sanity-lite fits per-tag linear SVMs on TF-IDF once you start tagging, and Scholar Inbox (ACL 2025 Demos, Flicke et al.) — the closest direct analog to ResearchIT, also arXiv-scoped, also ~3M papers — uses a hybrid of author-name search, Scholar Maps for exploration, and active-learning rating prompts. **Zero of them use a fixed arXiv-category checkbox as the primary signal**. But zero of them are pure-behavioral either. Scholar Inbox explicitly frames its category-labeled map as *cold-start tooling*; ResearchGate uses free-text interests plus behavior; Spotify's onboarding artist-picker is seed-*items* not genres; Netflix's three-title onboarding is also items, not genres.

The empirical pattern is consistent. Rashid et al. (IUI 2002, "Getting to Know You", IUI Most Impactful Paper 2020) showed popularity × entropy item selection beats both pure popularity (users know it, but it's not diagnostic) and pure entropy (diagnostic, but users don't recognize the item); McNee et al. (UM 2003) found user-chosen seeds produce **higher retention than system-chosen seeds despite taking longer** — a direct retention argument for letting researchers pick their own seed papers. Zhou, Yang, Zha (SIGIR 2011, Functional Matrix Factorization) showed adaptive interview trees beat static elicitation by several RMSE points; Nguyen et al. (arXiv:2406.00973, 2024) confirmed adaptive two-phase elicitation still beats Rashid-style static popularity×entropy on MovieLens/Netflix in 2024. Sanner, Balog, Radlinski et al. (RecSys 2023, arXiv:2307.14225) found LLM-consumed **language-based preferences** ("I study mechanistic interpretability of transformers") are competitive with item-based CF in near-cold-start — suggesting a free-text interests box is worth more than category checkboxes. Spotify's own 2025 ablation ("Generalized User Representations", atspotify.com, Sept 2025) reports nDCG@50 drops 37.1% on favorite-artist clusters when modality encoders are removed but demographic and onboarding features stay — i.e. **behavioral dominates, but onboarding still contributes 4–12% even when behavior is rich**.

So Doc 05's pivot away from pure subject vectors was correct in direction but wrong in absoluteness. Fixed category centroids are a bad *primary* user model; they remain a useful *prior* for pool filtering, for cold-start days 1–3, and as categorical features for LightGBM reranking. The ISLE system (arXiv:2512.12760, 2025) — 1.73M arXiv+OpenAlex papers, BM25+MiniLM+RRF, uses OpenAlex topic tags as filters — is the most direct production-scale validation of "hybrid with taxonomy as filter, embeddings as signal."

## Is the centroid-in-nowhere argument actually correct?

Partially. The canonical demonstration is PinnerSage Figure 2 (Pal et al., KDD 2020): averaging embeddings for painting + shoes + sci-fi produces a vector closest to "energy-boosting breakfast," a topic related to none of the inputs. PinnerSage Table 1 quantifies this: on cosine ≥ 0.8 next-action prediction, a single decay-averaged vector achieves +25% over last-pin baseline, while an oracle 3-cluster representation achieves +98% — a ~4× gap that almost exactly matches Doc 05's folklore. The Weller et al. 2025 LIMIT benchmark (arXiv:2508.21038, Google DeepMind) formalizes this as a theoretical limit: a single dense vector cannot represent arbitrary top-k subsets, and state-of-the-art embedders including Gemini-Embedding and NV-Embed fail on constructed multi-topic queries. MIND/ComiRec/SINE/kNN-Embed ablations on Amazon Books and Taobao consistently show **15–30% Recall gains from multi-vector over single-averaged-vector**, with diminishing returns after K≈4.

**But the argument is overstated when applied generically**. Wieczorek et al. ("On the Unreasonable Effectiveness of Centroids in Image Retrieval", arXiv:2104.13643) show centroids work fine for *within-class* averaging (ReID, fashion). ColBERT token-pooling experiments (Clavié/Answer.AI, 2024) show pooling *similar* tokens at factor 2 retains 100.6% of retrieval performance, only dropping to 97% at factor 4. Brokos et al. (BioNLP 2016) used centroid-of-word-embeddings successfully for single-topic biomedical QA. The correct nuanced statement is: **averaging across distant semantic clusters drifts to unrelated regions; averaging within a coherent cluster is approximately lossless, sometimes beneficial**. This is exactly why PinnerSage clusters *first*, then uses medoids within clusters — the within-cluster representation collapse is fine, the across-cluster collapse is what kills you.

Practical implication: arXiv category tags (cs.CV, cs.CL, stat.ML) as features, filters, or auxiliary signals are empirically safe and validated (ISLE 2025; BioWordVec for MeSH in Scientific Data 2019; MedGraph 2022 showing hybrid > pure-embedding in biomedical IR). Using them as *primary user vectors to be averaged* is what's broken — which is exactly the thing Doc 01/02/04 proposed and Doc 05 correctly killed.

## Pinterest is moving away from Ward + medoids, but their reasons don't apply to you

Pinterest's lineage is well-documented through six papers: PinnerSage (KDD 2020, multi-embedding via Ward) → PinnerFormer/PinnerSAGE V3 (KDD 2022, **single 256-d embedding** via transformer + Dense All Action loss) → TransAct (KDD 2023, transformer on last 100 actions *inside* the ranker) → TransAct V2 (CIKM 2025, lifelong 16K-action sequences) → PinRec (arXiv:2504.10507, 2025, generative GPT-2-style retrieval) → PinFM (RecSys 2025, 20B-parameter foundation model with Deduplicated Cross-Attention Transformer). **Yes, Pinterest moved from multi-medoid retrieval to a single dense user vector**, and PinnerFormer beats PinnerSage-oracle R@10 0.229 vs 0.046 with a fraction of the storage. But **the reason is infrastructure, not quality**: PinnerFormer paper explicitly states multi-embedding scales poorly as a *downstream feature* — "storing 20+ 256-d float16 embeddings per row across billions of training rows is expensive" — because Pinterest has tens of ranking models consuming the embedding. ResearchIT has one ranking model and ~thousands of users; the multi-embedding storage cost is trivial. **Pinterest's motivation to abandon Ward+medoids does not apply to you**; PinnerSage's architecture remains the right reference design for a CPU-only, small-scale, multi-interest system.

PinnerSage's exact specification, which Doc 03 loosely follows, is: Ward hierarchical agglomerative clustering on pre-trained (frozen) PinSage pin embeddings over the last 90 days; medoid = arg min over cluster members of Σ ‖p_m − p_j‖²; cluster importance = Σ e^{−λ(T_now − T_i)} with **λ = 0.01** (λ = 0 pure frequency underperforms, λ = 0.1 too recent also underperforms); daily MapReduce batch + lightweight online inference on last 20 actions; **light user: 3–5 clusters, heavy user: 75–100 clusters** (no fixed K, dendrogram cut); at serve time sample **3 medoids proportional to importance**; candidates from each medoid flow to the downstream ranker (no RRF). This gives you exact numerical anchors: Doc 03's α_long=0.10 corresponds roughly to PinnerSage's λ=0.1 which they *rejected* as too recent-biased; if you're matching PinnerSage empirically, **α_long should be closer to 0.01–0.03, not 0.10**. This is a concrete parameter correction from the literature.

## The fusion fault in Doc 03: RRF lets the dominant cluster dominate

This is the most important engineering flaw to fix. Doc 03's plan is to issue K medoid queries and RRF-fuse the result lists. **RRF is the wrong fusion for this**. Cormack, Clarke, Büttcher's original SIGIR 2009 RRF paper designed it to fuse *different retrievers answering the same query* (BM25 + vector + Condorcet), where consensus is a strong signal. Across K *different* interest queries, consensus means "items near the centroid of your interests" — the exact thing multi-interest models exist to avoid. Bruch et al. (*An Analysis of Fusion Functions for Hybrid Retrieval*, SIGIR 2022) further show RRF improves Recall@K more than nDCG and loses to tuned convex combination when any labels exist.

PinnerSage itself does not use RRF. Per §5 of the paper, they sample 3 medoids proportional to importance, retrieve per medoid, and concatenate into the downstream ranker. Taobao's **ULIM** (Meng et al., RecSys 2025, arXiv:2507.10097) explicitly uses pointer-enhanced cascaded category-to-item retrieval with per-category parallel retrieval — the productionized quota approach, with reported +5.54% clicks and +11.01% orders. Pinterest's **Bucketized ANN Representation Learning** (SIGIR 2023, "Representation Online Matters") directly quotes the logic: bucketized retrieval ensures "minority-representation items are not dropped in the latter stages." ComiRec (KDD 2020) uses a controllable λ to merge K lists and still shows Recall gains over MIND — but with an *aggregation module*, not with RRF. Twitter's kNN-Embed (arXiv:2205.06205) draws candidates per cluster proportional to mixture weight — again quota, not RRF.

The correct fusion for ResearchIT, which I recommend replacing Doc 03's RRF plan with: normalize medoid importance scores to weights w_k = s_k / Σs_k; allocate F=30 feed slots as slot_k = max(⌊F·w_k⌋, F_min=3) with the remainder distributed by largest fractional part; deduplicate across lists by assigning each item to its highest-ranked list; LightGBM-rerank within each C_k and take top slot_k; merge; apply MMR (λ=0.6) over the union on BGE-M3 embeddings for fine-grained redundancy control; then apply the negative-profile penalty. This gives **importance-weighted quota with a floor** — a small NLP cluster can't be starved to zero slots even if a user's ML interest is 3× larger.

## Clustering specifics: Ward stays, with three corrections

Ward hierarchical clustering remains the right default for N=50–500 saved papers in 1024-d BGE-M3 space. At N=500, fastcluster takes under 50 ms and is deterministic. Alternatives: HDBSCAN handles noise (the −1 label is valuable for one-off saves) but degrades above ~50–100 dimensions and needs UMAP preprocessing — the Milvus HDBSCAN+BGE-M3 tutorial explicitly uses this pattern. Affinity propagation picks exemplars (medoids) automatically but is O(N²T) with convergence instability at damping/preference corners; Ward + k-medoids-per-cluster dominates it. K-means needs K and doesn't produce noise labels; GMM is ill-conditioned at 1024-d with N=500; spectral is overkill.

Three corrections to Doc 03. **First**, Ward in sklearn is Euclidean-only mathematically (Murtagh & Legendre, J. Classification 2014, require squared Euclidean); since BGE-M3 dense vectors should be L2-normalized, Euclidean and cosine are monotonically related via ‖a−b‖² = 2(1 − cos(a,b)), so *L2-normalize first, then Ward with Euclidean* gives the intended cosine behavior. If Doc 03's code does cosine Ward directly, it's silently wrong. **Second**, no fixed K; use a distance-threshold cut on the dendrogram with a cap at K_max = 20 for safety, sample top-3 by importance at retrieval — matching PinnerSage's production recipe where heavy users genuinely have 75–100 clusters. **Third**, cluster *assignment stability* is the real operational risk, not compute cost; persist cluster→medoid-paper-id mapping across reclusterings and use Hungarian matching against previous medoids on each recluster to maintain cluster IDs. At K ≤ 20 this is trivial and prevents UI churn where a user's "NLP cluster" suddenly gets renamed to cluster_7 tomorrow. Recompute nightly on the full 90-day window, patch online for the last 20 interactions — the PinnerSage two-pronged recipe.

Medoid vs centroid is empirically validated in production (PinnerSage online A/B: +4% Homefeed engagement, +2% propensity, −75% serving cost via medoid caching). Medoids are the right choice; Doc 03 is correct on this point.

## Reranking, diversity, evaluation: concrete recommendations

**LightGBM lambdarank is still correct for 2025–2026.** Semantic Scholar's production stack explicitly uses it: per Kinney et al. (arXiv:2301.10140, 2023), "Results are limited to 1000 matches, which are reranked using a trained LightGBM model." LightGBM with ~30–50 features on 500 candidates inferences in tens of microseconds per item on CPU — well under 1 ms total — leaving your entire 30ms budget for retrieval and diversity. The ACM RecSys Challenge 2024 (Ekstra Bladet news) was won with LightGBM lambdarank + feature ensembles. Doc 03's 2–5 ms estimate is actually pessimistic; real cost is sub-millisecond.

Training signal with no real users: bootstrap with citation-graph pseudo-labels exactly as SPECTER (Cohan et al., ACL 2020), SciNCL (Ostendorff et al., EMNLP 2022), and SciRepEval (Singh et al., EMNLP 2023) do — treat each paper as a query user, its cited papers = relevance 2, co-cited = relevance 1, random = 0. Supplement with author-as-user simulation: each author's first N papers form their profile, their next M papers' citations are held-out positives. These are established protocols, not hacks.

**Cross-encoder reranking on CPU is not viable in the hot path** at your latency target. Consolidated numbers from FlagEmbedding docs and multiple 2024–2026 benchmarks: `bge-reranker-v2-m3` (278M params) runs ~8 ms per pair on CPU fp32, meaning 50 pairs ≈ 400 ms, 100 pairs ≈ 800 ms — far over the 30ms budget. `ms-marco-MiniLM-L-6-v2` at ~2–4 ms per pair still costs 100–150 ms for 50 pairs. Only `ms-marco-TinyBERT-L-2-v2` (~0.3 ms/pair, ~15 ms for 50) or `jina-reranker-v1-turbo-en` fit. Doc 03's Phase 5 plan to add `BGE-reranker-v2` to the hot path is the second concrete architectural fault; if you want a cross-encoder signal, distill BGE-reranker-v2 offline into a TinyBERT student and use it as a LightGBM feature on top-20 only. Use the full BGE-reranker only for generating pseudo-labels in training, never at serving time. FlashRank uses this distillation recipe and keeps ~95% of teacher nDCG@10 at ~1/10 the latency.

**MMR λ=0.6 is a reasonable default** but the evidence across summarization, e-commerce, and search puts the sweet spot anywhere from 0.3 to 0.7 depending on the precision/discovery trade. For an academic feed, high-intent users value precision, so 0.5–0.7 is defensible — tune on offline nDCG@10. DPPs (Chen, Zhang, Zhou, NeurIPS 2018; pass Culture, arXiv:2509.10392, RecSys 2025) deliver 2–5% engagement lifts over MMR but require a learned kernel and more data; defer to v2. Pinterest replaced DPP with Sliding Spectrum Decomposition in early 2025 — DPP is no longer state-of-the-art even at Pinterest — so don't over-commit to it. Category-quota (which you already have from the fusion change above) plus MMR over the union is the right stack.

**Evaluation**. Report nDCG@10 as primary, Recall@50 as secondary, HR@10 as a sanity check, plus ILS (intra-list similarity), category entropy, and a "novel-in-top-10" count for diversity/discovery. This matches the SciDocs (Cohan et al., ACL 2020) and SciRepEval (Singh et al., EMNLP 2023) conventions. For benchmarks, use **unarXive 2022** (Saier et al., arXiv:2303.14957) — all arXiv with structured citations, best match for your 1.6M corpus — plus S2ORC (Lo et al., ACL 2020) and the Konstanz citation-recommendation benchmark (Maharjan, arXiv:2412.07713, 2024) for time-split protocols. Skip CiteULike (obsolete). Note Bruch et al.'s SIGIR 2022 caveat: RRF tuning optimizes Recall more than nDCG, so if you evaluate fusion choices use nDCG@10 as the tie-breaker — another reason the RRF → quota switch is well-founded.

**Negative signals**. Doc 03's EWMA negative profile with α=0.15 is a reasonable engineering pattern but is **not a named academic method**; the closest published analogs are eVSM (Musto 2011) and Pone-GNN's disinterest embeddings (TORS 2024). The definitive industrial reference is YouTube's 2023 paper (Xia et al., arXiv:2308.12256): using dislikes only as *input features* reduces similar recs by 22%; using them as *both input features and training labels* (negative loss target) reduces similar-content rate by **60.8%** and creator-similarity by 64.1% — a 3× gain from the richer treatment. Spotify's 2024 work (Wilm et al., arXiv:2406.04488) and Park et al.'s Pone-GNN (TORS 2024) confirm dual positive/negative embeddings beat single-encoder negative sampling. Recommended three-layer design: session-level hard filter (never re-show dismissed items in the same session); short-term item penalty at rerank `score -= α · exp(−Δt/τ_neg)` with τ_neg = 7–14 days; long-term EWMA "negative medoid" per user with MMR-style subtraction `final_score = λ·sim(cand, pos_medoid) − (1−λ)·sim(cand, neg_medoid)`. Generalize to category-level: if ≥3 dismissals hit the same arXiv category within a week, suppress that category for 2 weeks. Once you have ≥10K dismissals, retrain LightGBM with dismissals as negative labels (YouTube's 22% → 60% gain).

## Worked example: a user saves 5 papers

Suppose a new user signs up, ticks cs.LG and cs.CL on the category onboarding, pastes their ORCID so the system ingests their 4 authored papers (frozen BGE-M3 embeddings), and saves 5 papers manually: three on retrieval-augmented LLMs, two on diffusion models for image generation.

Day 1 at signup: the 9-paper profile (4 authored + 5 saved) triggers Ward clustering with L2-normalized BGE-M3 vectors. The dendrogram cut at cosine-distance threshold 0.35 produces two clusters — RAG/NLP (5 papers) and diffusion (2 papers); the remaining 2 authored papers on older topics become a third small cluster. Medoids are the three papers minimizing sum-of-squared-distances within each cluster, stored as paper IDs only (cacheable across users). Importance scores use EWMA with α_long=0.03 — correcting Doc 03's 0.10, which PinnerSage empirically rejected as too recent-biased — producing weights roughly 0.55 / 0.30 / 0.15.

Feed request: the system issues three independent ANN queries against Qdrant (dense) + Zilliz (sparse), one per medoid, retrieving top 200 each. For a 30-item feed, slot allocation gives 16 / 9 / 5 but the floor F_min=3 is respected (already satisfied). LightGBM lambdarank scores each candidate with features: max/mean/top-3 BGE-M3 cosine to medoid, sparse score, log-citation-count, age-decayed citation velocity, arXiv primary-category match to medoid, author-overlap count with saves, citation-graph overlap with saves (second-order). Per-cluster top-16/9/5 are picked, items appearing in multiple lists are assigned to their highest-ranked cluster (dedup). MMR with λ=0.6 runs over the merged 30-item set on BGE-M3 embeddings to smooth any within-quota redundancy (e.g., three near-duplicate RAG surveys get diversified). A negative-medoid penalty subtracts a fraction of similarity to the user's dismissed-items EWMA centroid. Final 30-item feed is served. End-to-end CPU latency on a modest instance: Qdrant+Zilliz queries ~10 ms combined (3 parallel), LightGBM rerank ~1 ms, MMR ~2 ms, quota + dedup ~0.5 ms, negative-profile penalty ~0.5 ms — comfortably under the 30 ms budget.

Day 30: the user has saved 80 papers and dismissed 25. Ward recluster produces 4–5 stable clusters; Hungarian matching against yesterday's medoids preserves cluster IDs for UX consistency. The onboarding category filter is now only used as a LightGBM categorical feature and a pool-prefilter when a cluster is small; the user's behavioral signal dominates. Category-level negatives suppress `cs.IR` for two weeks after the user dismisses 4 retrieval-systems papers in a row. This is the pure-behavioral regime Doc 03/05 describes, reached cleanly because the first-session experience was bootstrapped by the explicit filter rather than requiring 5 saves before any rec appears (the Semantic Scholar drop-off trap).

## Restructured phases

Amin's question 2 was whether Phase 1 (zero-ML) and Phase 2 (hybrid search, done) serve the endgame. They do, but the sequencing should change.

**Phase 0 — already complete**: hybrid BGE-M3 dense + sparse + RRF on Qdrant + Zilliz, 1.6M papers. Keep RRF *for search* (it's correct for fusing different retrievers over the *same* query); replace it with quota *for recommendations* (different queries over the same user).

**Phase 1 reframed — cold-start scaffolding, not throwaway**. Build the onboarding pipeline that will still be used at month 12: (a) arXiv category multi-select (5-second coarse filter and LightGBM feature), (b) ORCID / Semantic Scholar ID / Google Scholar author-name import that ingests authored paper embeddings as initial seeds, (c) "add 5 seed papers" library seeder, (d) a simple popularity-per-selected-category feed for the first session if users skip all three. This replaces Doc 03's implied "start empty, wait for saves" plan and addresses the Semantic Scholar empty-library trap. Infrastructure: SQLite is fine initially; migrate to Supabase when you cross 10 concurrent writes/sec. Stick with Render.

**Phase 2 reframed — behavioral profile and Ward clustering, the core recommender**. Implement the full PinnerSage-style profile: Ward + L2-normalized Euclidean clustering with dendrogram-threshold cut (cap K_max=20), medoid selection, importance scoring with α_long=0.03 (not 0.10), α_short=0.40, α_neg=0.15; Hungarian-matched stable cluster IDs; nightly batch + online patches. Retrieval: 3 medoids sampled by importance, K separate Qdrant+Zilliz queries, **importance-weighted quota fusion with F_min=3, not RRF**. Reranking: LightGBM lambdarank with ~30 features including arXiv category match, citation-graph overlap, author overlap, recency; trained initially on SPECTER/SciNCL-style citation-graph pseudo-labels + author-as-user simulation on unarXive 2022. Diversity: MMR λ=0.6 over the union. Negatives: session hard-filter + short-term item penalty + long-term EWMA negative medoid with MMR-style subtraction + category-level suppression.

**Phase 3 — evaluation and tuning before users scale**. Time-split citation-prediction evaluation on unarXive 2022 and S2ORC; author-as-user simulation for personalization metrics; nDCG@10 primary, Recall@50 secondary, HR@10 sanity; ILS + category entropy + novel-in-top-10 for diversity. Tune MMR λ, cluster cut threshold, and EWMA alphas against nDCG@10. Use SciDocs and SciRepEval as published benchmarks for comparison against SPECTER2 baselines.

**Phase 4 — Claude-generated interest summaries and distilled cross-encoder rerank**. Claude API generates human-readable per-cluster summaries ("You're reading about retrieval-augmented generation, particularly evaluation and hallucination") stored with medoids for UI. Offline, distill `BAAI/bge-reranker-v2-m3` into a TinyBERT-L2 student (FlashRank recipe); deploy on top-20 post-LightGBM as a feature, keeping hot-path latency under 30 ms. Do **not** put full BGE-reranker-v2 in the hot path.

**Phase 5 — exploration and collaborative filtering**. Epsilon-greedy exploration (5–10% of slots) against a budget of novel-category items — mirrors Spotify's BaRT (McInerney et al., RecSys 2018) and Pinterest's Warmer-for-Less (arXiv:2512.17277, 2025). LightFM / collaborative filtering when you cross the 500-user threshold Doc 03 mentions. At that scale, retrain LightGBM with dismissals as negative labels (YouTube's 3× gain).

**Phase 6 — optional upgrades, only if warranted**. Semantic IDs / TIGER-style generative retrieval (Rajput et al., NeurIPS 2023; ActionPiece, arXiv:2502.13581, Feb 2025) becomes interesting at ≥10⁵ users with rich sequence data; PinnerFormer-style single-vector transformer user models become relevant only if you ship many downstream models consuming user embeddings; DPPs with learned kernels over MMR become worth the complexity only with substantial engagement data. None of these belong in the first 12 months.

## Contradictions across Docs 01–05, resolved

Doc 01/02/04 advocate explicit subject vectors as primary user models; Doc 05 pivots to pure behavioral citing four problems (taxonomy limits, granular specificity, drift, centroid-in-nowhere); Doc 03 commits to pure behavioral with Ward + medoids + RRF. **Resolution**: all four Doc-05 objections are valid *against subject vectors as the primary vector*; none are valid against subject labels as *filters, features, and cold-start priors*. Use subject taxonomy for the first three things, use behavioral medoids for the primary user representation. This gives both 01/02/04's first-session usability and 03/05's long-run personalization.

Doc 03's α_long=0.10 contradicts PinnerSage's empirical λ=0.01 optimum with λ=0.1 explicitly rejected as too recent-biased; **resolution**: α_long=0.03 (split the difference toward PinnerSage's finding); keep α_short=0.40 and α_neg=0.15 as reasonable. Doc 03's RRF fusion contradicts PinnerSage's importance-weighted sampling, ComiRec's controllable aggregation, Taobao ULIM's quota, and Pinterest Bucketized-ANN's quota — every production multi-interest system uses quota, not RRF; **resolution**: importance-weighted quota with floor F_min=3. Doc 03's BGE-reranker-v2 in the hot path contradicts CPU latency reality (~800 ms for 100 pairs); **resolution**: distill offline, use TinyBERT student as LightGBM feature. Doc 03's LightFM deferral is correct (≥500 users is the right threshold). Doc 03's MMR λ=0.6 is a defensible default; tune on nDCG@10.

## Flagged architectural faults in Doc 03

Four concrete faults, in order of impact. First, **RRF fusion across K medoid queries** lets dominant clusters dominate — replace with importance-weighted quota + F_min floor + MMR smoothing. Second, **α_long=0.10 is empirically too aggressive** — PinnerSage's rejected value — drop to 0.03. Third, **BGE-reranker-v2 in the serving path is infeasible on CPU at 30ms** — distill offline or drop. Fourth, **cosine Ward is not mathematically Ward** if run directly; always L2-normalize first and use Euclidean Ward, or explicitly use `linkage='average', metric='cosine'` and accept the small cluster-size-regularity loss. Minor: no Hungarian-matching for cluster ID stability will cause UI churn as users save new papers.

## What changes if this verdict is right

The implementation surface looks almost identical to Doc 03 from 50 feet away — Ward clustering, medoid representatives, EWMA dynamics, LightGBM rerank, MMR diversity, Supabase/Render infra — but three subsystems differ in ways that determine whether the system actually works for multi-interest users: the fusion step switches from RRF to importance-weighted quota with a floor, the onboarding gains a coarse category filter + ORCID/seed-paper import that make the first session functional without contradicting the long-run behavioral vision, and the reranking stack uses LightGBM as the terminal CPU-path reranker with any cross-encoder signal distilled offline rather than at serving time. The system Amin built so far (hybrid search on Qdrant + Zilliz) is load-bearing infrastructure for Phase 2, not scaffolding, because the *search* RRF fusion is correct even though the *recommendation* RRF fusion is not — these are different problems with different correct answers. The strongest evidence for this architecture is that it's what Pinterest (with 1000× more users), Scholar Inbox (with the same arXiv domain and 3M papers), Taobao ULIM (with the same quota pattern at billion-scale), Semantic Scholar (with the same LightGBM terminal reranker), and the empirical onboarding literature since Rashid 2002 all converge on — hybrid, quota-merged, behaviorally-dominated, with explicit signals as priors not primary vectors.

## Changelog

### 2026-07-30 — Corpus freshness becomes a first-class constraint
**Decision:** The corpus is no longer treated as a fixed snapshot. arXiv ingestion is a standing requirement, and corpus recency ranks above further ranking-model work in priority.
**Supersedes:** Nothing explicitly, but it reorders the Phase 7-9 roadmap: Phase 8 (reranker retrain) is now gated behind Phase 7 (freshness + persistence).
**Rationale:** Measured 2026-07-30, the newest paper in the index was published 2025-05 while live arXiv holds 290,672 papers in our configured categories that we lack. For cs.AI, arXiv published nearly half again as many papers in that window as the entire index contains. No ranking improvement can surface a paper that is absent, so the ceiling on recommendation quality is currently the data, not the model.
**Action items:**
- `scripts/ingest_arxiv.py` + `scripts/ingest_backends.py` — fetch, encode, upsert to all three stores.
- New embeddings must reuse the original text construction (`title[:256] + abstract[:1024]`, `max_length=512`) so they share the existing vectors' distribution.
- Stored abstracts are no longer truncated. The 500-char cap left 90.0% of rows cut off, discarding ~60% of each abstract from the cross-encoder's input.
- Full plan and capacity analysis in `docs/phases/PHASE7-Data-Freshness-And-Capacity.md`.

### 2026-07-30 — Do not evict pre-2010 papers to reclaim disk
**Decision:** Rejected as a capacity remedy. Disk pressure is addressed by narrowing ingest scope, not by deleting history.
**Rationale:** Removing all 180,197 pre-2010 papers frees only 0.46 GB against the 0.75 GB the ingest needs, so it does not even net out. It is also expensive and irreversible: 86% of those papers have citations, 17,776 have 100+, and the damage concentrates in quant-ph (35,763), astro-ph (30,329), hep-th and mathematics — fields whose literature ages slowly and which our onboarding taxonomy offers as first-class choices. Those users would lose the foundations of their field while ML users lose nothing.
**Action items:**
- If disk is genuinely constrained, narrow ingest to the 8 core CS/ML categories (170,668 papers, 0.44 GB) instead.
- Settle the disk-usage measurement discrepancy in PHASE7 §4.2 before any deletion is considered again.

### 2026-05-29 — Direct BGE Cross-Encoder Search Reranker Integration
**Decision:** BAAI/bge-reranker-v2-m3 (cross-encoder) is integrated into the Search hot path.
**Supersedes:** Section "Reranking" (above) which strictly forbade BGE-reranker on serving path due to latency.
**Rationale:** The user explicitly requested to bypass the latency restrictions for high-relevance search. To minimize CPU latency impact on the HF Spaces free tier (2 vCPUs), BGE Cross-Encoder reranking is constrained strictly to the top-20 candidates fetched by RRF. This limits the extra latency to ~150-250ms while delivering premium-tier semantic relevance.
**Action items:**
- Implemented `app/reranker_bge_svc.py` to lazily manage and execute FlagReranker.
- Eagerly warmed up BGE Reranker in `app/main.py` lifespan alongside BGE-M3.
- Integrated BGE reranking stage in `app/hybrid_search_svc.py` after RRF and scaled its logit scores into RRF-compatible probabilities using a sigmoid function.
- Updated `app/templates/partials/search_results.html` to display "BGE Rerank" timing in the search pipeline breakdown view.
### 2026-08-12 — Keep RRF for search, but stop calling it settled
**Decision:** Search continues to use RRF. Section 3.1's rule stands, with its
justification narrowed: RRF is retained for robustness, not because it is the
higher-relevance choice. Revisit once the sparse arm is live and an evaluation
harness exists.
**Supersedes:** Nothing yet. This records a contradiction rather than resolving
it, per CLAUDE.md §6.3.
**Rationale:** OpenSearch's published BEIR comparison has rank-based RRF scoring
**3.86% lower NDCG@10 on average** than tuned score normalization, and **4.81%
lower on SciDocs** — the dataset closest to this corpus (RRF 0.1525, min-max/L2
normalization 0.1602). On SciDocs plain BM25 (0.155) also beat RRF hybrid. The
usual defence, that RRF is cheaper, does not hold either: the same benchmark
puts it 1.62% ahead at p50 and −0.033% on CPU, so the relevance cost buys
essentially nothing in latency.

Against that: the measurement is someone else's corpus and someone else's
retrievers, and normalization only wins *tuned* — untuned weights can lose to
RRF outright. This project currently has no ground truth to tune against beyond
a 60-query known-item set, so switching now would mean fitting fusion weights to
60 queries, which overfits more reliably than it improves.

Note the sharper implication of the same table: search today runs the dense arm
alone, and dense-only scores **0.1075** on SciDocs — 33% below the best hybrid.
The fusion strategy is not the biggest lever; the missing sparse arm is.
**Action items:**
- Land the `[PENDING]` lexical arm (Zilliz sparse → FTS5) from PHASE8 §1 first.
- Build the evaluation harness before touching fusion. Scholar Inbox's released
  ratings and SciRepEval are both candidate ground truth (both unverified —
  confirm before building on them).
- Only then A/B RRF against weighted score normalization on this corpus, and
  sweep `SEARCH_RRF_K` (currently 60) rather than leaving it at the copied
  default.
- If normalization wins on our own data, update section 3.1 and this changelog
  together.

### 2026-08-12 — Accept the 510-paper gap between the vector shards and the sidecar
**Decision:** Not fixing it now. Documented, not chased.
**Rationale:** `/healthz/shards` totals 1,798,838 vector points (899,456 +
697,131 + 202,251) against 1,799,348 rows in the metadata sidecar — a 510-paper
gap, 0.03% of the corpus. Those papers are reachable by lexical/FTS5 lookup and
invisible to vector search. At that scale it cannot move any retrieval metric,
and the diagnosis (which 510, and why) costs more than the outcome is worth
while the sparse arm is still unbuilt.
**Action items:**
- Fold a reconciliation check into whatever ingest work Phase 7 produces, rather
  than writing a one-off script for it now.
- Re-measure after the next ingest. A gap that *grows* is a broken ingest and is
  worth stopping for; a static 510 is not.

### 2026-08-12 — Cold-start feed churn: impression memory + epsilon-greedy Tier 0
**Decision:** Tier 0 (zero-interaction cold start) drops papers it has already
shown the user, then fills slots epsilon-greedily at eps=0.25, logging the true
selection probability. Tiers 1-3 record impressions but do not yet use them.
**Supersedes:** The Tier 0 branch of the Phase 5 cold-start design, which sorted
deterministically with `explore: []` and logged `propensity=1.0`.
**Rationale:** Measured on the deployed pipeline: a new cs.LG+cs.CL user
refreshing the feed received byte-identical papers in identical order,
indefinitely. `seen` tracks only saves and dismissals, so a reader who refreshes
without clicking anything is remembered as having done nothing. Separately,
`propensity=1.0` is a policy with no support over unchosen actions, so none of
that data could ever support the IPS/SNIPS/DR analysis §3.11 exists to enable.

Refresh for a triage feed should mean "what else have you got", not "shuffle" --
reordering the same ten papers is more disorienting than leaving them still --
so the churn comes from progressive discovery (impression memory), with only a
light stochastic component on top.

epsilon-greedy rather than Plackett-Luce/softmax specifically because its
propensities are exactly computable, in the form §3.11 already documents
("n_explore/pool_size for exploration"). A stochastic ranking policy needs
approximated top-k marginals, and an approximate propensity is a silently biased
estimate later.

**Scope note:** §4 lists epsilon-greedy under Phase 9 (gated on 500+ users).
Only the exposure-randomisation half is implemented; there is no bandit and
nothing learns from outcomes, which is the part that genuinely needs users. What
is borrowed early is what fixes a feed that never changed.

**Measured after the change:** 5 refreshes x 10 papers = 50 distinct papers,
zero repeats (was: 10 papers, repeated forever).
**Action items:**
- Extend impression memory to Tiers 1-3, which are also deterministic given an
  unchanged profile. Data is already being recorded for them.
- Revisit eps once real engagement exists; §4's 0.05 for established users is
  the intended second value.
- feed_impressions is deliberately not replicated to Turso; if restart-repeats
  become a complaint, that is the knob.

### 2026-08-12 — Turso trending fallback was missing its recency filter
**Decision:** Fixed. The Turso path now applies TRENDING_RECENCY_MONTHS, with an
unwindowed fallback in the same round trip for thin categories.
**Rationale:** `local_meta.fetch_trending` (sidecar) windowed correctly, but the
Turso SQL fallback had no date predicate at all -- it was a pure all-time
citation sort, which is exactly what the comment on TRENDING_RECENCY_MONTHS
forbids ("the same canonical papers to every user forever"). Reproduced: a new
user was served Attention Is All You Need, BERT, GPT-3, word2vec x2, Bahdanau,
RoBERTa, GRU, T5 and Seq2Seq -- nothing newer than 2020 -- labelled "trending".

Production was unaffected because the sidecar is present, which is what made
this dangerous: the Dockerfile treats a failed sidecar download as non-fatal and
falls back here, so the trigger is silent. Nothing errors; the feed just
regresses to a museum.
**Action items:**
- `update_date` is the last-revision date, not the publication date the sidecar
  derives from the arxiv id, so the two paths window slightly differently.
  Acceptable for a fallback; worth unifying if the fallback ever becomes primary.

### 2026-08-16 — Quota binds the served order, not just the candidate pool
**Decision:** The importance-weighted quota of §3.1 is now enforced at two points, not one. `merge_quota_results` interleaves clusters on a proportional stride schedule instead of concatenating their blocks, and a new terminal stage, `fusion.enforce_quota_on_ranking`, re-derives the cross-cluster arrangement of the final feed from importance after rerank and MMR have run. Order *within* a cluster remains entirely the ranker's; only the arrangement *across* clusters is quota's to decide.
**Supersedes:** Nothing in the quota formula itself — allocation is unchanged. It corrects the implicit assumption that bounding the pool's composition also bounds the page's.
**Rationale:** Quota held in the pool and then dissolved before the screen. Two mechanisms destroyed it, and neither was visible from any single stage's tests:

  1. The merge emitted all of cluster 0, then all of cluster 1. `reranker.compute_features` sets feature 1 (`candidate_position`) from the index in that merged list and feature 35 (`position_inverse`) from `1/(pos+1)`, which `heuristic_score` — the live scorer, since `RERANKER_MODE` defaults to `heuristic` — weights at 0.10. Fusion order was therefore laundered into a relevance signal: the merge decided position, position inflated the score, the score decided the feed.
  2. Even with a correct merge, `rerank_candidates` sorts globally and `mmr_rerank` selects greedily. Both are blind to clusters and both reward the high similarity the dominant cluster produces, so a long-term EWMA profile leaning one way dragged every score with it.

Measured on a synthetic two-interest user, first page of ten, before → after: importances `[0.75, 0.25]` gave `{0:10}` → `{0:8, 1:2}`; `[0.6, 0.3, 0.1]` gave `{0:10}` → `{0:6, 1:3, 2:1}`; `[0.5, 0.5]` — two *equally* important interests — gave `{0:10}` → `{0:5, 1:5}`. Verified against the live pipeline, which now serves the computed importance ratio exactly (85/15 importance → 8/1 cards).

Interleaving is proportional rather than strict round-robin on purpose: one-each alternation would hand a 25% interest half the head of the list, over-serving it as badly as block ordering under-served it.
**Action items:**
  - `_RANKER_VERSION` bumped to `v9.0_served_quota`. Tier 1 orderings before and after are not comparable; any A/B or IPS analysis must partition on it.
  - `mmr_rerank` was vectorised in the same pass — 7-9 ms → 0.20 ms for the n=100, k=60 pool it is actually called with, bringing the stage inside the §3.8 budget it had been violating. Verified identical output to the previous implementation over 60 randomised trials.
  - **Open, deferred to Phase 7:** cluster importance is `sum(1/(i+1))` over save *position* (`clustering.py`), which is steeply top-heavy — an even five-and-five save split yields 78/22, and the ratio is decided by save order rather than elapsed time. The serving quota now faithfully transmits whatever that formula produces, which makes its bias directly visible in the feed for the first time. Candidates if it needs softening: `1/sqrt(i+1)` (even split → 64/36) or true time decay. Left unchanged deliberately — it is a documented hyperparameter and there is no ground truth to tune it against until the eval harness exists.

### 2026-08-21 — MMR runs WITHIN each interest, not across them
**Decision:** MMR is applied per cluster against that cluster's own slot budget, and the per-cluster selections are then interleaved on the stride schedule. The single global `mmr_rerank` over the merged pool is gone for multi-cluster users; the single-cluster path is unchanged. `enforce_quota_on_ranking` stays as a final correction for rounding and exhausted groups.
**Supersedes:** Nothing in §3.5's parameters — λ stays 0.6. It corrects an implementation that contradicted §3.5's own division of labour: *"Quota (3.1) handles cross-cluster diversity. MMR handles within-quota redundancy."*
**Rationale:** A global MMR is cluster-blind **and it truncates**, which together made it the last stage to destroy quota. Measured on a live two-interest user: the merge handed MMR a correct 61/39 pool; MMR selected 39 papers from the dominant interest and exactly **1** from the other, dumping the remaining 60 minority papers into the exploration pool. The 2026-08-16 serving-order fix could not repair that — by the time it ran there was a single minority paper left to arrange. This is the same defect as the concatenating merge, one stage further down: quota honoured in the pool, then discarded before it reached the reader.

After the change, the same user's ranked pool went from `{minority: 1, dominant: 39}` to `{minority: 34, dominant: 22}`, and the first ten served went from 1/7 to **6/4** against an importance entitlement of 6.1/3.9. Verified live end-to-end: a user whose persisted importances were 85/15 is served 8/2.
**Action items:**
  - `_RANKER_VERSION` → `v9.1_per_cluster_mmr`. Not comparable with v9.0 or earlier.
  - Exploration is now drawn from a pool that is no longer a dump of one starved interest — worth revisiting whether uniform sampling over it is still the right choice.

### 2026-08-21 — Per-candidate cluster identity was silently broken (§3.10)
**Decision:** Cluster lookups go through a `{cluster_idx: cluster}` map. Indexing the `clusters` **list** by `cluster_idx` is a bug, and is now guarded by a test.
**Supersedes:** Nothing — this restores the §3.10 invariant that was already meant to hold.
**Rationale:** `compute_clusters` assigns `cluster_idx` from `enumerate(unique_labels)` (clustering.py:200) and then re-sorts the list by importance (clustering.py:208); `stabilize_cluster_ids` reassigns the ids outright to keep them stable across reclusterings. So the id and the list position diverge — measured as a mismatch on **every** cluster of a 3-cluster user. `recommendations.py` indexed the list by id in two places, so candidates were scored with another cluster's importance and medoid, corrupting LightGBM feature slots 23 and 24 exactly for the minority-interest papers §3.10 exists to protect.
**Action items:** Guarded by `test_cluster_idx_is_not_a_list_position` and `test_router_looks_up_clusters_by_id_not_position`.

### 2026-08-21 — The long-term profile was one paper, and most updates never landed
**Decision:** Three changes to `app/recommend/profiles.py`, none of which touches α.
  1. **Running-mean warmup.** `α_effective = max(α, 1/(count+1))`, converging to the documented α at 33 saves.
  2. **The stored accumulator keeps its magnitude.** `load_profile` normalises on read; `load_profile_raw` feeds the update path.
  3. **Per-user serialisation** of the read-modify-write, via a `WeakValueDictionary` of `asyncio.Lock`.
**Supersedes:** Nothing in §3.2 — `ALPHA_LONG_TERM` remains 0.03, and the steady state is bit-for-bit the documented EWMA. What changed is the startup transient and the concurrency behaviour.
**Rationale:** The long-term profile is the largest term in `heuristic_score` (0.40), the relevance axis MMR selects against, and the entirety of Tier 2. The UI calls it "the overall profile built from your saved papers". It was neither built from them nor overall.

*Startup.* A plain EWMA seeded with its first observation never escapes it. Because the profile was L2-normalised after every step it always had unit magnitude, so each new paper got a fixed 3% pull against a full-strength incumbent. Measured with the real function over near-orthogonal saves, `cos(profile, first_save)` was 0.996 after ten saves and **still 0.982 after forty**.

*Normalisation.* The magnitude of a running mean of unit vectors encodes how much those vectors agree — measured 0.989 for one tight interest, 0.565 for three, 0.214 for an incoherent library. Rescaling to 1 after every step destroyed that and re-inflated the incumbent. For a three-interest user over twenty saves, `cos(profile, centroid)` was 0.726 normalising each step against 1.000 accumulating. No consumer needed the unit form — the reranker, MMR and Qdrant all normalise defensively — so only the stored accumulator changed.

*Concurrency.* Every update is a read-modify-write across two awaits, unserialised. Ten concurrent saves for one user left `interaction_count` at **1**. The live database carried exactly that signature: users with 10 saves holding profile counts of 2, 4 and 6. Saving several papers quickly is the normal way to use this product, so most of what a new user told the system was discarded. Separately, the background task was passed to `asyncio.create_task` with no reference retained, which asyncio documents as collectable mid-flight; it is now held in a set until done.

Combined effect, three interests over twenty saves: `cos(profile, centroid)` 0.446 → 1.000, `cos(profile, first_save)` ~0.99 → 0.61 (≈1/√3, the correct value for one of three equal clusters).
**Action items:** Existing profiles self-heal — a stored unit vector is a valid accumulator and its `interaction_count` feeds the warmup, so no migration is needed.

### 2026-08-21 — Exploration draws from the far half of the leftovers
**Decision:** The exploration pool keeps the half furthest from the long-term profile before shuffling.
**Rationale:** The pool is by construction "everything the ranker just rejected" — the next-best matches to the user's own clusters — and the card labels those "Something different". Sampling uniformly served slightly-worse versions of what the reader already sees under a label claiming the opposite. Restricting to the far half costs nothing, since the embeddings and profile are already in hand, and keeps the draw random so §3.11's propensity arithmetic stays exact.
**Action items:** This is not true serendipity: every candidate was retrieved by one of the user's own medoids, so the pool contains nothing from outside their neighbourhoods. That needs a retrieval the pipeline does not currently make.

### 2026-08-21 — Signal integrity and resource fixes (batch)
**Decision:** Five independent corrections, each verified by reproduction before the change.

**1. Un-saving is not disliking.** "Remove" on a saved card posted to `/not-interested` because that was the only unwind path that existed, so correcting a misclick was logged as `not_interested`, added to the negative deque, and folded into the negative EWMA profile the ranker subtracts at 0.15. New `POST /api/papers/{id}/unsave` with its own `event_type`, so an undo stays distinguishable from a dislike in any future training data. The already-mixed positive contribution is not reversible — an EWMA cannot un-mix a term — so the profile washes it out at the normal rate.

**2. The vector cache cost 8× its documented ceiling.** `qdrant_svc` documents "1024 floats = 4KB each. A 25K cap = ~100MB". True of a packed buffer, false of a Python list, which boxes every element: measured, one such list is **32,824 bytes**, making the real ceiling 0.82 GB on a 16 GB box already shared with BGE-M3, a cross-encoder and a 2.7 GB sidecar. Now float32 arrays at 4,096 bytes. The hazard introduced is that `if not vec` raises on an array; two call sites did that and now test `is None`, guarded by an `ast`-based test rather than a grep, so prose mentioning the pattern is not flagged.

**3. Sidecar I/O blocked the event loop.** `local_meta` is synchronous sqlite3 over 2.7 GB and three coroutines called it directly. Measured on a 300k-row stand-in, a 72 ms query held the loop for **78 ms** — every concurrent request waits, including ones that never touch the sidecar, and this runs as a single uvicorn worker on 2 vCPUs. Moved to `asyncio.to_thread`; safe by construction because the connection is already opened read-only with `check_same_thread=False`. Note `fts_svc.search_sparse` was **already** correct — the original claim overstated the blast radius.

**4. Loading indicators never appeared.** htmx adds `htmx-request` to the element named by `hx-indicator` — the indicator itself, not an ancestor — and every use here points at a separate element, so those nodes carry both classes. Only the descendant form `.htmx-request .htmx-indicator` was defined, which cannot match one element holding both. Search and the onboarding seed step ran with no loading state at all, on requests that take seconds.

**5. No theme toggle on mobile.** `.nav` is `display:none` below 720px and the toggle lived inside it. Moved into the top bar, since the bottom nav is a four-item grid a fifth would unbalance.
**Action items:** None outstanding; all five are covered by regression tests.

### 2026-08-21 — The site gets an owned destination for a paper
**Decision:** New `GET /p/{arxiv_id}` (`app/routers/paper.py`, `app/templates/paper.html`). The card title now links there instead of to arxiv.org; the arXiv id in the card foot still goes straight to the source.
**Supersedes:** Nothing. It fills a gap doc 01 assumes exists.
**Rationale:** No URL on this site was ABOUT a paper — every card was an exit. Three structural consequences, none cosmetic: nothing was shareable (the Open Graph tags could only ever describe the feed, so a pasted link never carried a paper's title); there was nowhere to put comprehension work, since a card in an infinite feed is not a reading surface; and every session ended at arxiv.org.

The page carries full metadata, the abstract unclamped, per-paper `og:title`/`og:description`/`og:type=article`, links to the PDF and the 3D map, and a "What sits near this" section drawn from the paper's OWN embedding rather than the reader's profile — a property of the paper, identical for every visitor, which is a different and more honest claim than "more like this for you".
**Action items:**
  - The page deliberately writes NO §3.11 ranking instrumentation. A save from here genuinely has no `query_id`, `propensity` or `policy_id`, and fabricating them would corrupt the IPS/SNIPS analysis those fields exist for.
  - This is the surface Phase 8's LLM summaries and any difficulty signal should land on.

### 2026-08-21 — The click-through, and rank 0
**Decision:** Opening a paper from a ranked surface logs `event_type="click"` with the feed provenance it was served with. The card title carries `qid/pos/src/prop/pol`, and `/p/{id}` records them. A bare visit — shared link, bookmark, crawler — logs nothing.
**Rationale:** `app/db.py` has declared `click` as a valid `event_type` since the schema was written and nothing ever wrote one. The system knew which papers were saved and which were dismissed, but not which were **opened** — the difference between "scrolled past" and "read", and the densest engagement signal a feed produces. Phase 7 cannot compute per-feed CTR without it.

Only logging when a `query_id` is present is the §3.11-honouring choice: a direct visit genuinely has no propensity and no policy, and recording one as though it did would corrupt the analysis those fields exist for rather than serve it.
**Action items:**
  - **`position or None` was discarding rank 0.** Position 0 is the top of the feed — the most-clicked slot and the one CTR-by-rank most depends on — and it was stored as NULL, indistinguishable from "no position at all". All four handlers (save, not_interested, unsave, click) now use a `-1` sentinel via `events._position`. Any analysis of interactions logged before this date is missing rank 0 entirely.

### 2026-08-21 — Comprehension lands on the paper page
**Decision:** Two surfaces, one LLM-free and one generated.

**Reading density** (`app/readability.py`) — a three-band badge from notation rate, acronym rate, long-word share and sentence length. No model, no network, microseconds.

It deliberately does **not** claim to measure how hard a paper is to UNDERSTAND. doc 01 names difficulty ratings as an unfilled gap and also says why the obvious approach fails: Flesch-Kincaid and FORCAST are calibrated on general prose and mislead badly on academic writing. A paper can be conceptually simple and score unreadable. What is honestly computable from text is how dense the abstract is to READ, so the wording is "Dense with notation", never "hard".

Thresholds are the **terciles of the real corpus** (n=525), not round numbers. My first guess of 0.34/0.50 put **82.5% of papers in a single band** — a label that never discriminates costs the reader attention and returns nothing. Terciles give 33/34/33. The blurbs are comparative for the same reason: terciles support only a claim relative to this corpus. `scripts/calibrate_readability.py` re-derives them.

**"In plain terms"** (`groq_svc.explain_paper`, `GET /api/papers/{id}/explain`) — three plain sentences for a capable reader from an adjacent field, per doc 07 §A. Content-addressed cache keyed on `(arxiv_id, abstract, prompt_version, model)`, so a backfilled full abstract is a different entry and repairing the corpus cannot keep serving explanations generated from 500-char stumps. Shared across users; the corpus warms itself.

Loaded by htmx **after paint**, never inline: generation is a network call with an 8s ceiling and the abstract must not wait behind it. When there is no honest summary — truncated input, no API key, a timeout, or the model's own `INSUFFICIENT` refusal — the endpoint returns empty and the section simply does not appear. An empty slot is quieter than an apology the reader cannot act on. The output is labelled "Generated from the abstract" and never presented as the paper's own words.
**Action items:**
  - Verified live: the badge separates *Attention Is All You Need* (accessible) from a convergence-theory paper (specialist).
  - `groq` is absent from the local venv by design, so the degradation path is what runs in development — which is a useful default, since it is also what runs whenever the API is down.

### 2026-08-21 — Corpus health, and the one defect it has
**Measured against live Turso:**

| | |
|---|---|
| total papers | 1,799,348 |
| newest `update_date` | **2026-07-30** |
| abstracts at the 500 cap | **1,630,273 (90.6%)** |
| abstracts missing | 0 |
| titles missing | 0 | 
| `citation_count` NULL | 0 |

**The corpus is NOT stale.** PHASE7 and the 2026-08-16 audit both assumed the newest paper was 2025-05; ingestion has clearly run since, and the index is three weeks behind live arXiv. That removes freshness from the problem list.

**Truncation is the corpus's one real defect**, and it degrades four things simultaneously: the card and paper page show a stump; `groq_svc.explain_paper` summarises that stump; `readability` scores it; and the search cross-encoder reranks on it.

Sampling 20 truncated rows against the arXiv API: **median full abstract 976 chars against the 500 stored, 17/20 materially longer, mean gain +451 chars.** So the repair roughly doubles the text, and the premise is confirmed rather than assumed.

`scripts/backfill_abstracts.py` does the repair — dry-run by default, resumable (selection is driven by the defect, so repaired rows drop out), and ordered by usefulness rather than id, so the first hour delivers most of the user-visible benefit. Full sweep is ~16,303 arXiv requests at 3s ≈ **13.6 hours**. It is a script and not a background job deliberately: a bulk mutation of the production metadata store is a decision for a person, not something a request handler starts.

### 2026-08-21 — Swallowed exceptions are now diagnosable
**Decision:** `app/errors.py`; `errors.report(module, message, exc)` prints the exception type, the failing file and line, a trimmed traceback of OUR frames, and any chained cause. Wired into the Tier-1 catch-all and the qdrant, turso and events handlers.
**Rationale:** `grep -rn "traceback|exc_info" app/` returned **zero** hits against 95 `except Exception` handlers. Every production failure was one line with no traceback — and the handler for the feed sits at the bottom of a **473-line** try covering clustering, quota, retrieval, metadata, reranking, MMR and labelling, so `multi-interest preprocessing failed: KeyError('x')` narrowed the fault to "somewhere in the feed" while the user silently dropped to Tier 2.

Swallowing is the correct choice — a feed that degrades beats one that 500s, and those paths are deliberate. Swallowing *silently* was not. Output goes through `print` to keep the §5.3 convention and the existing `[module]` log filters.
**Action items:** 91 handlers still use a bare `print`. The four upgraded are the ones where a silent failure is genuinely hard to trace; the rest can follow opportunistically.

### 2026-08-21 — Old-style arXiv ids were collapsing onto their category
**Decision:** `arxiv_svc._normalise_id` rewritten. Prefix stripping and version stripping are separate steps rather than one regex.
**Rationale:** The old pattern was `[^\s/v]+`, which stops at a slash. Old-style ids carry a category prefix — `math/0309136v1`, `hep-ph/0512038v2`, `acc-phys/9502001` — and **115,604 papers in the corpus (6.4%)** use that form. Every one normalised to just its category: `math/0309136v1` and `math/0511124v2` both became `math`.

Two consequences, both silent. A lookup by the real id missed, so the arXiv fallback returned nothing for 6.4% of the corpus; and distinct papers collided on one key, so one paper's metadata could be returned for another. That is the same defect class the codebase already documents for Qdrant point ids. The character class also treated a literal `v` as a terminator.

Found while writing the backfill, which needed to key results by id. None of the existing `_normalise_id` cases covered an old-style id, which is why it survived.
**Action items:** Six old-style cases added to the parametrised test, plus a collision test and a §3.9 string-type test.

### 2026-08-21 — Backfill run 1, and two bugs it exposed
**Result:** 335 rows repaired before arXiv rate-limited the run; 9 of 20 batches lost to 429.

**The truncation predicate was wrong.** `length(abstract_preview) >= 500` matches a repaired 976-character abstract just as well as a truncated 500-character one, so the script would have re-selected its own completed work forever — it was **not resumable**, despite a test asserting that it was. It also counted the 193,689 legitimately-long abstracts as damaged, overstating the job by 13%. The correct predicate is `= 500` exactly, and the true figure is **1,436,584 truncated (79.8%)**, not 1,630,273 (90.6%).

**The rate limiting was wrong.** Reusing `arxiv_svc.fetch_metadata_batch` was the mistake: it is tuned for small interactive lookups and fans any input into 20-id sub-requests at ~3/s, so a "batch of 100" was a burst of five requests and the 3s pause sat between bursts rather than between requests. The script now issues exactly one request per batch, at 200 ids, with exponential backoff on 429.

Corrected estimate for a full sweep: **~7,183 requests, ~10 hours** — better than both earlier figures, and for the first time counting both the arXiv leg and the write leg.

### 2026-08-24 — Repairing Turso does not reach production by itself
**Finding:** `turso_svc.fetch_metadata_batch` reads in this order: in-process LRU → **local sidecar** → Turso. The sidecar (`/app/data/metadata.sqlite`, 2.7 GB) carries `abstract_preview` for **all 1,799,348 rows**, so in production Turso is essentially never consulted for metadata. It is opened `mode=ro` and baked into the image at build time from a **pinned** HF dataset revision (`d241ff1c…`, Dockerfile:49).

So a Turso backfill is **invisible to users** until the sidecar is rebuilt. The full publish path is four steps, and only the first is a code/data operation:

1. Backfill Turso — ~15.5 hours, resumable.
2. `scripts/build_metadata_sidecar.py` — rebuild the 2.7 GB file from Turso.
3. Upload to the `siddhm11/researchit-metadata` HF dataset.
4. Repoint `METADATA_SIDECAR_URL` and rebuild the Space.

Worth stating plainly because the obvious mental model — "fix the database, the app shows the fix" — is wrong here, and acting on it would have meant a long job with no visible result.

**Safety note:** the repair is an in-place overwrite with no backup, but the old truncated value is a strict PREFIX of the new one, so a repair can only ever add text. There is no state in which it loses information.
**Action items:** Steps 2-4 require the HF account and a Space redeploy, so they are the developer's. Step 1 is running.

### 2026-08-24 — Backfill estimate, finally measured
**Decision:** `SECONDS_PER_BATCH = 7.75`, taken from real batches, replaces the modelled arithmetic.
**Rationale:** Four estimates, three of them wrong: 13.6h counted only the arXiv pause and ignored the writes; 22.6h added a guessed write cost but still used the inflated 90.6% truncation figure; 10h fixed the count but kept the guess. Timing real batches (400 rows in 15s, 200 in 8s) gives ~7.75s per batch of 200 and **~15.5 hours** for the remaining 1,436,005 rows. Validated before launch: 400/400 fetched, zero missing, zero 429s, and the truncated count fell by exactly the number written on each run — so it is genuinely resumable, which the earlier version was not.


### 2026-09-24 — Fresh discovery, explicit interests, and recoverable reading journeys
**Decision:** Refresh should offer mostly fresh relevant papers while recent
papers remain accessible. Apply seven-day impression exclusion before personalized
retrieval, retaining cluster quotas/MMR. Retry once without that exclusion only
when no ranked candidates remain, and label repeats. Cold-start recycling keeps
fresh candidates first and no longer deletes history. Bind feed cursors to users
and serialize generation/impression writes per user in the serving process.
**Supersedes:** The Tier-0-only impression policy from 2026-08-12. The user
explicitly selected mostly-fresh refresh with recent items accessible.
**Rationale:** Unchanged user vectors should not imply identical feed refreshes.
Passive exposure is temporary eligibility state, not negative preference. A
history view lets readers recover papers without keeping discovery static.
**Action items implemented locally:** editable category preferences, suggested
seed papers, recently opened/discovered views, feedback-state recovery fixes,
and regression coverage. Keyword retrieval is independent of encoding success.
An open never changes a save/dismissal decision or updates the EWMA profile.

**Implementation reconciliation:** FTS5 is already implemented and preferred;
older “dense-only”/“sparse pending” statements above are historical. The heuristic
is the recommendation default, with optional LightGBM. Four core user tables
replicate to Turso, not every table. Collection follows and exposure logs remain
outside that backup contract. Search/recommendation evaluation scripts exist,
but production relevance, freshness, and recovery need separate validation.
**Not changed:** fusion formulas, EWMA parameters, Ward hyperparameters, MMR
lambda, search reranker depth, and model training. YouTube linking and a verified
live-trending signal are deferred. See CURRENT-STATE.md and DISCOVERY-PLAN.md.

### 2026-09-25 — Separate source candidates from relevance evidence
**Decision:** Add a read-only Hugging Face Daily Papers adapter and a reproducible
recommendation audit outside the serving path. Source votes are attention signals,
not personalized relevance labels. Report synthetic mechanics, real source
observations, model-file inspection and blocked semantic checks separately.
**Supersedes:** No ranking invariants or production defaults.
**Rationale:** The user requested a serious recommendation/embedding test suite
and a readable HTML dashboard. A passing mechanical suite cannot establish that
readers receive useful emerging research. A learned ranker cannot replace ongoing
external discovery of future papers.
**Action items:** Adapter and audit implemented locally; staged ingestion,
independent relevance judgments, shadow comparison and live validation remain
before promoting Hugging Face candidates into the feed.

### 2026-09-25 — Implement the first HF shadow integration milestone
**Decision:** The user approved implementing the visual plan. Add separate local
SQLite source storage, a serialized scheduled worker, compatible local candidate
preparation, and source-capped comparison against a scratch-profile baseline.
**Supersedes:** “Collection and readiness proposed” status for the local shadow
milestone only. Production serving and existing algorithm constants are unchanged.
**Rationale:** Dated, recoverable source evidence and inspectable comparisons are
needed before HF candidates can enter the live feed. Bound retries, invalidate
changed-text embeddings, and reject stale/future candidates. Report source failures.
**Action items:** Implemented locally with pipeline regressions. Deploy a durable
worker and establish real-model/relevance evidence before an opt-in serving pilot.

### 2026-09-30 — Multi-interest from two saves, finer interest cap, grounded exploration
**Decision:** The user approved four changes after a trace of their own interests
(NLP, CV, ML, AI, Robotics, Software Engineering) with production vectors:
1. The multi-interest path (quota + per-cluster MMR) starts at 2 saves. Below 5,
   each save is its own interest (`per_save_clusters`); Ward still starts at 5.
2. `AVG_CLUSTER_SIZE_FLOOR` 4 → 2, so K ≤ ceil(N/2) (still ≤ `K_max = 7`), and
   `compute_clusters` takes the finest dendrogram cut at or below K that leaves
   no singleton before the existing singleton merge.
3. Exploration draws from ticked interests that no cluster medoid lists in its
   categories. Otherwise it keeps the far-half-of-own-candidates pool.
4. Tier 0 balances per ticked interest instead of per arXiv code. It ranks by
   citations per month since publication and gives every third slot per
   interest to a paper from the corpus's last three months.
**Supersedes:** Tier 2 EWMA single-vector search as the 3–4-save tier (it remains
the fallback); the soft cap of ceil(N/4); exploration drawn only from the reader's
own retrieval neighbourhoods; Tier 0 per-code round-robin by raw citations.
**Rationale:** Measured: averaging Llama 3 + π0 + OpenHands returned generic
"LLM agents" surveys matching none of them. Seven saves across five interests
allowed only 2 clusters, and 12 across six allowed only 3 (5 at the new floor),
while single-interest readers are unchanged. Lowering the floor alone collapsed
a noisy 3 + 2 reader into one cluster (the singleton merge folded the split
pair into the majority), which the singleton-free cut fixes. Tier 0 held 0 of
200 papers from 2026 and gave Computer Vision 22 of 200. Replay method and
figures are in the 2026-09-30 feed trace. There is no judged relevance set yet,
so these rest on the replay and on the invariants, not on an offline metric.
**Action items:** Implemented with regression tests. Refresh citation counts
(zero for every paper since 2025-06) and ingest papers after 2026-07-30, or
Tier 0's fresh lane has no quality signal. Build a small judged set before
tuning further; multi-medoid retrieval for loose clusters is deferred until then.

### 2026-10-04 — Groq models are an ordered, monitored failover chain
**Decision:** `groq_svc` no longer names a model. It walks `GROQ_MODELS`
(default `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`),
benching a model Groq reports as retired (1 h) or rate-limited (Retry-After).
Output cut off at the token cap is never served as if complete. `/healthz/deep`
reports model availability as `services.llm`, and the keepalive workflow fails
once the primary model is gone, even while a fallback answers.
**Supersedes:** the single hard-coded `llama-3.3-70b-versatile` (PHASE3 doc,
historical).
**Rationale:** Groq shut Llama 3.3 70B down on 2026-08-16. Query rewrite, the
search overview and explanations all degrade silently by design, so they were
off for seven weeks with no symptom. Qwen 3.8 leads on measured latency (rewrite
p50 114 ms vs ~450–490 ms for gpt-oss, inside the 285–642 ms encode it overlaps)
and on grounding read side by side, despite Groq's Preview label. The chain and
the alert are what make a Preview primary acceptable.
**Action items:** Done in code. Re-run `scripts/eval_groq_models.py` whenever
Groq announces a deprecation. Pending: `_looks_academic` matches every word as an
"acronym" under IGNORECASE, so queries of more than 6 words are never rewritten.
That is a search-behaviour change, so it is left for a separate decision.

### 2026-10-09 — Paper age is measured from first publication, not last revision
**Decision:** Every age-derived reranker feature (5 age, 6 recency, 9 year
difference, 13 is-newer, 17 citations per year) is computed from the month of
first submission, decoded from the arXiv ID (`local_meta.pub_year_month`, mid-
month). The metadata date is only the fallback for IDs that do not parse.
**Supersedes:** ageing from `published`, which `turso_svc._to_paper_dict` fills
with `update_date`, the date of the paper's latest arXiv version. The Phase 6
handoff already documented feature 5 as "days since publication"; the source
was wrong, not the intent.
**Rationale:** Revised classics were scored as new papers: "Attention Is All You
Need" (2017, revised 2023-08-03) collected recency 0.098 instead of 0.001. At
least 69,399 papers first posted 2007–2022 carry a 2023-or-later revision date.
Measured before shipping (`scripts/embedding_eval/validate_steps.py`, 272
simulated readers built from citation links): held-out right answers reached the
top 10 for 3.3 points more readers (95% CI +1.5 to +5.5; 9 better, 0 worse), and
the share of recent papers in the top 10 did not fall (43.0% → 43.8%). On 526
real readers' votes (Scholar Inbox ratings) it was neutral (−0.002 AUC, CI spans
zero). After implementation the installed code reproduced the reference top 10
for 100% of readers at pool sizes 30, 60 and 100.
**Action items:** Done in `app/recommend/reranker.py` with tests. LightGBM mode
(off by default) now receives the corrected age inputs. The model was trained on
the old source, so re-check it before switching `RERANKER_MODE`. Card display
(`year` in `turso_svc`) still shows the revision year; that is a separate UI fix.
Step 2 of the upgrade plan (similarity rescaling) was measured but not shipped:
as written it cuts the share of last-12-month papers in the top 10 from 43% to
10%, so the recency weight needs a product decision first
(`scripts/embedding_eval/step2_tradeoff.py`).

### 2026-10-09 — Personalised scoring: rescaled similarities, recency weight 0.45
**Decision:** In `heuristic_score`, readers with a profile get their long- and
short-term similarities rescaled across the candidate set (as the cold-start
branch already did with the Qdrant cosine), and recency is weighted 0.45
instead of 0.15. The cold-start branch keeps 0.15. Chosen by the project owner
as option C of a measured relevance/freshness frontier.
**Supersedes:** the CLAUDE.md §3.4 weight budget `0.40*lt + 0.25*st + 0.15*recency
+ 0.10*position - 0.15*neg` for the personalised branch (updated there).
**Rationale:** Raw profile similarities barely vary across a reader's candidates
(weighted spreads: 0.006 long-term, 0.005 short-term, against 0.026 recency and
0.012 position), so recency and position decided the order. Rescaling alone
lifted the held-out right answer's top-10 rate from 55% to 74% but cut
last-12-month papers in the top 10 from 43% to 10% (they are 6% of the candidate
pool: today's freshness came from promoting less relevant new papers). The
frontier (`scripts/embedding_eval/step2_tradeoff.py`, 272 simulated readers +
526 real readers' Scholar Inbox votes):

| Variant | Top-10 hit | Recent in top 10 | Real-reader AUC |
|---|---|---|---|
| Old formula | 55.1% | 43.0% | 0.684 |
| Rescaled, recency 0.15 | 73.5% | 9.8% | 0.715 |
| Rescaled, recency 0.30 | 72.4% | 17.9% | 0.712 |
| **Rescaled, recency 0.45 (chosen)** | **68.4%** | **28.3%** | **0.703** |
| Rescaled, recency 0.60 | 66.9% | 35.2% | 0.692 |

The chosen variant's AUC gain is +0.019 (95% CI +0.011 to +0.027; 251 readers
better, 185 worse). More recency weight does not find relevant new papers: the
hit rate for newer right answers stays ~60% in every variant. Two independent
scripts reproduce the reference numbers to the last digit, and the installed
code matches the reference at pool sizes 30, 60 and 100 (top 10 identical for
769 of 770 reader-pools; the one difference is a 1.5e-6 score tie).
**Action items:** Done in `app/recommend/reranker.py` with tests. The negative
penalty is still on the raw cosine scale; measure before rescaling it. The
freshness numbers come from readers built from random corpus papers; re-run
`step2_tradeoff.py` on real ResearchIT readers once impression logs allow.

### 2026-10-14 — Search reranker mxbai-rerank-xsmall; load-aware depth; filler-free FTS; split thread pools
**Decision:** The search cross-encoder is `mixedbread-ai/mxbai-rerank-xsmall-v1`
(70M, Apache-2.0) reading 256 tokens, always returning logits
(`activation_fn=Identity`). It scores the top 20 fused candidates when no other
inference job is queued or running (`SEARCH_RERANK_TOP_N_IDLE`), otherwise the
top 10 (`SEARCH_RERANK_TOP_N`). FTS5 drops filler words before matching and
gives up after `SEARCH_FTS_BUDGET_MS` (1.5 s), leaving the dense arm to answer.
Retrieval for the original query starts as soon as it is encoded instead of
after the Groq rewrite returns. Model inference runs on its own pool sized from
the container's usable CPUs, with torch threads set to match; blocking Qdrant
calls run on a separate io pool. `/healthz/runtime` reports memory, CPUs, pools
and recent stage timings.
**Supersedes:** CLAUDE.md §3.4 "ms-marco-MiniLM-L-6-v2 for the top-10
candidates"; PHASE8 §1 stage 5 and its depth rule, which was measured on MiniLM.
**Rationale:** On 847 known-item queries (LitSearch + synthetic) over the
production sidecar, MRR@10 rose 0.554 → 0.611 at depth 10 and 0.618 at depth 20;
blind LLM-judged nDCG@10 on 160 queries 0.665 → 0.684 / 0.706. Unlike MiniLM,
mxbai improves with depth. It returns probabilities by default, and the
pipeline's own sigmoid on top of them squeezed scores into 0.5–0.73, where the
title/citation boosts took over; that double sigmoid made it measure worse than
MiniLM until fixed. 128 tokens loses most of the gain. int8 dynamic quantisation
cost 1–2% MRR and was 3× slower on Apple silicon; untested on the Space's x86,
not adopted. The FTS filler fix kept MRR and recall while the word-match median
fell 2.0 s → 0.6 s and the max 14.7 s → 1.4 s; production's 20.7 s search
("how do transformers handle long context") was this OR fallback. On 9 of 14
timed production searches "encode" equalled "rewrite" to the millisecond: the
rewrite wait, not the model, was on the critical path. A feed load issues up to
~24 blocking Qdrant calls; on the shared default executor they could occupy
every thread while a search waited with both CPUs idle.
**Action items:** Done in code with tests (`tests/test_search_speed.py`,
`tests/test_fts_sparse.py`). Rollback without a rebuild:
`RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 RERANKER_MAX_LENGTH=512
SEARCH_RERANK_TOP_N_IDLE=10`. After deploy, read `/healthz/runtime` for real
memory, CPU count and stage timings before sizing anything else (a GTE-large
vector file or index). Eval scripts: `scripts/embedding_eval/search/`.

### 2026-10-15 — Correction: reranker gain measured on production text; idle depth back to 10
**Decision:** `SEARCH_RERANK_TOP_N_IDLE` defaults to 10, not 20. The reranker
swap stands.
**Supersedes:** the depth and gain figures in the 2026-10-14 entry above.
**Rationale:** The 2026-10-14 search evaluation gave the reranker full abstracts
(Kaggle CSV, mean 1,171 chars), but production serves the sidecar's
`abstract_preview`, cut at ~500 chars for ~78% of rows. Re-scored on production
text, on the same 847 queries, MiniLM-L6 @10 gives MRR@10 0.523 (production
today) and mxbai-rerank-xsmall @10 gives 0.563: **+0.040 (+7.7%, 95% CI +0.023 to
+0.057)**. Depth 20 adds nothing on truncated text (0.563), so it would cost ~2×
reranker CPU for no gain. With full abstracts mxbai @20 reaches 0.618 (+18% over
production today), so more than half of the available search gain waits on the
abstract backfill (PHASE7 §7.3). The 160-query LLM-judged figures in the entry
above were also scored on full text and overstate today's gain the same way.
**Action items:** Done in `config.py`. Backfill full abstracts (from the local
Kaggle CSV where it covers a paper, arXiv otherwise) into the reranker's input,
then re-run `rerank.py` with and without the `_sc` suffix and raise the idle
depth to 20 if the full-text result holds. Also done: reranker weights pinned to
exact Hugging Face commits (`config.RERANKER_PINS`, mirrored in the Dockerfile
and checked by a test) so a rebuild cannot drift from the evaluated model; the
Qdrant io pool capped at 12 threads, because the free clusters lose throughput
past ~4 concurrent 3-shard searches (8.8/s at 4, 3.8/s at 16, p95 8.6 s). The
local Kaggle CSV holds the full text of every truncated abstract sampled
(2,397/2,397, same opening 200 chars), so the backfill needs no arXiv requests.
BGE-M3 is now pinned too (`config.BGE_M3_PINS`, commit 5617a9f6, the repo's main
since 2024-07-03, so production's weights are unchanged) in the app, the
Dockerfile and the ingest encoder, so query vectors cannot drift from the stored
corpus vectors on a rebuild.

### 2026-10-16 — Word match ranks inside FTS before reading `papers`; feed policy v11
**Decision:** The FTS5 query sorts and limits inside the FTS virtual table and
joins `papers` only for the `limit` winners (`fts_svc._SQL`). The sidecar is read
once into the page cache at startup (`SIDECAR_WARMUP=0` disables it). Feed
`policy_id` / `_RANKER_VERSION` is `v11_rescaled_pubdate`.
**Supersedes:** the join-then-sort query in place since the FTS arm landed, and
the "1.5 s only trips on genuine outliers" premise of the 2026-10-14 budget.
**Rationale:** A read-only flight test of production after the 2026-10-16 deploy
found the lexical arm dropped by the 1.5 s budget on ~40% of searches (19 log
lines, "LLM" on 3 consecutive tries; locally the same query took 83 ms). The old
query fetched the `papers` row of every match before sorting: 56k random reads
into a 1.7 GB table for "LLM", 280k for "learning". On 865 queries (847 eval +
18 probes) the new query returns identical rows in identical order on all 1,509
AND/OR attempts; per-search time (4 processes in parallel on a laptop) median
877 -> 288 ms, max 4.6 -> 0.9 s, over 1.5 s 51 -> 0, total CPU -66%. The
2026-10-14 "same quality with the budget" result was measured where the budget
never fired, so production search quality was below it until this lands. The
feed ordering changed on 2026-10-16 (rescaled similarities, recency 0.45, first
publication age) but kept the v10 label; rows logged from 2026-10-16 19:27 UTC
until this deploy need a time split, not a policy split.
**Action items:** Done. `/healthz/runtime` now reports `timings.fts`
(`and_ms`, `or_ms`, `fts_ms`), `counters.fts_*` (and_hit, or_hit, empty,
budget_skip, error) and `sidecar_warmup` (MB/s of the Space's disk). Re-run the
flight test after deploy; if budget skips persist, the disk itself is the limit.

### 2026-10-19 — Redesign foundations: paper source layer, full abstracts, figure licences
**Decision:** Each paper's arXiv HTML page is read once, in the background, by
`app/paper_source_svc.py` (switch `PAPER_SOURCES`), and figures, sections and
paragraphs are served only from what it saved; no reader-facing request waits on
arXiv. Cut abstracts are replaced by full ones fetched from the arXiv API on first
open (`app/abstract_svc.py`, switch `FULL_ABSTRACTS`). `paper_explanations`,
`paper_abstracts` and `paper_sources` replicate to Turso on an AUTOINCREMENT `id`
watermark and restore in pages. Figures show for every licence (`FIGURE_LICENCES`
defaults to `all`), the owner's decision on 2026-10-19 while ResearchIT is a
personal project; every figure carries a credit and licence line.
**Supersedes:** nothing in the ranking design. The plan's proposal to show figures
only for Creative Commons papers (docs/redesign/02-implementation-plan.md) is
replaced by the owner's choice above.
**Rationale:** Readers need more than a 500-character abstract to understand a
paper, and arXiv publishes HTML for most papers (54 of 60 sampled new papers).
Fetching once and caching keeps arXiv load at one request per paper and keeps the
feed's latency unchanged. A timestamp watermark skips rows in a burst, and one
unpaged SELECT of a large table times out at boot, hence the id watermark and
paged restore. arXiv's default licence grants distribution rights to arXiv only:
narrow `FIGURE_LICENCES` (for example to `CC-BY-4.0,CC-BY-SA-4.0,CC0-1.0`) before
the project is offered more widely.
**Action items:** Done on branch `feat/paper-source-svc` and the branches below
it. Switch `PAPER_SOURCES` to `shadow` first and watch `/healthz/sources` for
three days. Display-only throughout: no ranking, quota, MMR or scoring change.

### 2026-10-19 — Paper briefing: model-written, code-checked, display-only
**Decision:** The redesigned paper page shows a briefing ("Three things to
know", "How it works", "Where you'd use this") written by the configured model
(the `GROQ_MODELS` chain, the owner's choice on 2026-10-19) from the paper's own
passages, behind switch `BRIEFING` (off, shadow, preview, on). Code, not the
prompt, decides what is shown: each point must cite a passage that was sent and
contain only numbers found in it; fewer than two surviving points means no
briefing. Every model call goes through `app/llm.py`, is logged to the
replicated `llm_calls` table, and stops for the day at `LLM_DAILY_CALLS` or
`LLM_DAILY_USD_CAP`. Briefing calls run one at a time, 8 s apart, with prompts
capped at 16,000 characters.
**Supersedes:** "Claude/Groq-generated cluster summaries (Phase 8)" stays out of
scope; this is per-paper text on the paper page, not interest summaries, and it
never touches ranking.
**Rationale:** Readers asked to understand a paper without leaving. Prompt-only
grounding (doc 07 §A.6) can't stop an invented number; checking numbers and
anchors in code can. Measured 2026-10-19: a SatNav briefing took 1.8 s and
passed every check; four large prompts back to back benched every Groq model,
so briefings are paced and a saturated chain is retried rather than cached.
**Action items:** Run `scripts/eval_briefings.py` on `scripts/eval_papers.txt`,
rate the 50 results, and switch `BRIEFING` past shadow only at 45 or more good.

### 2026-10-19 — Ask this paper: model-drafted, code-checked answers
**Decision:** The redesigned paper page has an "Ask this paper" panel behind
switch `ASK` (off, preview, on). Passages are picked without a model (abstract,
named figures and their citing paragraphs, BM25 top 8 in plain Python); the
`GROQ_MODELS` chain drafts JSON; code shows only quotes found word for word in
their cited paragraph (in the paper's own words), only cites that were sent,
and replaces any answer with an unfound number or no valid cite by a
no-support message. Suggested questions are cached per paper with server-side
question text; free-form questions are not cached. Per-reader daily cap
(`ASK_DAILY_CAP`, 30), rate limit, and the shared model budget. Ask writes no
`interactions` row and logs counts only, never question text.
**Supersedes:** the plan's BGE-M3 section retrieval for Ask
(docs/redesign/02-implementation-plan.md, data model): the Space runs one
inference worker, so embedding paragraphs at question time would queue
searches behind it.
**Rationale:** Readers want to question a paper without leaving, and the risk
is a confident wrong answer. Grounding is enforced in code so a wrong or
injected draft fails a check instead of reaching the page.
**Action items:** Run `scripts/eval_ask.py` on 20 papers, rate the answers, and
switch `ASK` past preview only at the gate in docs/redesign/07-pr-8-ask-this-paper.md.

### 2026-10-19 — Two-column feed keeps served order; interactions record their layout
**Decision:** The redesigned feed card (`FEED_CARD_V2`) and a wide feed
(`FEED_WIDE`) with a rail and, from 1180 px, two columns. The two columns show
the feed as ONE list in served order: a CSS grid of 4 px rows with each card
spanning rows to its measured height, `grid-auto-flow: row` and never `dense`,
so no card sits above the card served before it. CSS multi-columns, `dense`
packing and separate column containers are ruled out: each reorders the feed
or breaks infinite scroll. Every save, pass and click from a card records
`layout` (`list` or `grid2`) in `interactions`; feed-control use goes to a
local `ui_events` table.
**Supersedes:** nothing; extends §3.1 ("quota binds the SERVED order") to the
page layout and §3.11 instrumentation with one field.
**Rationale:** Quota decides the order across interests (2026-08-16), and a
masonry layout that fills the left column first would undo it on screen while
the logs still claimed it. In two columns position no longer maps to attention
the same way, so any CTR or position analysis must split by layout or it mixes
two exposure models.
**Action items:** Split evaluation and `scripts/baseline_metrics.py` output by
`layout` once `grid2` rows exist. Keep the card's enrichment local and after
the exposure write. Turn `FEED_WIDE` on only after `FEED_CARD_V2` has been on
for a week.

### 2026-10-20 — Vector figures are served as cleaned SVG
**Decision:** Paper pages and feed cards show SVG figures too: files an
`<object>` points to, and drawings LaTeXML renders inline in the arXiv page.
Each is rebuilt by `app/svg_clean.py` from an allow-list of SVG, HTML and
MathML elements (no scripts, no `on*` attributes, no links or `url(...)` outside
the document, no CSS escapes, no entity declarations), served from `/fig/...svg`
with `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline';
img-src data:; sandbox`, and shown only through `<img>`. Saved figures go from 8
to 24 per paper, and the `/fig` rate limit from 240 to 600 a minute per address.
**Supersedes:** the picture-only figure handling in the 2026-10-19 source-layer
entry, and its cap of 8 figures.
**Rationale:** In 50 papers from September 2026, 118 of 333 figure images were
SVG, 11 papers showed no figure at all, and 27 of 44 were missing at least one.
Rasterising would need a new native dependency (cairo); serving cleaned SVG
needs none, stays sharp in the lightbox, and three layers keep script out.
Some inline drawings are incomplete on arXiv's own page (2201.11903 Figure 2
shows one of its bars); ours match arXiv's page, and the PDF stays linked.
**Action items:** Watch `figure_failed` in `/healthz/runtime` after deploy. If a
class of real figures fails cleaning, widen the allow-list with a test for it,
never by relaxing the attribute checks.

### 2026-10-20 — arXiv access: a readers' lane and a 15-second background lane
**Decision:** Requests to arxiv.org are split by who is waiting. Fetches a
reader triggers (opening a paper, viewing its figures) keep the shared pacer of
one per 2 s in bursts of 3: each replaces what that reader's own browser would
have downloaded from arXiv, and is cached for everyone. Work nobody is waiting
for (the figure warm-up, which now also reads papers people were shown in the
feed and the top 5 search results) additionally waits 15 s after the previous
background request, at most 800 papers a day, and gives way to any reader
fetching the same paper. Search shows a small figure beside each result
(`SEARCH_FIGURES`).
**Supersedes:** the figure warm-up's shared pacer and batch of 1,500
(2026-10-19, two-column feed entry).
**Rationale:** arxiv.org/robots.txt sets `Crawl-delay: 15` for all automated
clients and says indiscriminate automated downloads are not permitted;
info.arxiv.org/help/bulk_data says arXiv's "first priority is to support
interactive use by human users" and points bulk work to S3 and export.arxiv.org
(whose robots.txt disallows everything; its API terms ask one request per 3 s).
Background reading at one request per 15 s stays inside the rule even at full
volume. Thumbnails in results helped people find the wanted result faster
without changing how they scan, in Google's eye-tracking and a 2025 IJHCI study.
**Open risk, owner's call:** arXiv's API terms say e-print content may be
retrieved and stored for personal or research use, but redistributing it needs
the copyright holder's permission. Figures are shown for every licence by the
owner's choice (2026-10-18); the arXiv default licence (about 41% of one day's
cs papers) grants no reuse rights. `FIGURE_LICENCES` restricts figures to
Creative Commons licences without a deploy.
**Action items:** Watch `queued`, `read_today` and `soft_blocks_last_hour` in
`/healthz/sources` after turning `FIGURE_PREWARM` on. Revisit the licence
policy before the audience grows.

### 2026-10-20 — Every model call is measured; caching flaws found
**Decision:** All five AI features log each model call with its tokens, the
answering model's place in the chain, truncation, status and Groq's
rate-limit window (`llm_calls`), plus daily counts of cache use and answer
quality (`ai_daily`), shown at `/healthz/ai`. `LLM_DAILY_CALLS` keeps covering
only the briefing and Ask.
**Supersedes:** nothing; before this only the briefing and Ask logged calls, in
characters.
**Rationale:** Groq's per-model limits are 8,000 tokens a minute and 200,000 a
day. A briefing is about 3,700 tokens and a search about 1,850 (rewrite plus
overview), so two briefings fill a model's minute. The response headers report
requests left today and tokens left this minute, but nothing reports tokens
used today, so only our own count can say how close a day comes.
**Findings to act on** (production, 2026-10-19, one day of data):
1. The briefing cache key includes `PARSER_VERSION` and the configured model.
   The figure-only parser bump of 2026-10-19 invalidated every briefing;
   1706.03762 was regenerated, and 2608.11967 went from ok to insufficient.
   Key on the passages actually sent, and never replace an ok briefing with a
   worse result.
2. Search rewrites and overviews are not cached; an identical search asks the
   model again. Cache the rewrite by normalised query and the overview by
   query plus its top five papers.
3. A plain summary can be written twice, from the cut abstract and then the
   full one (1 of 10 papers).
**Action items:** Build the dashboard on `/healthz/ai`'s data; fix 1 and 2 and
measure the saving in `ai_daily` cache counts.

### 2026-10-20 — Reader and search analytics, with search queries kept 90 days and unlinked
**Decision:** Searches are logged (`search_log`), each AI call records its paper
or search query, and each reader cookie gets one row per day of what it did
(`reader_days`), all replicated and shown on owner-only dashboards. By the
owner's choice, search query text is kept 90 days and then cleared here and in
Turso. It is never stored against a reader; a search links to the clicks and
saves on its results only through its `query_id`, as before. The owner's own
browsers (holding the preview token) are flagged and left out by default. The
AI overview shown for a query over its top five papers is stored, which also
makes it the overview cache.
**Supersedes:** nothing; searches, visits and questions were not recorded before.
**Rationale:** Of 476 recorded actions since 2026-08-09, 321 of 335 saves came
from search results, yet no search had ever been recorded. 109 of 131 readers did
exactly one thing and 7 came back on another day; October's three active
readers were the owner. Without searches, visits and an owner filter, none of
that could be measured going forward.
**Action items:** Watch the share of searches that find nothing and that lead
to an opened or saved paper. Measure the readers who return within seven days
before and after each product change.

