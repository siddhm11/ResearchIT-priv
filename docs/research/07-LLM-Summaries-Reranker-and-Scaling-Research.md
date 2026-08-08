# ResearchIT Phase 4 Implementation Plan and Phase 5 Preview — Research Report for Amin

This report synthesizes 2024–2026 sources (RecSys/SIGIR/KDD/NeurIPS/ACL/EMNLP papers, production blogs from Pinterest, Spotify, YouTube, Netflix, and documentation from BAAI, Jina, Mixedbread, Anthropic) into an implementation-ready plan. The headline recommendation is to run Phase 4a (Claude summaries) and 4d (use-cases doc) in parallel over weeks 1–3 after a one-week ADR sprint, then spend weeks 4–9 on 4b (distilled reranker) — total ~10–12 weeks for Phase 4 with buffer. Nearly every Phase 5 workstream (exploration, IPS, propensity logging, telemetry schema) must be architected *before* Phase 4 code lands, even though the workstreams themselves are gated on user-count thresholds. The single most valuable decision to make now is the telemetry event schema, because retrofitting propensity, policy-id, and position fields after you have real-user data is painful and blocks all later counterfactual evaluation.

## A. Phase 4a — Claude-API-generated per-cluster interest summaries

### A.1 Prompt engineering

The closest published analogue to Amin's use case is **Scholar Inbox** (Flicke et al., ACL 2025 Demo, arXiv 2504.08385), which generates 4-level hierarchical labels (field → subfield → subsubfield → method) from t-SNE paper clusters using Qwen; their appendix §6.1 contains the exact prompt. Microsoft's **TnT-LLM** (KDD 2024) and **TopicGPT** (Pham et al., NAACL 2024) converge on the same pattern: structured XML-tagged inputs, constrained vocabulary, and JSON output. The recommended template for ResearchIT:

```
You are summarizing a research interest cluster for a specific user.

USER PROFILE CONTEXT (tone only, not content):
{short profile string}

CLUSTER MEDOID PAPER (most representative):
<medoid><title>{...}</title><abstract>{...}</abstract></medoid>

NEAREST NEIGHBOR PAPERS:
<papers>
<paper id="1"><title>...</title><abstract>...</abstract></paper>
... (up to 20)
</papers>

TASK: Produce JSON {"label": "<1-sentence 'You're reading about X, particularly Y' framing>", "themes": [<≤5-word bullet>, ... up to 4]}

RULES:
- Every technical term in "label" and "themes" MUST appear verbatim in at least one provided title or abstract.
- Do NOT introduce methods, datasets, or concepts not present in inputs.
- If fewer than 3 papers share a theme, omit it.
- Prefer specific phrases ("retrieval-augmented generation evaluation") over generic ones ("NLP research").
- Output JSON only.
```

**Start zero-shot with this constrained prompt; add 2–3 hand-written few-shot examples only to anchor the "You're reading about X" voice.** Spotify Research's Dec 2024 "Contextualized Recommendations Through Personalized Narratives Using LLMs" post found zero-shot adequate but converged on 3–5 "golden" style examples for tone. The Anthropic cookbook's `using_citations.ipynb` demonstrates the **Citations API**, which returns structured citation objects and explicitly "will not return citations pointing to documents or locations that were not provided as valid sources" — **use the Citations API for ResearchIT**, it eliminates the hallucination vector at the API level.

### A.2 Regeneration frequency

The 2024–2026 literature (Google's arXiv 2510.20260 on "Balancing Fine-tuning and RAG for Dynamic LLM Recommendation Updates"; Spotify's production narratives cache per-item) strongly favors **event-triggered regeneration over fixed nightly cadence**. Concrete hybrid policy:

Regenerate when the medoid paper changes, when Jaccard distance between old and new paper-ID sets exceeds 0.3, or when a cluster is added/merged/split. Apply a **7-day TTL fallback** even when nothing changes (captures embedding/context drift). **Do not regenerate nightly** — it is roughly 7× the cost for negligible UX gain on Ward clusters whose membership is stable over the timescale of a single day.

### A.3 Pricing (April 2026) and cost estimate

Verified current pricing from platform.claude.com/docs, cross-checked against Finout/MetaCTO/PE Collective reporting: **Haiku 4.5 at $1/$5 per MTok in/out**, **Sonnet 4.6 at $3/$15**, **Opus 4.7 at $5/$25** (released April 16, 2026, with a new tokenizer that can inflate token counts up to 35%). Cache reads are 10% of base input; cache writes 125% (5-minute) or 200% (1-hour). Batch API gives a flat 50% discount with ≤24h turnaround and stacks with caching. Haiku 3 is deprecated April 19, 2026 — do not build against it.

For 1,000 users × 5 clusters × 20-paper contexts (~6,000 input tokens each) regenerated weekly, monthly traffic is ~130M input + ~3.25M output tokens. Total monthly cost by model:

- **Haiku 4.5 + Batch API: ~$73/month; with prompt caching on stable prefix, ~$50–60/month**
- Sonnet 4.6 + Batch API: ~$220/month (~$150–180 with caching)
- Opus 4.6/4.7 + Batch API: ~$366/month (~$280 with caching)

**Recommendation: Haiku 4.5 + Batch API is the right default.** The task (label a cluster from provided abstracts) sits comfortably within Haiku's capability. Reserve Sonnet for offline A/B quality evaluation on a minority of calls. Skip Opus entirely for this task. Prompt caching savings are modest because each cluster's paper context is unique per cluster; the real economic lever is the **shared cross-user dedup** (§A.7), not prompt caching within a single call.

### A.4 Content-addressed caching

Construct the cache key as `sha256(sorted(paper_ids) + prompt_version + model + schema_version)`. Sort paper IDs before hashing for order-independence; include prompt and model version so stale summaries don't survive a template change; **omit user ID** from the shared cache key (that's the entire point — §A.7). Use an immutable, content-addressed store (`summaries[hash] = {label, themes, generated_at, model, tokens_used}`) — never overwrite; let old entries age out on a 90-day LRU. This mirrors CDN asset hashing (`main.a3f2b1c9.js`) and matches the Anthropic Claude Code cache-invalidation discussion (issue #29230) recommending SHA-256 of all source files be part of the cache key.

Expected exact 20-paper dedup rate is low (papers are drawn from 3M+ arXiv), but a **two-tier cache** with a "narrow" key (medoid + top-5 neighbors) as fallback increases hit rate substantially.

### A.5 Explainable-recommender UX in academic search

None of Scholar Inbox, Connected Papers, Elicit, ResearchRabbit, Semantic Scholar, Consensus, or Undermind currently displays a **personalized "You're reading about X" per-user cluster narrative**. Scholar Inbox's Scholar Map labels are the closest analogue but are global/shared across users. This means ResearchIT's Phase 4a is **genuinely novel UX for academic search**, and the right place to borrow heavily is Spotify (which reports up to 4× CTR on niche content when LLM narratives personalize discovery) and Wang et al.'s "LLMs for User Interest Exploration in Large-scale Recommendation Systems" (RecSys 2024, arXiv 2405.16363), an architecturally identical recipe (interest clusters + constrained LLM descriptions). Lubos et al.'s UMAP 2024 user study on "LLM-generated Explanations for Recommender Systems" confirms users rate LLM explanations highly for decision support.

UX recommendation: lead with the 1-sentence "You're reading about X, particularly Y" framing, then an expandable bullet list of 3–5 sub-themes, with **source paper titles as linkable chips** under each bullet (the Anthropic Citations / deterministic-quoting pattern, which kills trust issues by letting users verify). A subtle "regenerated on {date}" timestamp plus a manual refresh button gives users control.

### A.6 Hallucination prevention

The 2024–2026 state-of-the-art for grounding evaluation is **MiniCheck** (Tang, Laban, Durrett, EMNLP 2024, arXiv 2404.10774) — a 770M-parameter fine-tuned Flan-T5 that matches GPT-4 fact-checking accuracy at ~400× lower cost. Ranked strongest-to-weakest, grounding techniques are: (1) deterministic quoting (surface verbatim source text in the UI); (2) **Anthropic Citations API** (native, recommended); (3) prompt-based "use only phrases from source" rules; (4) post-hoc NLI verification with MiniCheck-FT5; (5) constrained decoding (overkill for 1-sentence labels).

Recommended stack: Anthropic Citations API + explicit "verbatim-phrase" rule in prompt + post-hoc substring verification on noun phrases (reject and regenerate if >1 unsupported phrase). Run MiniCheck-FT5 offline on a sample as an ongoing faithfulness metric. Zhou et al. (Findings EMNLP 2023) "context-faithful prompting" shows instruction-only grounding measurably reduces hallucination but is not sufficient alone — combine with a verification layer.

### A.7 Per-user vs shared summaries

**Use a hybrid two-stage design.** Stage 1 generates a **shared, content-addressed, public-paper-only** cluster description (the Claude call gets only paper titles/abstracts, never user profile text) — identical cluster content produces identical summary across users and days, enabling aggressive dedup. Stage 2 wraps the shared summary with per-user framing either via client-side string templating ("You're reading about {shared_label}") or via a lightweight per-user LLM pass cached at `(user_id, shared_hash)`.

This matches Spotify's item-level-narrative + per-user-context split and Google's arXiv 2510.20260 offline-bulk/online-lookup separation. **Privacy payoff:** shared summaries are pure functions of public arXiv content, so they can ride Anthropic's Batch API with ZDR safely, be logged freely, and be cached cross-user. User profile text never leaves your infrastructure (or does so only in a heavily-filtered form for Stage 2). This is the architectural decision (ADR A2) that must be made **before** building the caching layer, because switching from per-user to shared requires a full cache-schema migration post-launch.

## B. Phase 4b — Distilled cross-encoder reranker

### B.1 FlashRank recipe and student candidates

**FlashRank (PrithivirajDamodaran) does not train its own students** — it repackages existing open checkpoints as quantized ONNX. The default "Nano" is `ms-marco-TinyBERT-L-2-v2` (14M params, ~17MB fp32, ~6MB INT8), "Small" is `ms-marco-MiniLM-L-12-v2`, and "Medium" is `rank-T5-flan`. The engineering pattern to steal is ONNX + INT8 dynamic quantization + the `tokenizers` Rust library only (no PyTorch/transformers at runtime), keeping cold-start under 500ms on serverless.

For Amin's 6ms-for-20-pairs CPU budget (≈0.3ms/pair), **the only candidates that fit with headroom are 2-layer students**:

| Model | Params | INT8 CPU latency/pair | BEIR nDCG@10 |
|---|---|---|---|
| **ms-marco-TinyBERT-L-2-v2** | 14M | ~0.3–1.0ms | ~43–45 |
| ms-marco-MiniLM-L-4-v2 | 19M | ~1.5–2ms | ~46 |
| ms-marco-MiniLM-L-6-v2 | 22M | ~3–5ms (tight on budget) | ~48 |
| jina-reranker-v1-turbo-en | 38M | ~3–5ms | 49.60 (95% of jina-base) |
| jina-reranker-v1-tiny-en | 33M | ~2–3ms | 48.54 (92.5%) |
| mxbai-rerank-xsmall-v1 | 71M | ~8–12ms (over budget) | 43.9 |

Tonellotto et al.'s "Shallow Cross-Encoders" (SIGIR 2024, arXiv 2403.20222) found that at latency ≤10ms on CPU, TinyBERT-gBCE reaches nDCG@10 of 0.652 on TREC-DL-2019, a +51% gain over MonoBERT-Large (0.431). **The architectural choice (2L vs 12L) matters more than the teacher weights at tight latency.** Don't pick a bigger student.

### B.2 Domain adaptation — how much does arXiv-specific fine-tuning buy?

**Typical gain from in-domain distillation at the 2-layer scale: +1 to +3 nDCG@10 points on SciDocs**, not 10. MedCPT (PubMed, Jin et al. arXiv 2307.00589) surpasses BM25 only after ~150M query-article pairs, showing diminishing returns for modest training budgets. The listwise-distillation paper arXiv 2505.19274 demonstrates that a general RankT5-3B teacher is competitive with in-domain rerankers on SciDocs/SciFact/NFCorpus, within noise. **No BGE-reranker-v2 checkpoint fine-tuned on scientific text exists on Hugging Face as of April 2026** (searched).

### B.3 Distillation objectives

The 2025 reproducibility study (arXiv 2603.03010) benchmarks nine loss functions across nine backbones with SPLADE-v3 top-1000 candidates. Average rank across out-of-domain BEIR:

1. InfoNCE (rank 1.83)
2. **MarginMSE** (2.17) — Hofstätter-style pairwise distillation
3. DistillRankNet (3.61)
4. ADR-MSE (3.66)
5. Hinge (3.99)
6. BCE (5.74) — significantly worse than every other

Critically, "**MarginMSE with BM25-mined negatives is statistically equivalent to InfoNCE with ColBERTv2 hard negatives**" — loss formulation matters more than negative-pool quality. BAAI/BGE uses MarginMSE + self-knowledge-distillation from ensembles. Jina uses explicit KL on logits from the full-size teacher. Yang, He, Yang's SIGIR 2024 paper proposes CKL (contrastively-weighted KL) outperforming MarginMSE+plain KL on MS MARCO + BEIR zero-shot, but the gap is small.

**Recommended loss:** `L = α·MarginMSE(student, teacher, pos, neg) + β·KL(σ(student/T), σ(teacher/T)) + γ·BCE(pos, 1)` with α=1.0, β=0.5, γ=0.1, T=1.0. MarginMSE alone is a fine MVP.

### B.4 Integration architecture

Three options: (A) TinyBERT score as one feature in a second LightGBM pass; (B) TinyBERT as a direct re-ranker on top-20 replacing LightGBM at that stage; (C) two-stage LightGBM with TinyBERT in between. Bing's LambdaMART over hundreds of features (including BERT scores), Pinterest's TransActV2 feeding neural scores into GBDT, Google/DeepMind's DASALC+TFR-BERT, and TREC TOT 2025 (arXiv 2601.15518) all converge on **the neural score as one feature among many in a final LambdaMART**, not as a terminal reranker.

**Recommendation: Option C (≈Option A).** Keep the upstream LightGBM-lambdarank, score the top-20 with TinyBERT (~0.3ms/pair × 20 = ~6ms), and feed the student scores back into a second LightGBM pass that has access to the full personalization feature set. **Do not do Option B** — replacing LightGBM with TinyBERT at top-20 throws away user features, citation-graph features, and temporal decay that LightGBM already incorporates. Engineered features for LightGBM-2: `tinybert_score`, `tinybert_rank_position`, `tinybert_score_normalized_within_query`, and the interaction `tinybert_score − bm25_score`.

### B.5 Hard negative mining

The 2024 standard is **NV-Retriever** (arXiv 2407.15831) "positive-aware" filtering: mine top-100 ANN neighbors, then filter with the teacher cross-encoder, dropping candidates whose teacher score is within 0.3 of the positive (likely false negatives or duplicates). For academic papers, supplement with SPECTER/SciNCL citation-graph negatives: **SPECTER** uses 2 "citation-of-citation" hard negatives per query; **SciNCL** (Ostendorff et al.) improves on this by sampling from a continuous citation embedding space (PyTorch-BigGraph over S2ORC) with controlled distance margins (k_min=3998, k_max=4000 on a 52M-node graph), delivering +1.8 points on SciDocs. Recommended mix per (seed, positive): 3 SciNCL-style citation-of-citation negatives, 5 teacher-filtered ANN negatives (top 10–100 with teacher score below 95th percentile), 2 random in-batch. Critically, re-score all candidates with BGE-reranker-v2-m3 and **drop any within 0.3 teacher-score of the positive**.

### B.6 Evaluation and distillation quality gap

Typical retention rates from the 2024–2025 literature: jina-v1-base → jina-v1-turbo retains 95% (52.45 → 49.60); TinyBERT-4L retains ~96.8% of BERT-base on GLUE; MiniLM-L6 → MiniLM-L2 rerank retains ~85–90%. **For Amin's ~20× compression from BGE-reranker-v2-m3 (278M) → TinyBERT-L2 (14M), expect 82–88% retention of nDCG@10.** If below 80%, something is wrong (bad negatives, insufficient data, teacher-label leakage into eval).

Run evaluations on SciDocs (focusing on Co-view / Co-read / Cite / Co-cite tasks), SciRepEval proximity tasks, the BEIR scientific subset (NFCorpus, SciDocs, SciFact, TREC-COVID), and held-out unarXive 2024–2026 queries with citation-graph ground truth. **CPU latency protocol: 50 warmup inferences discarded, 1000 measured inferences at seq_len=128, batch=20; report P50/P95/P99, not mean** (Pinterest standard).

### B.7 Off-the-shelf scientific-domain rerankers

**There is no well-maintained small (<50M param) scientific-domain cross-encoder reranker on Hugging Face as of April 2026 that beats MS MARCO-trained TinyBERT on SciDocs at the 6ms budget.** SPECTER/SPECTER2/SciNCL are bi-encoders (embedders), not rerankers. MedCPT is biomedical-specific. Third-party SciBERT cross-encoders exist but are not validated at MS-MARCO MiniLM-L6 quality. No BAAI bge-reranker fine-tuned on scientific corpus published.

**Decision tree:**

- **If Amin already has a pseudo-label pipeline producing >200K (query, doc, teacher_score) triples** → distill TinyBERT-L-2 from bge-reranker-v2-m3 on arXiv data. Expect +1–3 nDCG over off-the-shelf.
- **If Amin wants MVP now** → deploy `cross-encoder/ms-marco-TinyBERT-L-2-v2` with INT8 ONNX (HF already ships `onnx/model_qint8_avx512_vnni.onnx`), measure on held-out eval. If gap vs teacher is <3 nDCG@10, ship; distill later if needed.

**Strong recommendation: go off-the-shelf first.** Distillation is ~2–4 weeks of solo-dev work and the marginal gain at 2-layer scale is usually small. Time is better spent on hard-negative mining and LightGBM-2 feature engineering.

### B.8 ONNX / FastAPI hot path

Latency ranking for BERT-base-class inference on x86 with AVX-512 VNNI:

- PyTorch eager fp32: baseline (1.0×)
- PyTorch INT8 dynamic CPU: 0.4×
- ONNX Runtime fp32: 0.3×
- **ONNX Runtime + INT8 dynamic AVX-512 VNNI: 0.15–0.25× (up to 6× over ORT fp32)**
- torch.compile: 1.5–2× over eager but still behind ONNX on CPU

For TinyBERT-L-2-v2 on Render's standard ~2 vCPU x86: fp32 PyTorch seq=128 ≈3–5ms/pair; **INT8 ONNX ≈0.3–1.0ms/pair single-thread; batched 20 pairs ≈2–4ms total wall-clock on AVX-512 VNNI hardware** (2–3× slower without VNNI). Production code pattern:

```python
import onnxruntime as ort
from tokenizers import Tokenizer

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2  # match Render vCPUs
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

session = ort.InferenceSession(
    "model_qint8_avx512_vnni.onnx", sess_options, providers=["CPUExecutionProvider"]
)
tokenizer = Tokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
tokenizer.enable_truncation(max_length=128)
tokenizer.enable_padding(length=128)

def score_pairs(query, docs):
    enc = tokenizer.encode_batch([(query, d) for d in docs])
    return session.run(None, {
        "input_ids":      np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
    })[0].squeeze(-1).tolist()
```

Critical tips: pin padding length to enable kernel fusion; use `tokenizers` (Rust, ~0.1ms for 20 pairs) not `transformers.AutoTokenizer` (~5ms); cache sessions globally; disable thread spinning; skip QAT (dynamic INT8 costs <0.5 nDCG).

### B.9 Latency scaling top-20 → top-50 → top-100

Linearity is approximately valid but with caveats. K=20→50 ≈ 2.5× latency (6ms → 15ms) with modest sub-linear batching gains of 5–10% from amortized Python/tokenization overhead. K=100 ≈ 4.5× rather than 5×. Memory pressure kicks in at K≥64 with seq_len=512 but not at seq_len=128. Render's 2-vCPU boxes saturate at intra_op_num_threads=2.

| K | Strategy | Expected latency |
|---|---|---|
| 20 | single batch of 20 | 2–6ms |
| 50 | single batch of 50 | 6–15ms |
| 100 | 2 batches of 50, pipelined | 12–25ms |
| 200 | upgrade to MiniLM-L-4 or go async | 30–50ms |

**Beyond K=50, the right move is NOT to batch harder but to prune harder upstream** — make LightGBM-1 more selective. Pinterest and Bing aggressively trim before the expensive stage.

## C. Phase 4d — Use-cases and information-gain design doc

### C.1 User personas

Foundational literature: Bates's "berrypicking" (Online Review 1989) — real scholarly search is iterative, multi-source, goal-mutating, not one-shot. Ellis/Wilson's six activities (starting, chaining, browsing, differentiating, monitoring, extracting) map cleanly: monitoring = stay-current mode; chaining+differentiating+extracting = literature-review mode. Al-Shboul & Abrizah (2014, Journal of Academic Librarianship) is the explicit persona-template reference. Gordon et al. (Taylor & Francis 2020/2021) quantify scholarly pain: only 15.4% of physicists feel successful at staying current; 28.6% feel unsuccessful. Mysore et al. (CHIIR 2023) and Soufan, Ruthven, Azzopardi (CHIIR 2024) empirically confirm berrypicking in modern AI/ML workflows. Niwanputri et al. (SIGIR ICTIR 2025) "Untangling Cognitive Processes in Academic Information Searching" is the 2025 SIGIR anchor. **Scholar Inbox (Flicke et al. 2025, arXiv 2504.08385)** is the closest comparable system — they released an 800k-rating dataset and use an active-learning rating onboarding pattern.

Drop-in persona cards for the doc:

| # | Persona | Profile state | Mode | Day-1 signal | UX demand |
|---|---|---|---|---|---|
| P1 | Brand-new (cold start) | Empty EWMA | Exploration-forced | Categories + 5–10 ratings | Active-learning onboarding (Scholar Inbox) |
| P2 | PhD student, active | 50–500 interactions, 2–4 tight clusters | Stay-current/deep | Daily skim, narrow topic | Don't flood with diversity early |
| P3 | Senior researcher/PI | 1k+ interactions, 8–15 clusters | Mixed monitoring | Scan many, save few, dismiss often | No single cluster >40% |
| P4 | Cross-disciplinary | Multiple distinct medoids | Parallel stay-current | Per-cluster cadence diverges | Cluster-balanced delivery |
| P5 | Lapsed (3-mo gap) | α_long preserved, α_short stale | Re-orient | High dismissal first 3 sessions | "What changed" framing |
| P6 | Cold-restart pivot | Has history, wants new field | Explicit pivot | System seeds new cluster | "Start new interest" UI |
| P7 | Literature-review session | Any profile + deep-session intent | Deep single-cluster | Many click-throughs, long dwells | Suppress MMR, amplify depth |
| P8 | Stay-current daily | Any profile, 10-min daily | Monitoring | Fast skim, binary save/dismiss | Strong MMR, proportional cluster coverage |

### C.2 Information gain per interaction

Foundational: Joachims (KDD 2002) clicks as relative pairwise preferences; Joachims et al. (TOIS 2007) eye-tracking validates ~80% reliability for "click i, skip i−1" pairs; Yi et al. (RecSys 2014) dwell time ≥30s as valid-engagement threshold; Xie et al. (WWW 2023) "valid read" = click + sufficient dwell; Yin et al. (WSDM 2013) "Silence is also evidence" — short dwell after click is negative, not missing. **The central paper** is Wang et al. RecSys 2023 (arXiv 2308.12256): dislike as feature only → −0.34% dislike rate (not significant); dislike as **feature AND training label** → −2.44% dislike rate, **−9.60% repeated dislike on same creator**, −2.05% dismissing users, and counterfactually **60.8% reduction in similar-content recommendations versus 22% when dislike is feature-only**. Implicit skip as negative label delivered +0.40% user enjoyment, +0.61% DAU≥1h.

Drop-in information-gain table (normalized to click = 1.0 baseline):

| Interaction | Sign | Relative strength | EWMA update | ~Bits info |
|---|---|---|---|---|
| Explicit category at onboarding | + | 5–10× | α_long seed | 3–5 |
| Save / bookmark | + | 3–5× | α_short + α_long | ~2 |
| Click-through to arXiv (no dwell) | + | 1.0× | α_short | ~0.5 |
| Long dwell (>30s) on abstract | + | 2–3× | α_short elevated | ~1 |
| Short dwell (<5s) after click | − weak | −0.5× | small α_neg | ~0.3 |
| Share / export to bib | + | 4–6× | α_short + α_long strong | ~2–3 |
| Dismiss (feature only) | − | −1× | Layer-1 only | ~0.3 |
| **Dismiss (feature + training label + similar suppression)** | − | **−3× to −4×** | All three layers | ~1.5–2 |
| "Don't recommend cluster" mute | − | −10× | Hard filter persistent | 3+ |
| Passive skip / scroll-past | − very weak | −0.1× | Aggregate only | ~0.05 |
| Revisit saved paper | + | 2× | α_long | ~1 |

**Product principles derived:** every save must move the EWMA profile measurably (if α_short=0.40 doesn't produce a visible medoid shift after one save, the profile is broken); dismissals must be 1-click because their information value is ~3× passive skip; dwell must be normalized per device/context; explicit negatives must enter both the LightGBM feature vector AND the training label — feature-only is essentially wasted.

### C.3 Longitudinal journeys

Time-drift literature (Koren KDD 2009 timeSVD++; Mansoury CIKM 2020 feedback loop; TDLRP-MF MDPI Systems 2025; TransActV2 arXiv 2506.02267) validates Amin's α_short/α_long split. The temporal-drift papers consistently show α_short ≈ 10× α_long is healthy; Amin's 13× ratio is in range. Per-persona day-1/7/30/90 table: P1 progresses from explicit ratings + popularity-biased exploration to 1–2 tight clusters by week, to 2–4 stable medoids at 30 days, to indistinguishable from P2 at 90 days. P5 on return at d=90 starts with stale α_short; decay α_long by (1−α_long)^90 ≈ 0.065 to partially refresh. P7 is session-scoped only (MMR λ down, cluster depth up, session-TTL long). P8 is steady monitoring at 10-min daily, evolving slowly in α_long regime.

### C.4 Instrumentation priorities

Production references: Spotify Event Delivery Infrastructure (8M events/s, schema-first, session-context qualifies every signal); Pinterest TransActV2 (arXiv 2506.02267, real-time top-100 sequence, **p99 latency as production-critical metric not mean**); YouTube Covington RecSys 2016 + Wang 2023 (80B signals/day, separate logging for watch/search/subscribe/dismiss/satisfaction); OpenTelemetry Weaver (2025) for schema-first telemetry with SDK generation. The schema must be frozen before any real-user logging (ADR A4) because post-launch migrations are painful.

Minimum event families to log: session_start/end + mode_declared; feed_request/served with slot_index, cluster_id, medoid_id, popularity_prior_weight, mmr_lambda, exploration_flag; positive (click, dwell_end, save, bookmark, share, export_bib, revisit) with dwell_ms, scroll_depth, device_context; negative (dismiss, mute_cluster, hide_author, explicit_dislike) with reason_code, layer_applied; profile ops (ewma_step, cluster_rebuild, medoid_shift) with α_used, silhouette_delta; model ops with per-stage latency; health/error events (empty_candidate_set, stale_profile_warning, popularity_fallback_triggered). **Log p50/p95/p99 latency percentiles per stage.** Nightly aggregations for SLO dashboards: personalized-to-popularity ratio (target ≥0.85 after day 7), cluster-share Gini (alert >0.7), exploration-slot fire rate (target 1/10 ±50% drift alert), per-cluster dismiss rate (>35% → mute candidate), save-to-click ratio, α_short day-over-day distance (alert if zero for 14 days), time-between-sessions (detects lapsed users).

### C.5 Product principles

Netflix North-Star thinking (Gibson Biddle) suggests **"saves per active week"** as ResearchIT's primary leading indicator — tied to customer value, directly moves α_long, not gameable by dismissals. Spotify contextual-session principle: a skip in stay-current mode ≠ a skip in lit-review mode. Pinterest tail-latency principle: operational metrics on p99 not mean. Stated principles for ResearchIT: every save must measurably move the profile; dismissals are always 1-click, always logged as both feature and label; three-layer negatives have distinct half-lives (session/α_neg=0.15/persistent-until-unmuted); context qualifies every signal; exploration is a budget not an afterthought; cluster balance beats global top-K for cross-disciplinary users; cold-start is active not passive (Scholar Inbox pattern); latency SLOs on p99; stale profiles must announce themselves; never dark-launch a ranker change without a popularity-baseline A/B.

### C.6 Mode-switching / intent-conditioned recommendation

Broder (SIGIR Forum 2002) navigational/informational/transactional extends to informational-narrow (lit-review) vs informational-broad (stay-current). **Jannach, Mobasher et al. TORS 2024 (arXiv 2406.16350) "A Survey on Intent-Aware Recommender Systems"** is the 2024 anchor — categorizes diversification-based, intent-prediction, and latent-intent modeling; identifies gap of offline-only evaluation. RecSys 2024 reproducibility study "A Worrying Reproducibility Study of Intent-Aware Recommendation Models" is cautionary: most intent-aware claims don't replicate. **Industry validates explicit mode switching over fully-latent intent** (Pinterest Homefeed vs Related-Pins vs Search; Spotify Deep-Focus vs What's-New).

Recommendation: start with an **explicit two-mode toggle** in UI ("Stay Current" / "Lit Review"): stay-current has high MMR λ, per-cluster quota on, small popularity prior, 10-min session TTL; lit-review has low MMR λ, high single-cluster depth, citation-chain exposure, 60-min session TTL. Add latent intent fallback: if session shows 3 consecutive clicks into one cluster with long dwells, quietly switch to lit-review. Defer sophisticated latent-intent models.

### C.7 Failure modes and detection

Chaney, Stewart, Engelhardt (RecSys 2018) prove feedback loops amplify homogeneity; Mansoury et al. (CIKM 2020) quantify bubble intensification across rounds; Nguyen et al. (WWW 2014) first longitudinal filter-bubble measurement; Tang et al. (arXiv 2508.11239, Aug 2025) "Mitigating Filter Bubble from Community Detection" defines filter-bubble index = fraction of recs inside user's own community — **directly operationalizable using Ward clusters as the Louvain analog**. Drop-in detection rules:

| Failure | Detection rule | Mitigation |
|---|---|---|
| Feed collapse | 7-day rolling cluster-share Gini >0.7 OR top-cluster share >0.6 | Force MMR λ up; inject exploration; cap per-cluster at 40% |
| Stale profile | α_short unchanged for 14 days AND last session >30 days | "Refresh interests" card; boost popularity prior; Scholar-Inbox-style re-prompt |
| Cluster fragmentation | Cluster count >K_max OR >40% clusters with <3 neighbors | Lower Ward threshold; merge |
| Cluster over-merging | Silhouette week-over-week Δ <−0.15 | Raise Ward threshold; split top-variance cluster |
| Filter bubble | Filter-bubble index >0.95 for 30 days | Cross-cluster sampling; raise exploration budget |
| Popularity collapse | popularity_fallback >0.2 DAU/day | Ranker may be broken; verify LightGBM not degenerate |
| Latency regression | p99 > SLO for 1h | Standard SRE playbook |
| Dismissal ineffective | In-cluster rec rate within 7 days of dismiss > baseline | Verify three-layer pipeline; check layer-2 re-training |
| Feedback-loop amplification | Avg pairwise served-item similarity trending up 4+ weeks | CD-CGCN community-aware negative sampling |
| Cold-start stuck | Personalized score share <0.3 at day 7 | Push active-learning prompts; lower warm threshold |

## D. Phase 5 preview at Phase-4-level detail

### D.1 Epsilon-greedy exploration

**Spotify BaRT** (McInerney, Lacker, Hansen, Higley, Bouchard, Gruson, Mehrotra; RecSys 2018; DOI 10.1145/3240323.3240354) is the canonical reference. Two-stage contextual bandit over Home shelves (rows + explanations) and cards (playlists). Reward = factorization machine over user × item × explanation × context features predicting a binary stream event (≥30s listen). Epsilon-greedy per-slot: with probability ε pick uniformly among candidates, otherwise argmax. Conditional exploration separates "explore the item" from "explore the explanation" sharing one reward model — this keeps propensities tractable. Training uses counterfactual risk minimization with IPS on logs. Heavier exploration for new users, lighter for established.

**Pinterest "Warmer for Less"** (arXiv 2512.17277, Dec 2025) targets industrial cold-start items: **targeted lightweight augmentations (~+5% params) to the main model can match heavier bespoke approaches**. Strongly validates leaning on BGE-M3 content embeddings + light corrections for new arXiv papers rather than a separate CF/graph cold-start pipeline.

Literature consensus on exploration budget clusters at **5–15%, with 10% as default**. For ResearchIT:

- **Pre-launch → 100 users: ε-greedy at ε=0.10, slot-reservation pattern** (reserve 1/10 feed slots for exploration candidates — cleaner and lower-variance than per-slot coin flips).
- **100–500 users: stratified exploration** (ε distributed over arXiv primary categories the user hasn't engaged with × medoid-to-item cosine uncertainty).
- **500–1K users, >1K eng/week: Beta-Bernoulli Thompson sampling at category level.**
- **>5K users, >10K eng/week: neural-linear bandit (mtNLB-style, KDD 2024 DOI 10.1145/3637528.3671649) reusing LightGBM scorer as representation — only if ε-greedy shows regret plateau.**

Thompson vs ε-greedy: Chapelle & Li (NeurIPS 2011) and Vermorel & Mohri (2005) show vanilla ε-greedy routinely matches or beats TS/UCB at small N. TS at item level across 1.6M items with <1K users is infeasible; TS at category or cluster level is tractable. Other contextual bandit references: LinUCB (Li, Chu, Langford, Schapire WWW 2010); NeuralUCB (Zhou ICML 2020); NeUClust (Atalar et al. arXiv 2410.14586, Oct 2024) — contextual-combinatorial for list recommendations; ENR (CIKM 2023) epistemic neural nets for scalable TS; Ban/Qi/He WebConf 2024 tutorial.

### D.2 LightFM collaborative filtering

LightFM (Kula 2015, arXiv 1507.08439) is legacy-but-still-competitive; in 2026 it remains perfect for Render's CPU-only deployment because every user/item embedding is a sum of feature embeddings (including a unique-ID feature), enabling **strong cold-start with metadata — exactly ResearchIT's setting**. Alternatives: implicit ALS (industrial baseline but no cold-items); LightGCN (SIGIR 2020 arXiv 2002.02126, ~16% avg lift on standard datasets but training overhead); two-tower (Google, needs GPU); UltraGCN (marginal gains). 500-user rule-of-thumb: LightFM with WARP loss crosses above content-only when users×interactions >5K; at 500 users × ~10 positive interactions = ~5K, exactly threshold.

**Integration: Pattern 2 (CF score as a LightGBM feature).** Spotify and Pinterest production consistently run CF + content-based candidate generators in parallel with a learned ranker blending them; within the ranker, CF is one feature among many. This gracefully handles users with weak CF signal because LightGBM learns to down-weight it. Don't do separate quota slots (worst at blending score scales). Warm-start uses LightFM's feature-averaging: a new user with claimed research categories/authors gets a warm embedding without any interaction history.

### D.3 Dismissal-labeled LightGBM retraining

**Minimum viable signal: ~1K dismissal events total** to distinguish systematic item-level dismissals from session noise. **For LightGBM retraining with dismissals as labels: ~10K events.** At 500 users × 5% dismissal rate × 50 impressions/week = ~125 dismissals/week → ~10K takes ~80 weeks of steady use. **Action: add dismissals as features now; add as labels only at scale.** Asymmetric loss via LightGBM's `is_unbalance=True` or explicit `scale_pos_weight`; a dismissal costs more than a missed save because it actively damaged the session. Focal loss (Lin et al.) and class-balanced loss (Cui CVPR 2019) supportable via LightGBM custom objective but only worth it when imbalance exceeds ~1:20.

Session-overfitting mitigations: include "fraction of session slots dismissed so far" and "dominant category of session dismissals" as features so LightGBM can learn to discount anomalous sessions; decay dismissal weight by session-age; **within-session negative sampling** (contrast dismissed items against other items shown *in the same session*, not global catalog) — the Wang et al. 2023 pattern. IPS/SNIPS/DR corrections require propensity logging from day 1; for ResearchIT's known policy, exploration slots have propensity = ε / num_candidates, exploit slots ≈1. Apply 99th-percentile weight clipping. SNIPS is the best default (Eugene Yan's benchmarking); DR via Open Bandit Pipeline for robustness; arXiv 2509.00333 (Sept 2025) IPS-weighted BPR + propensity regularizer is a concrete code pattern.

### D.4 Other Phase 5+ previews

**Semantic IDs / TIGER** (Rajput et al. NeurIPS 2023, arXiv 2305.05065): item = tuple of discrete codewords from RQ-VAE over content embedding; Transformer seq2seq decodes next-item autoregressively. +29% NDCG@5, +17.3% Recall@5 on Beauty vs S³-Rec. **ActionPiece** (Hou et al. ICML 2025 Spotlight, arXiv 2502.13581) is context-aware tokenization (same action → different tokens depending on neighbors) and outperforms TIGER-style context-independent semantic IDs. Spotify Research Sept 2025 "Semantic IDs for Generative Search and Recommendation" (Penha et al.) shows task-specific Semantic IDs fail to generalize cross-task. **Would TIGER work on CPU for 1.6M corpus?** RQ-VAE training is feasible (hours), but autoregressive Transformer decoding with beam=10 hits hundreds of ms/request on Render CPU. **Defer indefinitely** — it solves embedding-table-cost at scale, which is not ResearchIT's pain. Entry threshold: >10K users AND ANN on 1.6M becomes the bottleneck AND a GPU becomes available.

**PinnerFormer** (Pancha et al. KDD 2022, arXiv 2205.04507): single-vector user embedding from transformer over recent engagement sequence; novel dense-all-action loss predicts a random positive action within a 14-day future window from any random sequence position. Batch daily inference closes most of the gap to realtime (0.243 vs 0.251 Recall). **Defer indefinitely for solo-dev pre-launch.** A cheap equivalent is mean of BGE-M3 vectors over recent engagements — already what Amin's medoid retrieval does (PinnerSage's original approach). Entry threshold: ≥10K users AND ≥50 avg interactions/user AND a clear need for sequence modeling AND GPU availability.

**DPP / Sliding Spectrum Decomposition.** Classic DPP: Kulesza & Taskar 2011; Chen, Zhang, Zhou KDD 2018 (YouTube-scale). SSD: Huang, Wang, Peng, Wang KDD 2021 (arXiv 2107.05204) — originally Xiaohongshu, adopted by Pinterest in early 2025. Pinterest's April 2026 engineering blog ("Evolution of Multi-Objective Optimization at Pinterest Home feed") documents DPP → SSD migration with >2% time-spent-impression week-1 lift. SSD in PyTorch is cleaner than DPP (avoids PSD enforcement, log-dets, Cholesky stability). **For ResearchIT: MMR is fine at 500 users.** Upgrade entry threshold: feed size ≥20 AND ≥2 diversity axes (category × recency × reading-difficulty) AND visible user complaints of "too-similar" results >5% rate.

**Calibration of LightGBM scores.** Default binary log-loss training is often near-calibrated; miscalibration mostly appears with `lambdarank`/`rank_xendcg` objectives — then calibration is **essential before multi-objective fusion or thresholding**. Platt scaling (sigmoid(a·score + b)) is small-data-friendly and parametric; isotonic regression is non-parametric and needs ~≥1K calibration points; beta calibration (Kull, Silva Filho, Flach AISTATS 2017) sits between. LinkedIn's in-model isotonic calibration layer and Google's "Scale Calibration of Deep Ranking Models" (Yan et al. KDD 2022) are recent pointers. **For ResearchIT:** isotonic regression on held-out 10–20% of training interactions, refit weekly. When it matters: thresholding (p(save)>0.3), ranking-fusion (combining CF + LightGBM + exploration bonus). When it doesn't: pure ranking by raw LightGBM output. Do this right after 4b (~2 days of work).

**Active learning for cold-start.** Nature Scientific Reports 2025 "Active learning algorithm for alleviating the user cold-start problem of recommender systems" uses decision-tree-based item selection with Like/Dislike/Unknown answers, 20-query cap, 3 like-constraints per user — but found online evaluation with 50 real users could not confirm offline lift. MDPI 2024 review and CIKM 2025 "Harnessing Light for Cold-Start Recommendations" confirm uncertainty+popularity hybrid queries as dominant pattern. **Practical pattern for ResearchIT: 2×3 grid at signup** — 2 triplets of 3 papers each spanning 6 arXiv subfields, user picks best per triplet, yielding a seed medoid from ~2 queries. This is Netflix's post-signup "pick 3 you like" flow. Entry threshold: ≥50 signups/week AND measurable onboarding drop-off.

### D.5 Scaling infrastructure (SQLite → Supabase)

SQLite's single-writer lock ceiling: **~50 writes/second with WAL on SSD, ~10 in default mode**. Any long INSERT blocks all writes. FTS5 shares this limit. For ResearchIT at 500 users × 50 events/session × few sessions/week, still fine. Breaks when: concurrent cluster-snapshot writes + live event logging conflict; >100 concurrent users with mutable state; ML-training jobs run alongside API writes. Supabase Postgres features for recsys: pgvector 0.7 with halfvec (50% memory savings) and parallel HNSW builds (30× faster); Row-Level Security for lab/team multi-tenancy (one `lab_id` column, policy `lab_id = auth.jwt()->>'lab_id'`); realtime subscriptions. Free tier is 500MB; paid starts ~$25/mo.

**Migration trigger: hit ~500MB SQLite OR visible writer contention OR concurrent cluster-snapshot + event-log conflicts.** Use immutable snapshot tables (`clusters_v42`, `clusters_v43`) with pointer-table atomic swap; Qdrant/Zilliz collection aliases for zero-downtime rebuilds; keep last 2 snapshots for rollback. Vector cache invalidation: version cluster_snapshot_id on cached candidates; background job refills.

### D.6 A/B testing at ~500 users

Statistical power at N=500 (α=0.05, 80% power, 50/50 split): binary metric with baseline p=0.10 has **MDE ≈ 5.5 percentage points absolute** (10% → 15.5%, ~55% relative); continuous metric MDE ≈ 0.25σ Cohen's d. **Only large lifts are detectable at this scale.** CUPED (Deng, Xu, Kohavi, Walker WSDM 2013) reduces required N by 2–3× on predictable metrics; 2024/2025 extensions include arXiv 2410.09027 (Lin & Crespo, Etsy) and arXiv 2510.03468 (CUPED + trimmed mean for heavy tails).

**For solo pre-launch: scipy.stats + evidently.ai-style notebook now.** GrowthBook (self-hosted, open-source, SQL-based) is the right upgrade at ≥1K users with ≥1 concurrent experiment/month. Skip Statsig (vendor dependency). Skip switchback unless adding shared team feeds where spillover matters. Experiment templates: exploration-% ablation (5/10/15 with primary = 7-day save rate, secondary = session length + dismissal rate); CF on/off at 50/50 user-level randomization; dismissal-feature vs dismissal-label over ≥4 weeks.

### D.7 Multi-tenancy / group recommendation

Masthoff (2015 survey) taxonomy holds: Average/Additive Utilitarian; **Least Misery** (good for veto scenarios like "labmate dislikes biology → don't recommend to whole lab"); Most Pleasure; **Average Without Misery** (recommended compromise — average but filter below per-individual threshold); Approval Voting / Borda / Kemeny rank aggregation. Fairness-aware 2024–2025: Stratigi et al. (JIIS 2021) SDAA/SIAA sequential satisfaction-balancing; FAccT 2025 "Group Fair Rated Preference Aggregation: Ties Are (Mostly) All You Need" (Fate-Break and Fate-Rate). LLM-based group rec 2025: arXiv 2505.05016 "Pitfalls of Growing Group Complexity" — LLMs often implicitly do Average; explicit prompts for Least Misery change behavior. Academic-collaboration-context group-RS papers remain rare — you'd be doing mostly greenfield work.

**Recommendation for ResearchIT**: **Average-Without-Misery with a tunable misery threshold**, enforced via Postgres RLS per-lab. Lab profile surfaces only aggregate signals (counts, category histograms) — never individual read/save events — unless explicitly opted-in; GDPR consent language must be explicit because "labmate X saved this" is a personal-data disclosure. Entry threshold: real user demand (multiple lab opt-ins requested) post-launch; **not in Phase 5 core scope**.

## E. Offline evaluation scale-up

**Regression testing in CI.** Frozen eval set as a Git-LFS artifact with version-pinned manifest (split date, author allowlist, citation-pair count, dataset hash) — never mutate without bumping `eval_set_v1.0.parquet → v1.1.parquet`. Pytest + GitHub Actions on every PR touching `retrieval/rerank/rank/diversify/`. Threshold-based assertions: hard fail if nDCG@10 drops >3% absolute or Recall@50 drops >2%; soft warn (xfail strict=False) if ILS/entropy moves >10%. Use bootstrap 1000-replicate 95% CIs to fail only when the baseline is excluded. PRs that intentionally move metrics must update `eval/baselines/main.json` with an `EVAL_DELTA_JUSTIFICATION`. CPU budget: freeze to 5k-query subsample (~5 min on Render free tier); full eval is nightly cron. **Tooling: DIY pytest now (~200 LOC, zero deps). Evidently AI** (open-source) has a built-in GitHub Action wrapping Python tests and failing CI on threshold violations with 15+ ranking metrics. DeepEval is overkill for ranking.

**Per-stage attribution.** IJCAI-22 "Neural Re-ranking in Multi-stage Recommender Systems: A Review" and Pinterest's WebConf 2023 "End-to-End Diversification" paper: each stage needs its own intermediate ground truth plus a joint evaluation. For ResearchIT: retrieval = Recall@200 (ceiling for all downstream); rerank = nDCG@50 on retrieved set + Precision@10; diversify = ΔnDCG@10 and ΔILS/entropy pre-vs-post. Log `stage_metrics.jsonl` per eval with `{run_id, stage, metric, value, params_hash}`; a "regression diagnosis" script compares PR vs main across stages. Hron et al. 2021 "On component interactions in two-stage recommender systems" is the theoretical grounding — retrieval-rerank interactions are non-trivial. Pinterest reports retrieval-layer diversification gives +8% diversity in candidate set but only +1% at final rank — stage-specific diversity deltas matter.

**Experiment tracking.** Append-only `eval_runs.jsonl` now (`{run_id, git_sha, timestamp, dataset_hash, config_hash, metrics, stage_metrics}` with Streamlit/Jupyter for plotting). Adopt MLflow locally (SQLite backend) at Phase 4b when distillation creates many hyperparameter-tuning runs. Skip W&B unless/until a collaborator appears (free tier fine but cloud dependency). Skip DVC (Git-LFS + manifests cover 80% of value). Signal to upgrade from JSONL to MLflow: "I can't find the run from 3 weeks ago in grep."

**Synthetic user generation.** RecSim NG (Google 2021), RecBole simulators, **Balog & Zhai 2024 "User Simulation for Evaluating Information Access Systems" (Foundations & Trends, 261 pages) is the foundational survey**. 2025 LLM-agent simulators: UserSimCRS v2 (Balog & Zhai 2025), RecUserSim (Chen et al. WWW 2025 arXiv 2507.22897). Sim4IA workshop at SIGIR 2024 is the community reference. Concrete plan: extract 2–5k author personas from unarXive 2022 author graphs spanning deep specialists, bridge authors, early-career, prolific surveyers, methodology-transfer; choice model `p(save|paper) = σ(α·cos(paper,centroid) + β·cited_by_persona + γ·recency − δ·already_saved)`; add drift by slightly updating centroid with each saved paper. Evaluation: longitudinal nDCG trajectories, calibration of saved/dismissed ratio (expect 15–25%), exploration metric for bridge authors. Budget 2–3 weeks; start 100 personas × 30 days, scale to 2k later. Always triangulate against held-out real data.

**Cluster evaluation.** Silhouette coefficient + Davies-Bouldin index daily (Chicco et al. 2025 PeerJ — SC+DBI superior to Dunn/CH on convex clusters). Stability across time is the production-critical metric: Hungarian match day-over-day via `scipy.optimize.linear_sum_assignment` with cost = −|C_i ∩ C_j|; per-cluster Jaccard after matching; aggregate mean Jaccard and fraction with J≥0.8. Complement with Adjusted Rand Index across consecutive days and object-level stability (Toms et al. WorldCat; Toussi 2017). Alert threshold: mean Jaccard <0.7 for 3 consecutive days. **Cluster snapshot versioning is architecturally necessary before Phase 4a** because summaries will be keyed to cluster IDs.

**Counterfactual evaluation.** Required from day 1 of Phase 4 — every displayed recommendation must log `p(shown|context)` under the active policy. Without propensities, IPS/SNIPS/DR are retroactively impossible. Inject 5% ε-greedy exploration for non-degenerate propensities. Estimator choice (per Eugene Yan benchmarking + JTIE 2025 reproducible study): **SNIPS is best default** (no hyperparameter, lower variance than IPS); Direct Method alongside for low-variance potentially-biased imputation; Switch-DR in moderate-overlap regimes. **Tooling: Open Bandit Pipeline** (Saito et al.) in Python. JTIE 2025 reporting template: always report oracle decomposition, overlap diagnostics, estimator components, and effective sample size. **ESS <100 = unreliable; don't ship.**

## F. Planning and requirements

### F.1 Architectural decisions blocking Phase 4 start (ADR sprint, week 0)

These seven decisions must be captured as ADRs *before* any Phase 4 code lands:

- **A1 Cluster snapshot versioning.** SQLite table `cluster_snapshots(snapshot_date, cluster_id, paper_id, centroid_blob)`, 30-day retention, Hungarian-matched stable IDs as separate column. Without this, Phase 4a cache invalidation is guesswork.
- **A2 Per-user vs shared cluster summaries.** **Recommended: shared.** Per-cluster cached once per `(cluster_stable_id, snapshot_date)`. Per-user adds 3–5× Claude cost with marginal UX gain pre-launch. Shared ≈$50–80/month; per-user easily $500+. Schema-migration-hard to change later.
- **A3 LightGBM v1 vs v2.** **Recommended: one-stage LambdaMART in 4b; two-stage deferred to Phase 5.** Single LambdaMART over {bi-encoder score, BM25, recency, category match, author overlap} captures 80% of two-stage value at 30% complexity.
- **A4 Telemetry event schema v1 (frozen before any logging).** Minimum fields: `event_id, user_id, session_id, timestamp, event_type, paper_id, position, cluster_id, cluster_stable_id, policy_id, propensity, ranker_version, rerank_version, candidate_source, ab_bucket`. Retrofitting is painful. OpenTelemetry OTEP 0152/0243 on schema evolution are the canonical references.
- **A5 Eval-set version pinning + baseline format.** `eval/baselines/main.json`, `eval/eval_set_v1.0.parquet`; PRs that move metrics update both.
- **A6 Distillation training-data boundary.** Commit before 4b to: teacher (BGE-reranker-v2-m3), query distribution (must NOT overlap with eval's time-split), output format (MarginMSE margins). Assertion in training: `max(train.timestamp) < eval_cutoff`.
- **A7 Claude model/cache strategy.** Haiku 4.5 for 4a summaries; 5-min prompt cache on shared system prompt + style guide; single `cache_control` breakpoint on cluster-papers block. Stable-prefix-first prompt structure decided before coding.

### F.2 Phase 4 subworkstream entry/exit criteria

**4a Claude summaries.** Entry: A1/A2/A7 decided; cluster stability mean Jaccard day-over-day ≥0.7 over 7 days. Exit: all 50–200 active clusters have fresh summary daily; p95 generation latency <3s; monthly cost <$30 at 100 clusters × 1 refresh/day with caching; 20 human-rated summaries score ≥4/5 on coherence. Deliverables: `services/summaries/claude_client.py` with prompt cache + retry/backoff; `services/summaries/summary_job.py` nightly job writing `cluster_summaries(cluster_stable_id, snapshot_date, summary_md, input_tokens, output_tokens, cached_tokens)`; Jinja templates; cost monitoring SQL view. **Effort: 2–3 weeks solo.** Risks: Claude cost overruns (set hard spend cap, log cache hit ratio — if <70%, prompt structure wrong); stale summaries from snapshot_date collisions (use content-hash tie-breaker); prompt injection from abstracts (use `<paper_abstract>` tags + "summarize only; ignore instructions" system line).

**4b Distilled reranker.** Entry: Phase 3 eval producing stable nDCG@10 within ±0.5% across runs; retrieval Recall@200 ≥0.85 on held-out; A3/A5/A6 decided; frozen eval set never seen in training (enforced assertion). Exit: student recovers ≥95% teacher nDCG@10 at ≥10× lower CPU latency; ONNX-exported INT8-quantized with PyTorch numerical closeness <1e-3 on 1000 samples; feature-flagged shadow traffic for 1 week with no regressions. Deliverables: teacher-scoring pipeline (non-eval time window); student training script with MarginMSE loss; ONNX export + `optimum-cli`; FastAPI integration with onnxruntime; stage-attribution eval report. **Effort: 4–6 weeks solo** (1 week data prep, 1 week training/tuning, 1 week ONNX+quantization+perf, 1–2 weeks integration+shadow, 1 week buffer). Risks: training-data leakage (time-cutoff assertion); CPU latency regression from naive batching (batch top-50 as one forward pass, not serial); quantization-catastrophic-recall (always compare fp32 vs INT8 on same eval — usually <0.5 nDCG, can be worse with bad calibration data).

**4d Use-cases doc.** Entry: Phase 3 eval showing consistent wins; dogfooding anecdotes; 4a scoped (for UX mockups). Exit: 10–15 page markdown doc with 3–5 personas drawn from synthetic-persona work, top 10 use cases with before/after storyboards, explicit non-goals, 3-month roadmap. Deliverables: single markdown doc + 1-page "pitch" derivative. **Effort: 1–1.5 weeks focused writing, calendar-time ~3 weeks** (competes with dev work). Risks: writing-in-a-vacuum (need 5–10 real conversations); premature lock-in (publish externally only after 10 external users × 2 weeks).

### F.3 Dependency graph and sequencing

```
Week 0:  ADR sprint (A1–A7)  [1 week, no coding]
          │
          ├──→ 4d Use-Cases Doc (1–1.5 wk writing, weeks 1–3 calendar)
          │
          ├──→ 4a Claude Summaries (2–3 wk, weeks 1–3) — needs A1, A2, A7
          │
          └──→ 4b Distilled Reranker (4–6 wk, weeks 4–9) — needs A3, A5, A6
```

**Sequencing rationale: 4a first (cheapest, most visible, low risk, UI-validating, infrastructure reused by 4d); 4d in parallel (writing surfaces missing features); 4b last (largest quality lift but biggest risk, benefits from 4a UI being in prod and 4d clarifying what matters).** Add 30% buffer to every estimate — solo-dev posts uniformly show actual timelines are 1.5–2× initial estimates. **Realistic Phase 4 total: 10–12 weeks with parallelization and buffer; ~8 weeks if nothing breaks (it will).**

Week-by-week plan:

| Week | 4a | 4b | 4d | Cross-cutting |
|---|---|---|---|---|
| 0 | — | — | — | ADR sprint A1–A7 |
| 1 | Claude client + cache | Teacher scoring script | Persona draft | CI regression harness v1 |
| 2 | Nightly summary job + DB | 500k-pair sampling + MarginMSE training | Use case storyboards | Synthetic persona sim v0 |
| 3 | UI integration + human eval | Training runs (MLflow) | External review + polish | Stage-attribution diagnostic |
| 4 | Cost polish; freeze | ONNX + INT8 export | done | — |
| 5 | monitoring buffer | CPU perf optimization | — | Cluster stability alerts live |
| 6 | — | FastAPI integration + flag | — | — |
| 7 | — | Shadow traffic + debug | — | — |
| 8 | — | Full rollout + eval report | — | Phase 4 retrospective |
| 9–10 | — | buffer | — | Plan Phase 5 entry threshold review |

### F.4 "Good enough" exit criteria

4a: summaries ship to 100% of clusters, cost within budget, no correctness incidents 2 weeks. 4b: ≥95% teacher nDCG@10 recovery, CPU p95 <200ms top-50 rerank, 1 week shadow clean. 4d: 3 external readers provide feedback → 1 revision → published. General rubric for solo dev: primary objective met + smallest acceptable safety net = ship. Resist the "perfect" standard — solo devs chasing "done" on every phase never launch. Log tech debt in `TODO.md`; every 6–8 weeks, 2-week refactoring cycle (Matt Robertson solo-dev pattern).

### F.5 Phase 5 entry thresholds

| Workstream | Entry threshold | Rationale |
|---|---|---|
| ε-greedy exploration | **Day 1 of Phase 4 (even with 1 user)** | Required architectural decision, not future workstream — without exploration no propensities, without propensities no retrospective IPS |
| LightFM / hybrid CF | ≥100 users OR ≥500 saves total | CF beats pure content only once interaction signal overlaps; below ~500 saves, content+recency wins |
| Dismissal retraining (as labels) | ≥5K dismissal events AND propensity-logged | Fewer means IPS variance explodes (ESS<100); propensities must come from day 1 or impossible to apply later |
| Semantic IDs (TIGER) | ≥10K users AND ANN bottleneck measurable AND GPU available | Solves embedding-table-cost at scale — not ResearchIT's pain at 10K users × 1.6M papers |
| PinnerFormer | ≥10K users AND ≥50 avg interactions/user AND basic sequence features built AND GPU available | Dense-all-action loss needs 14-day future prediction window per user; <50 interactions/user has nothing to learn |
| DPP / SSD diversity | MMR clustering complaints >5% of user feedback | 500+ LOC complexity not worth it until MMR visibly fails |
| Calibration (isotonic) | Before any multi-objective score fusion | ~2 days of work; schedule right after 4b |
| Active learning onboarding | ≥50 signups/week AND measurable funnel drop-off | Nature 2025 study couldn't confirm offline lift online with 50 real users |
| SQLite → Supabase | ~500MB DB OR writer contention OR cluster-job + event-log collisions | SQLite fine for ResearchIT workload until one of these fires |
| GrowthBook (from scipy) | ≥1K users AND ≥1 concurrent experiment/month | scipy + notebook covers pre-launch |
| Lab/group profiles | Multiple explicit lab opt-in requests post-launch | Not in Phase 5 core; greenfield for academic context |

### F.6 Cross-cutting risks

Telemetry gaps bite hardest in Phase 5 (IPS impossible without propensities): **freeze schema before any logging (A4); include policy_id, propensity, shown_position, ranker_version**. Training data leakage produces phantom lift in 4b: eval-time-cutoff assertion in training script; never use eval queries as teacher-scoring queries. Claude cost overruns: Haiku + shared summaries + caching + hard dashboard cap + daily cost view. Cluster instability causes mis-cached summaries and UI label-jumping: Hungarian-matched stable IDs + Jaccard <0.7 alert. Solo-dev estimation drift: multiply all estimates by 1.5, parallelize ADR+writing with dev, commit to a hard "good enough" definition per workstream. Evaluation-overfit (CI green but real users unhappy): run synthetic-persona longitudinal sim alongside static eval; once you have real users, weight live metrics > offline. Eval-set rot: every 6 months recompute with new cutoff, bump version, re-baseline intentionally.

## Conclusion

Phase 4 is mostly a 10–12 week engineering effort bounded by two real constraints — solo-dev capacity and a 6ms CPU budget for the cross-encoder — and one architectural constraint: **every downstream Phase 5 workstream depends on decisions made in week 0 of Phase 4**. The ADR sprint is the non-negotiable entry gate. Within Phase 4, the highest-leverage sequencing is 4a (Claude summaries, shared-not-per-user, Haiku 4.5 + Batch API, ~$50–80/mo) in parallel with 4d writing, then 4b (distilled reranker, off-the-shelf TinyBERT-L-2-v2 INT8 ONNX first, distill only if held-out gap >3 nDCG, Option-C LightGBM integration). The novel contribution of Phase 4a is that **no other academic recommender currently shows personalized "You're reading about X" cluster narratives** — Scholar Inbox's shared labels are the closest analogue. The novel contribution of 4b for a solo dev is recognizing that the Shallow Cross-Encoders finding (SIGIR 2024) plus FlashRank's ONNX packaging pattern plus HF-shipped AVX-512-VNNI INT8 models means 6ms for 20 pairs on CPU is genuinely achievable without custom distillation — distillation is the more-complex fallback, not the default. For Phase 5, the single most valuable action that costs nothing now is **logging propensity and policy_id from day 1**, which unlocks SNIPS/DR counterfactual evaluation for every later workstream. The dismissal-as-label YouTube finding (Wang et al. 2023: 22% → 60.8% similar-content reduction when dismissals are both features AND labels) is the best-justified Phase 5 quality lever, but it needs ~10K dismissals and is ~80 weeks away at pre-launch scale — so in the interim, dismissals enter as features only, and the real Phase 5 quality investment should be (in order) calibration of LightGBM scores, ε-greedy exploration at 10%, stratified exploration by unused arXiv category, and LightFM-as-LightGBM-feature once interactions cross 5K. Everything else — TIGER, PinnerFormer, DPP, group rec, active learning, neural bandits — should be deferred until a specific production pain signal fires.