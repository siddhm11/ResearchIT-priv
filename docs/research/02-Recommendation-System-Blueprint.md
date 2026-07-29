# Building a personalized paper recommendation system on BGE-M3 hybrid search

Your existing hybrid search system retrieves papers matching a query — but **a recommendation system must predict what a user wants before they ask**. The architectural gap is the absence of a user model: a persistent, evolving representation of each person's research interests that drives proactive paper surfacing. The good news is that your BGE-M3 embeddings in Qdrant/Zilliz are already the hardest piece to build, and Qdrant's native Recommendation API provides a direct path to personalization with minimal new infrastructure. What follows is a complete technical blueprint, from gap analysis through MVP implementation, grounded in the latest (2023–2025) research.

---

## The architectural gap between search and recommendation

Search is reactive: user types query → system returns relevant results. Recommendation is proactive: system surfaces papers the user would find valuable without explicit queries. Three components are missing from your current stack.

**First, a user model.** You have no persistent representation of who each user is — their interests, their evolving research focus, their engagement patterns. Search treats every request as independent. A recommendation system maintains a profile vector (or set of vectors) per user that accumulates signal over time and drives candidate retrieval.

**Second, an interaction logging layer.** Your system currently has no memory of what happened after results were returned. Which papers did the user click? How long did they read? Did they save or dismiss? These signals are the training data for personalization. Without event capture, you cannot learn.

**Third, a recommendation-specific retrieval and ranking pipeline.** Standard search optimizes for query-document relevance. Recommendation optimizes for user-document affinity, which requires different scoring, different candidate generation (often without a query at all), and different diversity/exploration strategies to avoid filter bubbles. Your RRF fusion of dense and sparse retrieval is excellent for search — but for recommendation you need a parallel pathway that takes a user profile vector (not a query) as input and retrieves candidates based on predicted interest.

The transition path is incremental: keep your search pipeline intact, and layer a recommendation pipeline alongside it that reuses your existing embeddings and vector infrastructure.

---

## How to build a user taste profile from search and click signals

User modeling is the core differentiator between search and recommendation. You have two signal types: search queries (explicit information needs) and paper clicks/reads (implicit preference signals). Here is how to turn them into a usable profile.

### The baseline: weighted average of interacted paper embeddings

The simplest and most effective starting point is computing a user profile vector as the **weighted average of BGE-M3 dense embeddings** of papers the user has interacted with. A 2023 RecSys paper by Bendada et al. ("On the Consistency of Average Embeddings for Item Recommendation") formally analyzed this approach and found it theoretically sound under reasonable distributional assumptions, though real-world embeddings show some consistency degradation compared to theory. Pinterest's PinnerSage (KDD 2020) validated weighted averaging at billion-scale production.

The weighting scheme should incorporate two dimensions — **interaction type** and **recency**:

```python
def compute_user_embedding(interactions, paper_embeddings, half_life_days=30):
    type_weights = {
        'click': 0.3, 'read': 0.7, 'save': 1.2, 'search_click': 0.5
    }
    weighted_sum = np.zeros(1024)  # BGE-M3 dense dim
    weight_total = 0.0
    now = time.time()
    
    for interaction in interactions:
        base_weight = type_weights.get(interaction.type, 0.3)
        # Dwell time modulation
        if interaction.dwell_ms and interaction.dwell_ms < 10000:
            base_weight *= 0.1  # <10s = likely noise
        elif interaction.dwell_ms and interaction.dwell_ms > 120000:
            base_weight *= 1.5  # >2min = deep read
        # Exponential recency decay
        age_days = (now - interaction.timestamp) / 86400
        decay = 2 ** (-age_days / half_life_days)
        w = base_weight * decay
        
        weighted_sum += w * paper_embeddings[interaction.paper_id]
        weight_total += w
    
    return weighted_sum / weight_total if weight_total > 0 else None
```

This profile vector lives in the same 1024-dimensional space as your paper embeddings, so you can directly use it as a query vector for ANN search in Qdrant. **Recency half-life of 30 days** is a reasonable starting point for academic papers; tune it based on how quickly your users' interests shift.

### Handling multi-interest users with clustering

A single average vector has a critical flaw: it dilutes when a user has diverse interests. PinnerSage demonstrated this vividly — averaging embeddings of painting, shoe, and sci-fi interests produced recommendations for "energy boosting breakfasts." The solution is **multi-interest representation**.

For your scale, use K-means clustering on a user's interacted paper embeddings (K=3–5 for typical researchers), then represent each cluster by its **medoid** — the actual paper embedding closest to the cluster center. At recommendation time, retrieve candidates for each cluster medoid independently, then merge results. This captures a user who works on both NLP and reinforcement learning without blending those interests into a meaningless midpoint.

### Incorporating search queries as high-value signals

Search queries are your strongest signal — they represent explicit, articulated information needs. Encode each query with BGE-M3 and maintain a **separate query-based profile** (recent query embedding average). At recommendation time, blend the query-based profile with the interaction-based profile:

```python
final_profile = alpha * interaction_profile + (1 - alpha) * query_profile
```

Start with **alpha = 0.6** (favoring interaction history over queries) and tune based on engagement metrics.

### LLM-generated interest summaries as an alternative user representation

A powerful modern technique: use an LLM to generate a natural-language summary of a user's interests from their reading history, then embed that summary with BGE-M3 to get a user vector. Meta's **EmbSum** (RecSys 2024) demonstrated this at scale, using Mixtral-8x22B to generate interest summaries that a smaller T5 model then learned to replicate. For your scale, you can call an LLM directly:

```python
prompt = f"""Based on these recently read papers, summarize this researcher's 
interests in 2-3 sentences: {[p.title for p in user.recent_papers[:20]]}"""
interest_summary = llm.generate(prompt)
user_vector = bge_m3.encode(interest_summary)
```

This approach is surprisingly effective because it produces a semantically rich, denoised representation. The LLM acts as an implicit topic model, filtering noise and identifying coherent research threads. Run it as a **daily batch job** per user — the cost is negligible at 10–100 users.

---

## Deep dive on recommendation approaches

### Content-based filtering with BGE-M3: your foundation

Content-based filtering using your existing embeddings is the **highest-ROI approach** and should be your primary recommendation strategy. The pipeline: compute user profile vector → ANN search in Qdrant → post-process with diversity. This works with a single user and zero collaborative signal.

Qdrant's built-in Recommendation API simplifies this dramatically. Instead of manually computing average vectors, pass the IDs of papers the user liked as `positive` examples and disliked papers as `negative`:

```python
results = client.query_points(
    collection_name="papers",
    query=models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[saved_paper_id_1, saved_paper_id_2, saved_paper_id_3],
            negative=[dismissed_paper_id_1],
        )
    ),
    strategy="best_score",  # Evaluates each example independently
    limit=20,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="year", range=models.Range(gte=2023))]
    )
)
```

The `best_score` strategy (available since Qdrant 1.6) is superior to `average_vector` for multi-interest users — it evaluates each candidate against all positive and negative examples independently during HNSW traversal, producing more diverse results.

**Key limitation**: content-based filtering creates filter bubbles. Mitigate with **MMR re-ranking** (λ=0.6 for relevance/diversity balance) and **Qdrant's grouped recommendations** (`recommend/groups` by category) to ensure cross-topic diversity.

### Collaborative filtering at small scale: not yet viable

Pure collaborative filtering is **not viable with 10–100 users**. CF requires finding meaningful co-interaction patterns, and the user-item matrix at this scale is far too sparse. Research consistently shows CF needs **200+ users with substantial interaction overlap** before it outperforms content-based baselines. The `implicit` library's ALS implementation would produce unreliable, noisy recommendations at your scale.

However, **plan for CF from day one** by logging all interactions in a structured format. When your user base reaches ~200 users with >20 interactions each, you can train an ALS model in minutes:

```python
import implicit
model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20)
model.fit(user_item_sparse_matrix)
```

An intermediate technique that works sooner: **one iteration of Implicit ALS** with fixed item embeddings. Fix the item factors to your BGE-M3 embeddings and solve for optimal user vectors via linear regression. This is equivalent to a weighted average but optimized to predict the interaction matrix — a few lines of linear algebra per user that can extract slightly more signal than naive averaging.

### Hybrid approaches for the transition period

**LightFM** is your best option for hybrid recommendation as your user base grows. It learns user and item embeddings as the sum of their feature embeddings, gracefully handling cold start by falling back on content features when interaction data is sparse:

```python
from lightfm import LightFM
model = LightFM(loss='warp', no_components=64)
model.fit(interactions, item_features=item_feature_matrix, epochs=30)
```

Item features for your case: arXiv category (cs.CL, cs.CV, etc.), year, author IDs. LightFM's WARP loss optimizes directly for top-K ranking quality. **Start training LightFM at ~50 users** with a switching strategy: use content-based for users with <10 interactions, LightFM for users with >10 interactions.

A simpler hybrid: **weighted score fusion**. Run content-based retrieval and (when available) collaborative retrieval independently, then combine:

```python
final_score = 0.7 * content_similarity + 0.3 * cf_score  # Start content-heavy
```

Shift the weights toward CF as your data grows.

### Session-based recommendation for immediate personalization

Session-based recommendation captures within-session behavior to personalize results in real time — critical for both cold-start users and capturing evolving intent. The simplest effective approach: maintain a **rolling session embedding** updated after each interaction:

```python
session_embedding = 0.7 * session_embedding + 0.3 * newly_clicked_paper_embedding
```

This exponential moving average gives recent clicks more weight while retaining session context. Use the session embedding as a query vector for ANN search, fused with the long-term user profile via weighted combination. For new sessions with a single search query, the query embedding alone is your session representation.

More advanced session-based models like **SASRec** (Self-Attentive Sequential Recommendation) use transformer attention over item sequences but require training data that you won't have initially. Revisit these at scale.

### LLM-based approaches worth considering

Three practical LLM integrations ranked by implementation effort:

**Low effort, high value — LLM as interest summarizer.** Generate per-user interest summaries as described above. Use these for both embedding-based retrieval and as explainable profile descriptions shown to users ("We think you're interested in...").

**Medium effort — LLM-augmented candidate scoring.** For your top-20 candidates from vector retrieval, use an LLM to score relevance given the user's profile: "Given this researcher works on [interests], rate how relevant this paper is on a 1-5 scale: [paper title + abstract]." This adds ~1-2 seconds of latency but can dramatically improve precision for the final displayed results.

**Higher effort — query augmentation.** When a user searches, use an LLM to expand their query with terms from their profile: "The user is interested in NLP and transformers. They searched for 'attention mechanisms.' Generate an expanded search query." This bridges search and recommendation by personalizing search itself.

---

## Solving cold start for new users with no history

Cold start requires a **layered fallback strategy** that gracefully degrades as less information is available.

**Layer 1: Onboarding (strongest cold-start solution).** Present new users with the arXiv category taxonomy and let them select 3–5 areas of interest. Optionally let them paste URLs or search for known papers as seed interests. Even selecting categories gives you enough signal to compute a meaningful initial profile by averaging the centroid embeddings of selected categories. Semantic Scholar found that ~5 saved papers + 3 "not relevant" ratings are sufficient for good initial recommendations.

**Layer 2: First-query bootstrapping.** The moment a user types their first search query, encode it with BGE-M3 and use it as the initial user profile vector. This single query provides immediate personalization signal. Store it and update the profile as more queries arrive.

**Layer 3: Population-level priors.** For users who haven't yet searched or selected interests, recommend trending papers (high recent click velocity across all users) with diversity across categories. Compute trending scores as a **daily batch job**: papers with unusual engagement spikes in the last 7 days.

**Layer 4: Exploration with bandits.** Reserve **20–30% of recommendation slots for new users** (decreasing to 5–10% for established users) for exploration. Use **epsilon-greedy** (simplest) or **Thompson Sampling** (more principled) to show diverse items that help identify the user's interests quickly. The bandit naturally shifts from exploration to exploitation as confidence in the user's preferences grows.

---

## Practical architecture built on your existing stack

You need three additions to your current Qdrant + Zilliz + BGE-M3 setup: an event logger, a user profile store, and a recommendation service layer.

### Minimal but effective architecture

```
User Action → Event API → PostgreSQL (interaction log)
                       → Redis (update recent items + session state)

Recommendation Request →
  1. Fetch user's recent paper IDs from Redis
  2. If cold-start → onboarding interests + trending fallback
  3. If search query present → encode query, fuse with user profile
  4. Call Qdrant Recommend API (positive=recent_saves, negative=dismissed)
     OR: compute user_embedding → ANN search in Qdrant
  5. Apply MMR diversity filter
  6. Return results + log impressions

Daily Batch Job →
  - Recompute all user profile embeddings from full interaction history
  - Update interest clusters (K-means on user's paper embeddings)
  - Refresh trending paper scores
  - Optional: regenerate LLM interest summaries per user
```

### What data to store and where

**Redis** (hot path, <1ms): last 20 interacted paper IDs per user, current session embedding, cached user profile vector. Total memory at 100 users: negligible.

**PostgreSQL** (interaction log): every impression, click, dwell time, save, dismiss with timestamps, session IDs, and source attribution (did the click come from search or recommendation?). Schema should capture `user_id`, `paper_id`, `interaction_type`, `timestamp`, `dwell_time_ms`, `scroll_depth`, `source`, `position_in_list`. Index on `(user_id, timestamp DESC)`.

**Qdrant** (unchanged): paper embeddings with payloads (title, abstract, category, year). Optionally store user profile embeddings as a separate collection for user-to-user similarity if you later want collaborative signals.

### When to compute what

At your scale of 10–100 users, **recompute user profiles on every significant interaction** (save, extended read >30s, dismiss). This takes milliseconds with <100 users and keeps profiles maximally fresh. No streaming infrastructure needed — a synchronous update in your API handler is sufficient. Run the full batch recomputation (clustering, LLM summaries, trending scores) as a **nightly cron job**.

### Qdrant's Universal Query API for hybrid personalized retrieval

Qdrant's query API lets you run multi-stage hybrid recommendation in a single atomic call, combining your existing dense+sparse search with recommendation:

```python
results = client.query_points(
    collection_name="papers",
    prefetch=[
        # Branch 1: User profile-based semantic retrieval
        models.Prefetch(
            query=user_profile_dense_vector,
            using="dense",
            limit=50
        ),
        # Branch 2: Sparse retrieval for lexical diversity
        models.Prefetch(
            query=user_profile_sparse_vector,
            using="sparse",
            limit=50
        ),
    ],
    # Fuse via RRF, then optionally rerank with ColBERT
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=20,
    query_filter=models.Filter(
        must_not=[models.FieldCondition(
            key="paper_id", match=models.MatchAny(any=already_read_ids)
        )]
    )
)
```

This reuses your existing dense and sparse indexes, adds personalization through the user profile vector, filters out already-read papers, and fuses results — all in one network round trip.

---

## Designing feedback loops that compound over time

The core feedback loop is: show recommendations → capture signals → update user model → show better recommendations. Three design decisions determine whether this loop converges to excellent recommendations or degenerates into a filter bubble.

### Which signals matter most

Rank signals by **reliability as preference indicators**, not just ease of capture:

Saves/bookmarks and extended reads (>2 minutes) are your strongest positive signals — they indicate deliberate engagement. Clicks alone are noisy; **50–70% of clicks under 10 seconds are curiosity or misclicks**, not genuine interest. Dismiss/hide actions are your strongest negative signal and are critically underused — actively surface a "not interested" button. Search queries are high-value because they represent articulated needs.

**Position bias correction** is essential: papers shown in position 1 get ~3x more clicks than position 5 regardless of relevance. Log the position of every impression and either debias in modeling or use inverse propensity weighting when computing preference scores.

### Avoiding filter bubbles

Three concrete mechanisms prevent recommendation homogenization. First, **diversity-aware re-ranking**: after scoring candidates by user affinity, apply MMR or use Qdrant's grouped recommendations by category to ensure cross-topic coverage. Second, **exploration budget**: always reserve 10–15% of recommendation slots for papers outside the user's established profile — trending papers, recent high-impact papers in adjacent fields, or papers popular with similar users. Third, **monitor and alert**: track per-user recommendation diversity (average pairwise cosine distance within recommendation lists) over time. If diversity drops below a threshold, increase the exploration budget.

### Update cadence

At 10–100 users, **real-time profile updates on every interaction** are computationally trivial and provide the best user experience. Update the Redis-cached profile vector immediately. Run the full batch pipeline (re-clustering, LLM summaries, trending scores, model retraining if using LightFM) nightly.

---

## Measuring whether recommendations actually work

### Offline evaluation for development

Use **time-split evaluation**: train on interactions before timestamp T, test on interactions after T. This simulates real-world temporal dynamics better than random splits. Key metrics:

**NDCG@10** is your primary metric — it rewards both relevance and correct ranking order, handling graded relevance (save > read > click). **Hit Rate@10** measures the fraction of users for whom at least one recommended paper was later interacted with — a simple, interpretable sanity check. **Coverage** measures what percentage of your paper catalog ever gets recommended. Low coverage (<10%) signals a severe filter bubble.

For evaluation with few users, use **leave-one-out**: for each user, hide their most recent interaction and test whether the system ranks that paper in the top-K. Cross-validate across users.

### Online evaluation for production

**CTR** (click-through rate on recommendations) is your primary online metric but must be paired with **downstream engagement** (dwell time on clicked recommendations vs. search results, save rate). A system that optimizes only for CTR will learn clickbait patterns.

At 10–100 users, traditional A/B testing lacks statistical power. Use **interleaving** instead: show results from your old system and new system interleaved in a single list, and measure which system's results get clicked more. Interleaving detects differences with **100x fewer users** than parallel A/B testing. Alternatively, use a simple **pre/post comparison**: measure engagement metrics for 2 weeks before and after deploying recommendations, accounting for temporal trends.

---

## Step-by-step implementation roadmap

### Phase 1: MVP in 1 week — content-based recommendations via Qdrant

Build the event logging layer first. Capture every click, with timestamp and paper ID, in PostgreSQL. Add a "save" button and a "not interested" button to your UI. Store the last 20 interacted paper IDs per user in Redis.

Then wire up Qdrant's Recommend API. Pass saved papers as `positive` examples, dismissed papers as `negative`, use the `best_score` strategy, filter out already-read papers. Apply a basic year filter (papers from the last 2 years) and return the top 10.

Add a "Recommended for You" section to your UI that calls this endpoint. This is your MVP: **personalized recommendations with zero model training**, leveraging your existing BGE-M3 embeddings and Qdrant infrastructure.

### Phase 2: Refined user modeling in week 2–3

Replace Qdrant's built-in averaging with your own **weighted user profile vector** (recency-decayed, engagement-weighted). Compute it on every interaction update. Use this vector for ANN search as a complement to the Recommend API.

Implement **multi-interest clustering**: K-means (K=3) on each user's interacted paper embeddings. Retrieve candidates for each cluster medoid independently, then merge with deduplication. This handles researchers with diverse interests.

Add **MMR diversity re-ranking** as a post-processing step (λ=0.6). Add **onboarding flow** for new users: category selection + optional seed paper search.

### Phase 3: Advanced personalization in week 4–6

Integrate **LLM-generated interest summaries** as a nightly batch job. Embed summaries with BGE-M3 and blend with the interaction-based profile vector.

Add a **cross-encoder re-ranker** (BAAI's `bge-reranker-v2` is ideal since it pairs with your BGE-M3 embeddings) for the final stage: retrieve 100 candidates via ANN, re-rank to top 10 with the cross-encoder using augmented queries that include user interest context.

Implement **session-based recommendation**: maintain a session embedding updated in real-time, blend with the long-term profile for users in active sessions.

### Phase 4: Scale-ready additions at 50+ users

Train **LightFM** hybrid model with paper metadata (categories, year) as item features. Use the switching strategy: content-based for users with <10 interactions, LightFM for active users.

Build an **evaluation dashboard** tracking NDCG@10, hit rate, coverage, and diversity metrics offline, plus CTR and engagement depth online. Implement interleaving for comparing system variants.

Add **exploration via epsilon-greedy** (ε=0.1 for established users, ε=0.25 for new users). This completes the feedback loop: recommendations → engagement → profile update → better recommendations, with built-in exploration to prevent convergence to filter bubbles.

The end result is a system that starts delivering personalized recommendations from day one using your existing embeddings, progressively improves as interaction data accumulates, and scales naturally from 10 to 1000+ users without architectural changes — only the addition of collaborative signals and more sophisticated models as data permits.

---

## Conclusion

The critical insight is that your BGE-M3 embeddings are already the most valuable asset for recommendation — the gap is not in representation quality but in user modeling and feedback infrastructure. **Start with Qdrant's native Recommend API using positive/negative paper examples** — this is literally a one-day implementation that delivers real personalization. The weighted-average user profile vector with recency decay and multi-interest clustering gets you 80% of the way to a production-quality system. Collaborative filtering is a future investment that only pays off at 200+ users; until then, content-based methods with LLM-augmented user profiles and cross-encoder re-ranking will outperform any CF approach at your scale. The most overlooked element is negative feedback — adding a "not interested" button and using those signals as negative examples in Qdrant's recommendation queries will improve precision more than any model upgrade.