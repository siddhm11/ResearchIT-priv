# The Evolution of User Interest Tracking in ResearchIT

This document records the architectural shift in how ResearchIT models user interests, capturing the original project vision and explaining the transition to the current production architecture.

## 1. The Original Vision: Explicit Onboarding (Subject Vectors)

**The initial thought process:**
When a user opens the app for the very first time, they are greeted with an onboarding screen. They are given a list of overarching subjects (e.g., "Computer Vision", "Large Language Models", "Quantum Physics", "Biology"). 

The user explicitly checks the boxes for the subjects they care about. The system would retrieve fixed "Subject Vectors" corresponding to these categories and use them to query the vector database and generate the user's feed. 

### Why this made sense early on:
* **Cold Start Solution:** It immediately gives the system data to work with before the user has read a single paper.
* **Simplicity:** It maps perfectly to how legacy apps operate (like setting up a news aggregator).

## 2. The Problems with Explicit Subjects in Research

As the architecture matured, we realized that fixed subject vectors break down in an advanced academic context:

1. **Taxonomy Limitations:** Science moves faster than app menus. If a user selects "Reinforcement Learning," they miss out on emerging, unnamed sub-fields. Selecting predefined tags forces cutting-edge research into outdated buckets.
2. **Granular Specificity:** Selecting "Computer Vision" returns broad results (facial recognition, autonomous driving). But a researcher might only care about a hyper-specific niche like "unsupervised anomaly detection in industrial CT scans." Fixed subject vectors cannot capture this micro-granularity.
3. **Interest Drift:** A user might select "Robotics" during onboarding, but 6 months later, their thesis shifts to "Soft Materials." Relying on onboarding declarations means the app becomes stale unless the user constantly updates their settings.
4. **The "Centroid-in-Nowhere" Problem:** If a user selects distinct subjects like "Astrophysics" and "Economics", averaging these subject vectors mathematically results in a point in embedding space that means nothing, returning irrelevant garbage.

## 3. The Pivot to Implicit Behavioral Tracking (PinnerSage)

Instead of asking users what they want *once*, ResearchIT tracks what they *do* continuously. The architecture shifted from explicit, static vectors to implicit, dynamic embeddings driven by user interactions (saves and dismissals).

### A. EWMA Profiles (Temporal Dynamics)
Every time a user interacts with a paper, its 1024-dimensional BGE-M3 embedding is blended into their personal profile using an Exponentially Weighted Moving Average (EWMA):
* **Long-Term Profile ($\alpha=0.1$):** Updates slowly. Captures the user's enduring research identity across many sessions.
* **Short-Term Session Profile ($\alpha=0.4$):** Updates rapidly. Captures the immediate "rabbit hole" the user is delving into right now.
* **Negative Profile ($\alpha=0.15$):** Captures the embeddings of papers the user explicitly dismisses, allowing the system to learn what they *don't* like.

### B. Ward Clustering (Multi-Interest Routing)
To solve the "Centroid-in-Nowhere" problem, ResearchIT employs Ward Hierarchical Clustering. 
If a user saves papers on Biology and papers on NLP, the system does not average them. It detects the split and forms two distinct clusters. 
The system extracts the **Medoid** (the exact, real paper closest to the center of the cluster) from each grouping. 

### C. Prefetch and RRF 
During a feed request, the system sends the Long-Term medoids AND the Short-Term vector to Qdrant simultaneously as separate `Prefetch` queries. The results are unified seamlessly using **Reciprocal Rank Fusion (RRF)**.

## Conclusion

By deprecating manual "Subject Vectors" in favor of EWMA temporal tracking and Ward Clustering, ResearchIT transitioned from a standard filter-based aggregator into an intelligent, adaptive discovery engine capable of understanding complex, multi-disciplinary, and evolving academic interests without any manual user inputs.
