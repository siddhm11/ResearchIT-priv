**Interpretation.** Numbers deliberately omitted here; they are in the five lines above and the tables below.

* **Trust it for:** coarse topical similarity, judged by *percentile* against a field baseline rather
  than raw cosine. Unrelated random corpus pairs already score in BGE-M3's mid range, so a raw
  cutoff such as "> 0.5 means related" is meaningless. Use title+abstract text, as production does.
  Title-only BGE-M3 fails criterion 2 (focal pair not above the field p90).
* **Do not trust it for:** ordering papers inside one field's band. It ranks surface-term overlap
  (a neuroscience paper with "attention" in the title) above a methodologically closer ML paper,
  and does so in every BGE-M3 output. SPECTER2 and MiniLM with abstracts keep the trap below GAN.
* **Serving impact today is limited, for an uncomfortable reason.** That misordering sits far below
  the depth the app retrieves, so it does not reach feeds. But the same crowding means a seed
  with a dense neighbourhood (Attention) cannot reach its field-level relatives (B, BERT,
  Bahdanau) by dense nearest-neighbour search at all.
* **Evidence is thin:** one trap paper, one same-field control, two anchors. Criterion 1 for anchor
  B fails for every model, mostly on the debatable "positives must beat A" step; see the
  supplementary columns.
* **Single highest-value fix:** add a citation-trained similarity (SPECTER2 proximity) as a
  rerank feature over the BGE-M3-retrieved candidate pool. This leaves BGE-M3 retrieval and the
  1.8M stored vectors untouched. Gate it on re-running this harness over a few hundred labelled
  pairs (cited vs. keyword-sharing-but-uncited) to confirm the keyword bias is systematic. Adding
  a model is an architecture decision for doc 06, not something to ship off two anchors.
