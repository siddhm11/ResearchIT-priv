"""Reusable diagnostics. Synthetic results test structure, never semantic quality."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def ranking_metrics(ranked: list[str], judgments: dict[str, int], k: int = 10) -> dict:
    """Pooled-judgment metrics; refuse partial labels instead of treating unknown as bad."""
    if k < 1 or any(type(v) is not int or not 0 <= v <= 3 for v in judgments.values()):
        raise ValueError('Use positive k and integer grades 0..3')
    top = ranked[:k]
    if len(set(ranked)) != len(ranked):
        raise ValueError('Duplicate ranked IDs')
    coverage = sum(a in judgments for a in top) / len(top) if top else 0.0
    if not top or coverage < 1:
        return {'status': 'not_evaluable', 'judged_coverage': coverage,
                'precision_at_k': None, 'ndcg_at_k': None, 'recall_at_k': None}
    dcg = lambda grades: sum((2**g - 1) / math.log2(i+2) for i, g in enumerate(grades))
    ideal = dcg(sorted(judgments.values(), reverse=True)[:k])
    relevant = sum(g >= 2 for g in judgments.values())
    hits = sum(judgments[a] >= 2 for a in top)
    return {'status': 'measured', 'judged_coverage': coverage,
            'precision_at_k': hits / k, 'ndcg_at_k': dcg([judgments[a] for a in top])/ideal if ideal else None,
            'recall_at_k': hits/relevant if relevant else None,
            'returned': len(top), 'judged_pool_size': len(judgments)}


def vector_health(vectors: np.ndarray, expected_dim: int = 1024) -> dict:
    a = np.asarray(vectors)
    if a.ndim != 2 or a.shape[1] != expected_dim or not len(a):
        return {'status': 'invalid', 'reason': 'empty or wrong dimensions'}
    if not np.isfinite(a).all():
        return {'status': 'invalid', 'reason': 'nonfinite values'}
    norms = np.linalg.norm(a, axis=1)
    if np.any(norms < 1e-9):
        return {'status': 'invalid', 'reason': 'zero vectors'}
    unit = a / norms[:, None]
    off = (unit @ unit.T)[np.triu_indices(len(a), 1)]
    return {'status': 'measured', 'count': len(a), 'dimensions': a.shape[1],
            'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
            'mean_pair_cosine': float(off.mean()) if off.size else None,
            'near_duplicate_pairs': int((off > .9999).sum()),
            'note': 'Geometry only; healthy vectors can still retrieve irrelevant papers.'}


def structural_scenarios() -> list[dict]:
    from app.recommend.clustering import compute_clusters
    from app.recommend.fusion import allocate_quotas, merge_quota_results, enforce_quota_on_ranking
    from app.recommend.diversity import mmr_rerank
    from app.recommend.reranker import compute_features, heuristic_score
    cases = []
    for name, counts in [('Equal interests', [10,10]), ('Dominant + smaller interest', [16,4]),
                         ('Three interests', [8,8,8])]:
        first_pages = []
        cluster_counts = []
        violations = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            centers = np.eye(1024, dtype=np.float32)[:len(counts)]
            save_ids, saves = [], []
            for ci, n in enumerate(counts):
                for j in range(n):
                    save_ids.append(f'seed-{ci}-{j}')
                    v = centers[ci] + rng.normal(0,.004,1024)
                    saves.append(v / np.linalg.norm(v))
            clusters = compute_clusters(save_ids, np.array(saves, dtype=np.float32))
            cluster_counts.append(len(clusters))
            importance, origins, embeddings, pools = {}, {}, {}, []
            for c in clusters:
                topic = int(c.medoid_paper_id.split('-')[1])
                importance[c.cluster_idx] = c.importance
                ids = [f'candidate-{topic}-{j}' for j in range(40)]
                for aid in ids:
                    origins[aid] = c.cluster_idx
                    v = centers[topic] + rng.normal(0,.01,1024)
                    embeddings[aid] = v / np.linalg.norm(v)
                pools.append(ids)
            quotas = allocate_quotas([c.importance for c in clusters], 60)
            merged = merge_quota_results(pools, quotas)
            a = np.array([embeddings[x] for x in merged])
            # A deliberately dominant profile tests whether serving order restores minorities.
            metadata = [{'published':'2026-01-01','category':'cs.AI'} for _ in merged]
            scores = heuristic_score(compute_features(a, metadata, centers[0], centers[0]))
            ordered = [merged[i] for i in np.argsort(-scores, kind='stable')]
            score_of = dict(zip(merged, scores))
            selected = []
            for ci,c in enumerate(clusters):
                group = [x for x in ordered if origins[x] == c.cluster_idx]
                selected.extend(mmr_rerank(c.medoid_embedding,
                    np.array([embeddings[x] for x in group]),group,
                    [score_of[x] for x in group],top_k=min(quotas[ci],len(group))))
            final = enforce_quota_on_ranking(selected, origins, importance)
            share = Counter(int(x.split('-')[1]) for x in final[:10])
            first_pages.append([share[i] for i in range(len(counts))])
            if len(clusters) != len(counts) or len(share) != len(counts) or len(final)!=len(set(final)):
                violations.append(seed)
        cases.append({'name': name, 'status': 'pass' if not violations else 'fail',
                      'evidence': 'synthetic geometry / real algorithm composition',
                      'runs':20,'save_counts':counts,'cluster_counts':cluster_counts,
                      'first_page_counts':first_pages,'violating_seeds':violations,
                      'limitation':'Not the HTTP pipeline and not actual BGE-M3 embeddings. HTTP tests are listed separately.'})
    return cases


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    rows = []
    for tc in root.iter('testcase'):
        status = 'failed' if tc.find('failure') is not None else 'error' if tc.find('error') is not None else 'skipped' if tc.find('skipped') is not None else 'passed'
        rows.append({'name':tc.get('name'), 'group':tc.get('classname'), 'status':status,
                     'seconds':float(tc.get('time','0'))})
    return {'counts':dict(Counter(r['status'] for r in rows)), 'cases':rows}


def model_split_audit(path: Path) -> dict:
    if not path.exists():
        return {'status':'unavailable'}
    counts = Counter()
    trees = 0
    for line in path.read_text().splitlines():
        if line.startswith('Tree='): trees += 1
        if line.startswith('split_feature='):
            counts.update(int(x) for x in line.split('=',1)[1].split())
    return {'status':'measured','trees':trees,'splits':dict(sorted(counts.items())),
            'personalization_splits_20_30':sum(counts[i] for i in range(20,31)),
            'note':'Model-file audit, not predictive quality. Production default is the personalized heuristic.'}


def compare_ann_exact(client, collection: str, query: list[float], search_params, k: int = 60) -> dict:
    """Compare ordinary ANN with unquantized exact search on the SAME shard.

    Recall here is index agreement, not relevance. No database mutations.
    """
    from qdrant_client.models import SearchParams, QuantizationSearchParams
    import time
    rows = {}
    for label, params in [('ann',search_params),('exact',SearchParams(
            exact=True, quantization=QuantizationSearchParams(ignore=True)))]:
        start=time.perf_counter()
        points=client.query_points(collection_name=collection, query=query,limit=k,
                                  with_payload=False, search_params=params).points
        rows[label]={'ids':[str(p.id) for p in points], 'latency_ms':round((time.perf_counter()-start)*1000)}
    truth=set(rows['exact']['ids'])
    return {'status':'measured' if truth else 'not_evaluable', 'k':k,
            'index_recall_at_k':len(set(rows['ann']['ids']) & truth)/len(truth) if truth else None,
            'ann_latency_ms':rows['ann']['latency_ms'],'exact_latency_ms':rows['exact']['latency_ms'],
            'exact_count':len(truth),'note':'Stored-space neighbor agreement, not semantic relevance or encoding fidelity.'}


def evaluate_judgment_file(path: Path) -> list[dict]:
    """Score explicitly supplied blinded judgments, never generate labels from votes."""
    import json
    from datetime import datetime
    cases=json.loads(path.read_text())
    if not isinstance(cases,list):raise ValueError('Expected list of judged cases')
    out=[]
    for case in cases:
        if not all(k in case for k in ['case_id','system','ranked','judgments','as_of','latest_source_observed_at']):
            raise ValueError('Missing evaluation provenance')
        as_of=datetime.fromisoformat(case['as_of'])
        latest=datetime.fromisoformat(case['latest_source_observed_at'])
        if as_of.tzinfo is None or latest.tzinfo is None or latest>as_of:
            raise ValueError('Future information or missing timezone in evaluation')
        if not isinstance(case['ranked'],list) or any(not isinstance(x,str) for x in case['ranked']):
            raise ValueError('Ranked IDs must be strings')
        out.append({'case_id':case['case_id'],'system':case['system'],
                    **ranking_metrics(case['ranked'],case['judgments'],case.get('k',10))})
    return out
