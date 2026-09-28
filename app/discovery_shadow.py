"""Offline, source-capped comparison against an exported baseline page.

No writes, clicks, impressions or production configuration changes. This is an
experimental composition policy, not a second implementation of the serving feed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from app.discovery_store import EMBEDDING_CONTRACT, validated_vector
from app.recommend.clustering import compute_clusters
from app.recommend.diversity import mmr_rerank
from app.recommend.fusion import allocate_quotas, merge_quota_results
from app.recommend.reranker import compute_features, heuristic_score


def compare_feed(baseline: dict, fresh: list[dict], now: float, *, hf_cap: int = 3,
                 min_similarity: float = .45, max_paper_age_days: int = 90) -> dict:
    """Cap additions; replace within the same interest to preserve baseline shares.

    The relevance threshold is an experimental eligibility cutoff, not calibrated
    confidence. All input vectors must declare the same encoding contract.
    """
    if baseline.get('embedding_contract') != EMBEDDING_CONTRACT:
        raise ValueError('Baseline embedding contract missing or incompatible')
    if not 0 <= hf_cap <= 3 or not 0 <= min_similarity <= 1 or max_paper_age_days < 1:
        raise ValueError('Invalid experimental limits')
    seed_ids=baseline.get('seed_ids',[])
    vectors=baseline.get('seed_vectors',[])
    if not seed_ids or len(seed_ids)!=len(vectors) or len(seed_ids)!=len(set(seed_ids)):
        raise ValueError('Provide distinct seed IDs and aligned vectors')
    seeds=np.array([validated_vector(v) for v in vectors],dtype=np.float32)
    clusters=compute_clusters(seed_ids,seeds)
    medoids=np.array([c.medoid_embedding for c in clusters])
    medoids/=np.linalg.norm(medoids,axis=1,keepdims=True)
    excluded=set(seed_ids)|set(baseline.get('excluded_ids',[]))
    base=[];seen=set()
    for p in baseline.get('papers',[]):
        aid=p['arxiv_id']
        if aid not in excluded and aid not in seen:
            base.append({**p,'origin':'baseline'});seen.add(aid)
    base=base[:10]
    incoming=[];rejected=[]
    for p in fresh:
        aid=p['arxiv_id']
        if aid in seen or aid in excluded:
            rejected.append({'id':aid,'reason':'duplicate or excluded'});continue
        if p.get('embedding_contract')!=EMBEDDING_CONTRACT:
            rejected.append({'id':aid,'reason':'incompatible embedding'});continue
        last_seen=p.get('last_seen')
        if not isinstance(last_seen,(int,float)) or not 0 <= now-last_seen <= 48*3600:
            rejected.append({'id':aid,'reason':'stale or future source observation'});continue
        try:
            published=datetime.fromisoformat(p['published_at'].replace('Z','+00:00'))
            if published.tzinfo is None or not 0 <= now-published.timestamp() <= max_paper_age_days*86400:raise ValueError()
            vec=np.array(validated_vector(p['vector']))
        except (ValueError,KeyError,TypeError):
            rejected.append({'id':aid,'reason':'invalid metadata/vector'});continue
        similarity=float(np.max(medoids@vec))
        if similarity<min_similarity:
            rejected.append({'id':aid,'reason':'below experimental topic-fit threshold'});continue
        incoming.append({**p,'origin':'huggingface'});seen.add(aid)
    candidates=base+incoming
    if not candidates:
        return {'baseline':[],'hf_only':[],'combined':[],'rejected':rejected,'changes':[],
                'quality':'unjudged','source_cap':hf_cap}
    embeddings=np.array([validated_vector(p['vector']) for p in candidates],dtype=np.float32)
    assignments=np.argmax(embeddings@medoids.T,axis=1)
    # Existing heuristic; the profile proxy is explicit, not a restored user's EWMA.
    profile=seeds.mean(axis=0);profile/=max(np.linalg.norm(profile),1e-9)
    metadata=[{**p,'published':p.get('published_at',p.get('published',''))} for p in candidates]
    features=compute_features(embeddings,metadata,profile,profile)
    # Freeze the heuristic's time-dependent input to this comparison's as-of.
    # Other age cross-features are unused by heuristic_score; no learned model here.
    for i,p in enumerate(metadata):
        try:
            date=datetime.fromisoformat(p['published'].replace('Z','+00:00'))
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            age=max(0,int((now-date.timestamp())/86400))
        except (ValueError,KeyError,TypeError):
            age=365
        features[i,6]=np.exp(-.002*age)
    scores=heuristic_score(features)
    lookup={}
    for i,p in enumerate(candidates):
        ci=int(assignments[i])
        lookup[p['arxiv_id']]={k:v for k,v in p.items() if k!='vector'}
        lookup[p['arxiv_id']].update(interest=ci,score=float(scores[i]),
             similarity=float(embeddings[i]@medoids[ci]), reason=f'Related to seed {clusters[ci].medoid_paper_id}')
    hf_groups=[]
    for ci,c in enumerate(clusters):
        indices=[i for i,p in enumerate(candidates) if i>=len(base) and assignments[i]==ci]
        if not indices:
            hf_groups.append([]);continue
        indices.sort(key=lambda i: -scores[i])
        hf_groups.append(mmr_rerank(medoids[ci],embeddings[indices],
            [candidates[i]['arxiv_id'] for i in indices],scores[indices].tolist(),top_k=min(10,len(indices))))
    hf_order=merge_quota_results(hf_groups,allocate_quotas([c.importance for c in clusters],10))[:10]
    combined=[p['arxiv_id'] for p in base];changes=[]
    for aid in hf_order:
        if len(changes)>=hf_cap:break
        ci=lookup[aid]['interest']
        options=[(lookup[old]['score'],i,old) for i,old in enumerate(combined)
                 if lookup[old]['interest']==ci and lookup[old]['origin']=='baseline']
        if not options:continue  # never steal the last slot of another interest
        _,position,old=min(options)
        combined[position]=aid
        changes.append({'removed':old,'added':aid,'interest':ci,
                        'reason':'Experimental fresh-paper replacement within the same interest'})
    return {'profile':baseline.get('name','Reader'),
            'baseline':[lookup[p['arxiv_id']] for p in base],
            'hf_only':[lookup[aid] for aid in hf_order],
            'combined':[lookup[aid] for aid in combined], 'changes':changes,
            'rejected':rejected, 'source_cap':hf_cap,'min_similarity':min_similarity,
            'quality':'unjudged', 'max_paper_age_days':max_paper_age_days, 'as_of':datetime.fromtimestamp(now,timezone.utc).isoformat(),
            'method':'Imported baseline order; seed-mean scoring proxy, real Ward/MMR/quota helpers. Not the full serving pipeline.'}
