#!/usr/bin/env python3
"""Build a self-contained recommendation dashboard from actual test artifacts.

Default: local mechanics + dependency inspection. --live adds read-only source
and vector-store probes, and an isolated-store HTTP recommendation evaluation
when configured services are reachable. --encode exercises BGE-M3 (may download
weights). Neither option trains or writes to external services.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.recommendation_evaluation import structural_scenarios, parse_junit, model_split_audit, vector_health, compare_ann_exact, evaluate_judgment_file

TRIPLETS = [
    ('calibration','How can I tell whether a language model knows when it is wrong?',
     'Measuring calibration of language model confidence against answer correctness.',
     'Calibrating robotic cameras using geometric fiducial markers.'),
    ('retrieval','Find evidence before answering a question with a language model.',
     'Retrieval augmented generation retrieves supporting documents for grounded answers.',
     'Generating realistic images conditioned on natural language descriptions.'),
    ('compression','Run large language models using less device memory.',
     'Post-training low-bit quantization compresses language model weights.',
     'Human memory consolidation during sleep and language learning.'),
    ('robotics','Learn robot actions from human demonstrations.',
     'Imitation learning for robotic manipulation from demonstrations.',
     'Detecting machine-generated language through statistical watermarking.'),
    ('medical','Segment tumors in medical scans.',
     'Neural segmentation of brain tumors in magnetic resonance images.',
     'Segmenting customer markets using behavioral profiles.'),
    ('safety','Stop malicious instructions hidden inside retrieved documents.',
     'Defending retrieval augmented language models against indirect prompt injection.',
     'Optimizing document retrieval latency using approximate nearest neighbors.'),
]

async def live_probes() -> dict:
    import httpx
    from app import config, qdrant_svc
    from app.hf_papers_svc import fetch_snapshot
    results={}
    async with httpx.AsyncClient(follow_redirects=False) as client:
        try:
            snap=await fetch_snapshot(client)
            results['huggingface']={'status':'measured','snapshot':snap,
                'note':'Daily community source, not personalized ground truth or trend velocity.'}
        except (httpx.HTTPError,ValueError) as exc:
            results['huggingface']={'status':'blocked','reason':type(exc).__name__,
                'note':'Direct runtime request failed; no empty-feed success inferred.'}
        for name,url,key,collection in [
            ('primary',config.QDRANT_URL,config.QDRANT_API_KEY,config.QDRANT_COLLECTION),
            ('b',config.QDRANT_B_URL,config.QDRANT_B_API_KEY,config.QDRANT_B_COLLECTION),
            ('recent',config.QDRANT_RECENT_URL,config.QDRANT_RECENT_API_KEY,config.QDRANT_RECENT_COLLECTION)]:
            if not url or not key:
                results[name]={'status':'not_configured'};continue
            try:
                from urllib.parse import quote
                r=await client.get(url.rstrip('/')+'/collections/'+quote(collection,safe=''),
                                   headers={'api-key':key},timeout=8)
                r.raise_for_status()
                info=r.json()['result']
                results[name]={'status':'measured','points':info.get('points_count'),
                    'vectors_config':info.get('config',{}).get('params',{}).get('vectors'),
                    'note':'Collection metadata only; does not establish relevance.'}
            except (httpx.HTTPError,ValueError,KeyError) as exc:
                results[name]={'status':'blocked','reason':type(exc).__name__}
    # Only issue vector reads after collection reachability is established.
    # Application fanout can swallow shard errors, so require every configured shard.
    configured=[v for k,v in results.items() if k!='huggingface' and v['status']!='not_configured']
    if not configured or any(v['status']!='measured' for v in configured):
        results['stored_embeddings']={'status':'blocked','reason':'Configured vector shards not all reachable.'}
        results['live_recommendations']={'status':'blocked','reason':'Vector preflight unavailable; no synthetic replacement.'}
        results['ann_exact']={'status':'blocked','reason':'Vector preflight unavailable.'}
        return results
    seeds=['1706.03762','1810.04805','1907.11692','1910.10683','2201.11903',
           '1512.03385','2010.11929','1505.04597','2103.14030','2104.14294']
    try:
        import numpy as np
        vectors=await asyncio.wait_for(qdrant_svc.get_paper_vectors(seeds),timeout=45)
        health=vector_health(np.array(list(vectors.values())))
        results['stored_embeddings']={'status':'measured','requested':len(seeds),'found':len(vectors),'health':health}
        neighbors=[]
        for aid,vec in list(vectors.items())[:6]:
            start=time.perf_counter()
            hits=await asyncio.wait_for(qdrant_svc.search_by_vector_with_scores(vec.tolist(),limit=10,exclude_ids={aid}),timeout=30)
            neighbors.append({'seed':aid,'hits':hits,'latency_ms':round((time.perf_counter()-start)*1000),
                              'judgments':'pending independent human review'})
        results['stored_embeddings']['neighbors']=neighbors
        agreements=[]
        for backend in qdrant_svc._active_backends():
            with qdrant_svc.use_backend(backend):
                for aid,vec in list(vectors.items())[:2]:
                    try:
                        agreement=await asyncio.wait_for(asyncio.to_thread(
                            compare_ann_exact,qdrant_svc._client(),qdrant_svc._collection(),
                            qdrant_svc._quantize_query(vec.tolist()),qdrant_svc._SEARCH_PARAMS),timeout=45)
                        agreements.append({'backend':backend,'seed':aid,**agreement})
                    except Exception as exc:
                        agreements.append({'backend':backend,'seed':aid,'status':'blocked','reason':type(exc).__name__})
        results['ann_exact']={'status':'measured' if agreements and all(x['status']=='measured' for x in agreements) else 'incomplete',
                              'results':agreements,'note':'Two seed vectors per active shard; index fidelity, not semantic quality.'}
        from app import db, user_state
        from app.main import app
        from app.recommend import profiles
        from app.routers import recommendations
        await db.init_db()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://audit') as c:
            feeds=[]
            for label,ids in [('Language models',seeds[:5]),('Vision',seeds[5:]),('Two interests',seeds)]:
                uid='audit-'+label.replace(' ','-')
                state=await user_state.ensure_loaded(uid)
                usable=[aid for aid in ids if aid in vectors]
                if len(usable)<5:
                    feeds.append({'profile':label,'status':'blocked','reason':'Fewer than five seed vectors'});continue
                for aid in usable:
                    state.add_positive(aid)
                    await db.log_interaction(uid,aid,'save')
                    await profiles.update_on_save(uid,vectors[aid])
                pages=[]
                import re
                for _ in range(3):
                    start=time.perf_counter()
                    r=await asyncio.wait_for(c.get('/api/recommendations',cookies={config.COOKIE_NAME:uid}),timeout=90)
                    pages.append({'http_status':r.status_code,'ids':re.findall(r'data-arxiv-id="([^"]+)"',r.text),
                                  'latency_ms':round((time.perf_counter()-start)*1000)})
                feeds.append({'profile':label,'status':'measured','pages':pages,
                              'quality':'unjudged','note':'Public paper IDs only; temporary user storage.'})
            results['live_recommendations']={'status':'measured','feeds':feeds}
    except Exception as exc:
        results['live_recommendations']={'status':'blocked','reason':type(exc).__name__}
    return results


def semantic_probes() -> dict:
    if not importlib.util.find_spec('FlagEmbedding'):
        return {'status':'blocked','reason':'FlagEmbedding/BGE-M3 runtime is not installed.', 'cases':len(TRIPLETS)}
    try:
        from app.embed_svc import encode_query
        import numpy as np
        rows=[]
        for name,q,pos,neg in TRIPLETS:
            vecs=np.array([encode_query(text)[0] for text in [q,pos,neg]])
            health=vector_health(vecs)
            if health['status']!='measured':
                rows.append({'case':name,'status':'invalid','health':health});continue
            unit=vecs/np.linalg.norm(vecs,axis=1,keepdims=True)
            ps,ns=float(unit[0]@unit[1]),float(unit[0]@unit[2])
            rows.append({'case':name,'status':'pass' if ps>ns else 'fail','positive_cosine':ps,
                         'negative_cosine':ns,'margin':ps-ns})
        return {'status':'measured','results':rows,
                'note':'Six authored contrastive smoke cases; not a held-out research relevance benchmark.'}
    except Exception as exc:
        return {'status':'blocked','reason':type(exc).__name__}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/recommendations')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--encode',action='store_true')
    parser.add_argument('--judgments',type=Path,help='JSON file of blinded, time-stamped human judgments')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Set before importing app/config; never restore/sync or touch the normal DB.
    with tempfile.TemporaryDirectory(prefix='researchit-eval-') as temp:
        os.environ['DB_PATH']=str(Path(temp)/'evaluation.sqlite')
        os.environ['TURSO_SYNC_DISABLED']='1'
        os.environ['RESEARCHIT_TEST_ALLOW_TURSO_SYNC']='0'
        junit=args.output/'tests.xml'
        junit.unlink(missing_ok=True)
        cmd=[sys.executable,'-m','pytest','-q','-m','not live and not browser',f'--junitxml={junit}']
        proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        # No captured stdout or exception messages in public report: may contain URLs/secrets.
        tests=parse_junit(junit) if junit.exists() else {'counts':{},'cases':[]}
        tests['exit_code']=proc.returncode
        tests['command']='.venv/bin/python -m pytest -q -m "not live and not browser"'
        tests['deselected_note']='Live/browser marks excluded. Skipped cases are shown individually.'
        from app import config, local_meta
        report={'generated_at':datetime.now(timezone.utc).isoformat(),'version':1,
                'tests':tests,'structural':structural_scenarios(),
                'judged_quality':evaluate_judgment_file(args.judgments) if args.judgments else [],
                'browser_verification':'Not performed by this runner. Local HTML opening was blocked by browser URL policy during the September 25 session.',
                'environment':{'FlagEmbedding':bool(importlib.util.find_spec('FlagEmbedding')),
                    'torch':bool(importlib.util.find_spec('torch')),
                    'lightgbm':bool(importlib.util.find_spec('lightgbm')),
                    'metadata_sidecar_exists':Path(local_meta.SIDECAR_PATH).exists(),
                    'scorer':config.RERANKER_MODE},
                'model':model_split_audit(ROOT/'models/reranker-phase6/production_model/reranker_v1.txt'),
                'semantics':semantic_probes() if args.encode else {'status':'not_run','reason':'Run with --encode; may download model weights.'},
                'live':asyncio.run(live_probes()) if args.live else {'status':'not_run','reason':'Run with --live for read-only external probes.'}}
        source=ROOT/'reports/recommendations/hf-observed-sample.json'
        report['hf_observed_sample']=json.loads(source.read_text()) if source.exists() else None
        tracked=[*ROOT.glob('app/**/*.py'),*ROOT.glob('tests/test_*.py'),*ROOT.glob('scripts/*recommendation*.py'),ROOT/'scripts/recommendation_dashboard.html']
        h=hashlib.sha256()
        for path in sorted(tracked):h.update(str(path.relative_to(ROOT)).encode());h.update(path.read_bytes())
        report['code_sha256']=h.hexdigest()
        pipeline=ROOT/'reports/recommendations/hf-pipeline-status.json'
        report['discovery_pipeline']=json.loads(pipeline.read_text()) if pipeline.exists() else None
        from zoneinfo import ZoneInfo
        report['generated_local']=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
        (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env=Environment(loader=FileSystemLoader(ROOT/'scripts'),autoescape=select_autoescape(['html']))
        html=env.get_template('recommendation_dashboard.html').render(report=report)
        (args.output/'index.html').write_text(html)
        print(json.dumps({'tests':tests['counts'],'pytest_exit':proc.returncode,
                         'dashboard':str(args.output/'index.html'),'live':{k:v.get('status') for k,v in report['live'].items() if isinstance(v,dict)}}))
        return 1 if proc.returncode or any(c['status']=='fail' for c in report['structural']) else 0

if __name__=='__main__':raise SystemExit(main())
