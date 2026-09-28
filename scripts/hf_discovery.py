#!/usr/bin/env python3
"""Collect/prepare Hugging Face candidates in local shadow storage; compare feeds.

Use --watch for the long-running scheduler, or invoke `collect` from cron.
One filesystem lock serializes collection/preparation on the same host.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


from app.discovery_runtime import worker_lock


async def cycle(store, prepare: bool, limit: int) -> dict:
    import httpx
    from app.discovery_worker import collect_once,prepare_once
    async with httpx.AsyncClient() as client:
        result={'collection':await collect_once(store,client)}
        if prepare:result['preparation']=await prepare_once(store,client,limit=limit)
    result['store']=await asyncio.to_thread(store.status,time.time())
    from datetime import datetime, timezone
    result['checked_at']=datetime.now(timezone.utc).isoformat()
    return result


def render_comparison(report: dict, output: Path) -> None:
    from jinja2 import Environment,FileSystemLoader,select_autoescape
    output.mkdir(parents=True,exist_ok=True)
    env=Environment(loader=FileSystemLoader(ROOT/'scripts'),autoescape=select_autoescape(['html']))
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'comparison.html').write_text(env.get_template('hf_comparison.html').render(report=report))


async def capture_baseline(seed_ids: list[str], output: Path) -> dict:
    """Run the existing feed builder for a scratch profile, without impressions."""
    import tempfile
    import uuid
    from datetime import datetime, timezone
    from app import config, db, qdrant_svc, user_state
    from app.recommend import profiles
    from app.routers import recommendations
    from app.discovery_store import EMBEDDING_CONTRACT
    original=(config.DB_PATH,db.DB_PATH)
    uid='shadow-'+uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix='researchit-baseline-') as temp:
            config.DB_PATH=db.DB_PATH=str(Path(temp)/'user.sqlite')
            await db.init_db()
            seed_ids=list(dict.fromkeys(seed_ids))
            vecs=await asyncio.wait_for(qdrant_svc.get_paper_vectors(seed_ids),timeout=45)
            usable=[aid for aid in seed_ids if aid in vecs]
            if len(usable)<5:raise ValueError('Need at least five available seed vectors')
            state=await user_state.ensure_loaded(uid)
            for aid in usable:
                state.add_positive(aid)
                await db.log_interaction(uid,aid,'save')
                await profiles.update_on_save(uid,vecs[aid])
            entry=await asyncio.wait_for(recommendations._build_feed(uid,state,uuid.uuid4().hex),timeout=90)
            if entry is None:raise ValueError('Baseline feed unavailable')
            papers,_=await recommendations._build_page(entry,set(usable))
            papers=papers[:10]
            if not papers:raise ValueError('Baseline page empty')
            vectors=await asyncio.wait_for(qdrant_svc.get_paper_vectors([p['arxiv_id'] for p in papers]),timeout=45)
            if any(p['arxiv_id'] not in vectors for p in papers):
                raise ValueError('Incomplete baseline candidate vectors')
            result={'name':'Scratch profile from supplied seeds','embedding_contract':EMBEDDING_CONTRACT,
                    'captured_at':datetime.now(timezone.utc).isoformat(),
                    'seed_ids':usable,'seed_vectors':[vecs[aid].tolist() for aid in usable],
                    'papers':[{'arxiv_id':p['arxiv_id'],'title':p['title'],
                               'published_at':p.get('published',''),'vector':vectors[p['arxiv_id']].tolist()}
                              for p in papers],
                    'excluded_ids':usable,
                    'provenance':'Actual feed builder, scratch seed profile. No delivered impressions.'}
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(result,allow_nan=False))
            return {'captured':len(papers),'usable_seeds':len(usable)}
    finally:
        config.DB_PATH,db.DB_PATH=original
        user_state._cache.pop(uid,None)


def synthetic_demo(output: Path) -> None:
    """Exercise collection-to-comparison using an explicitly synthetic source/model."""
    import tempfile
    from datetime import datetime, timezone
    import httpx
    from app.discovery_store import DiscoveryStore, EMBEDDING_CONTRACT
    from app.discovery_worker import collect_once, prepare_once
    from app.discovery_shadow import compare_feed
    # Use a microsecond-exact clock across JSON and SQLite eligibility checks.
    now=datetime.now(timezone.utc).timestamp()
    date=datetime.fromtimestamp(now-86400,timezone.utc).isoformat()
    def vector(topic):
        a=[0.0]*1024;a[topic]=1.0;return a
    baseline={'name':'Synthetic two-interest reader','embedding_contract':EMBEDDING_CONTRACT,
              'seed_ids':[f'seed-{i}' for i in range(10)],
              'seed_vectors':[vector(i%2) for i in range(10)],
              'papers':[{'arxiv_id':f'demo-baseline-{i}','title':f'Simulated interest {i%2+1}: baseline {i+1}',
                         'vector':vector(i%2),'published_at':date} for i in range(10)]}
    rows=[{'paper':{'id':f'2609.{i:05d}','title':f'Simulated fresh research {i+1}',
                     'summary':'Synthetic fixture, not a real paper.', 'publishedAt':date,
                     'upvotes':0}} for i in range(5)]
    async def run(store):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=rows))) as client:
            collected=await collect_once(store,client,now)
            prepared=await prepare_once(store,client,now=now,encoder=lambda p:vector(int(p['arxiv_id'][-1])%2))
        report=compare_feed(baseline,store.ready(now),now)
        report.update(synthetic=True,store=store.status(now),execution={'collection':collected,'preparation':prepared})
        render_comparison(report,output)
    with tempfile.TemporaryDirectory(prefix='researchit-shadow-demo-') as temp:
        asyncio.run(run(DiscoveryStore(Path(temp)/'source.sqlite')))


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,default=ROOT/'data/discovery.sqlite')
    sub=p.add_subparsers(dest='command',required=True)
    collect=sub.add_parser('collect');collect.add_argument('--prepare',action='store_true')
    collect.add_argument('--limit',type=int,default=10)
    collect.add_argument('--report',type=Path,help='Write collection and readiness status as JSON')
    collect.add_argument('--watch',action='store_true');collect.add_argument('--interval',type=int,default=21600)
    sub.add_parser('status')
    capture=sub.add_parser('capture');capture.add_argument('--seeds',nargs='+',required=True)
    capture.add_argument('--output',type=Path,required=True)
    demo=sub.add_parser('demo');demo.add_argument('--output',type=Path,default=ROOT/'reports/recommendations/shadow-demo')
    compare=sub.add_parser('compare');compare.add_argument('--baseline',type=Path,required=True)
    compare.add_argument('--output',type=Path,default=ROOT/'reports/recommendations/shadow')
    args=p.parse_args()
    if args.command=='capture':
        try:print(json.dumps(asyncio.run(capture_baseline(args.seeds,args.output))));return 0
        except Exception as exc:
            print(json.dumps({'status':'failed','reason':type(exc).__name__}));return 1
    if args.command=='demo':
        synthetic_demo(args.output);print(str(args.output/'comparison.html'));return 0
    from app.discovery_store import DiscoveryStore
    store=DiscoveryStore(args.store)
    if args.command=='status':print(json.dumps(store.status(time.time()),indent=2));return 0
    if args.command=='compare':
        from app.discovery_shadow import compare_feed
        now=time.time()
        baseline=json.loads(args.baseline.read_text())
        report=compare_feed(baseline,store.ready(now),now)
        report['store']=store.status(now)
        render_comparison(report,args.output)
        print(str(args.output/'comparison.html'));return 0
    if args.interval<300 or not 1<=args.limit<=100:p.error('interval >=300 seconds; limit 1..100')
    # Hold for the worker's entire life; no racing cron or second --watch process.
    with worker_lock(args.store.with_suffix('.worker.lock')):
        while True:
            result=asyncio.run(cycle(store,args.prepare,args.limit))
            if args.report:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                temporary=args.report.with_suffix('.tmp')
                temporary.write_text(json.dumps(result,indent=2))
                temporary.replace(args.report)
            print(json.dumps(result),flush=True)
            if not args.watch:
                return int(result['collection']['status']!='ok' or result.get('preparation',{}).get('failed',0)>0 or result.get('preparation',{}).get('blocked',False))
            time.sleep(args.interval)

if __name__=='__main__':raise SystemExit(main())
