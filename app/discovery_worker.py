"""Scheduled-worker operations. External reads and local shadow storage only."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import time
import xml.etree.ElementTree as ET

import httpx

from app.discovery_store import DiscoveryStore, EMBEDDING_CONTRACT
from app.hf_papers_svc import fetch_snapshot


async def collect_once(store: DiscoveryStore, client: httpx.AsyncClient, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    try:
        snapshot = await fetch_snapshot(client, now=datetime.fromtimestamp(now, timezone.utc))
        if snapshot['input_count'] and not snapshot['papers']:
            raise ValueError('All source records rejected')
        await asyncio.to_thread(store.save_snapshot,snapshot,now)
        return {'status':'ok','accepted':len(snapshot['papers']),'rejected':snapshot['rejected']}
    except (httpx.HTTPError,ValueError) as exc:
        # Store type only: network exception strings can contain credentials/URLs.
        await asyncio.to_thread(store.record_failure,now,type(exc).__name__)
        return {'status':'failed','reason':type(exc).__name__}


async def fetch_abstract(client: httpx.AsyncClient, paper: dict) -> dict:
    from app.arxiv_svc import _parse_entry, _NS
    # One request, bounded by worker batch size and a 3.5s minimum inter-call pause.
    await asyncio.sleep(3.5)
    async with client.stream('GET','https://export.arxiv.org/api/query',
                             params={'id_list':paper['arxiv_id']},timeout=20) as r:
        r.raise_for_status()
        body=bytearray()
        async for chunk in r.aiter_bytes():
            body.extend(chunk)
            if len(body)>1_000_000:raise ValueError('Oversized arXiv metadata response')
    for entry in ET.fromstring(body).findall('atom:entry',_NS):
        meta=_parse_entry(entry)
        if meta['arxiv_id']==paper['arxiv_id'] and meta.get('abstract'):
            return {**paper,'abstract':meta['abstract'],'title':meta['title'],
                    'metadata_source':'arxiv'}
    raise ValueError('No matching arXiv metadata')


def encode_paper(paper: dict) -> list[float]:
    from app import config, embed_svc
    if config.BGE_M3_MODEL != 'BAAI/bge-m3':
        raise ValueError('Model differs from the shadow embedding contract')
    text=f"{paper['title'][:256]} {paper['abstract'][:1024]}"
    return embed_svc.encode_query(text)[0].tolist()


async def prepare_once(store: DiscoveryStore, client: httpx.AsyncClient, *,
                       now: float | None = None, limit: int = 10,
                       encoder=encode_paper, metadata_fetcher=fetch_abstract) -> dict:
    now=time.time() if now is None else now
    ready=failed=superseded=0
    pending = await asyncio.to_thread(store.pending,now,limit)
    if pending and encoder is encode_paper:
        # A broken shared runtime must not exhaust every paper's retry budget.
        from app import config, embed_svc
        try:
            if config.BGE_M3_MODEL != 'BAAI/bge-m3':
                raise ValueError('Incompatible embedding model')
            await asyncio.to_thread(embed_svc.get_model)
        except Exception as exc:
            return {'ready':0,'failed':0,'superseded':0,
                    'blocked':True,'reason':type(exc).__name__}
    for row in pending:
        paper=json.loads(row['paper'])
        try:
            if not paper.get('abstract','').strip():
                paper=await metadata_fetcher(client,paper)
            vector=await asyncio.to_thread(encoder,paper)
            success=await asyncio.to_thread(store.finish,row,paper,vector,now,EMBEDDING_CONTRACT)
            ready+=int(success);superseded+=int(not success)
        except Exception as exc:
            # Each candidate fails independently; three attempts then operator inspection.
            await asyncio.to_thread(store.fail_candidate,row,now,type(exc).__name__)
            failed+=1
    return {'ready':ready,'failed':failed,'superseded':superseded}
