"""
Quick health-check: ping Qdrant Cloud and Zilliz Cloud.
Prints status, collection info, and point counts.
"""
import os, sys, time

# Ensure project root is on sys.path so `app.config` resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app import config

DIVIDER = "─" * 60

# ── Qdrant ───────────────────────────────────────────────────────────────────
def ping_qdrant():
    print(f"\n{'═'*60}")
    print("  QDRANT CLOUD HEALTH CHECK")
    print(f"{'═'*60}")
    print(f"  URL        : {config.QDRANT_URL}")
    print(f"  Collection : {config.QDRANT_COLLECTION}")
    print(DIVIDER)

    try:
        from qdrant_client import QdrantClient
        t0 = time.perf_counter()
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=15,
            check_compatibility=False,
        )
        info = client.get_collection(config.QDRANT_COLLECTION)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"  ✅ STATUS  : ACTIVE")
        print(f"  Latency    : {latency_ms:.0f} ms")
        print(f"  Points     : {info.points_count}")
        print(f"  Vectors    : {info.vectors_count}")
        print(f"  Status     : {info.status}")
        print(f"  Segments   : {info.segments_count}")
        dim = None
        if info.config and info.config.params and info.config.params.vectors:
            vp = info.config.params.vectors
            if hasattr(vp, 'size'):
                dim = vp.size
            elif isinstance(vp, dict):
                for k, v in vp.items():
                    dim = getattr(v, 'size', '?')
                    break
        print(f"  Vector dim : {dim}")
    except Exception as e:
        print(f"  ❌ STATUS  : UNREACHABLE")
        print(f"  Error      : {e}")


# ── Zilliz ───────────────────────────────────────────────────────────────────
def ping_zilliz():
    print(f"\n{'═'*60}")
    print("  ZILLIZ CLOUD HEALTH CHECK")
    print(f"{'═'*60}")
    print(f"  URI        : {config.ZILLIZ_URI}")
    print(f"  Collection : {config.ZILLIZ_COLLECTION}")
    print(DIVIDER)

    try:
        from pymilvus import MilvusClient
        t0 = time.perf_counter()
        client = MilvusClient(
            uri=config.ZILLIZ_URI,
            token=config.ZILLIZ_TOKEN,
        )
        # List collections to verify connectivity
        collections = client.list_collections()
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"  ✅ STATUS  : ACTIVE")
        print(f"  Latency    : {latency_ms:.0f} ms")
        print(f"  Collections: {collections}")

        # Get stats for target collection
        if config.ZILLIZ_COLLECTION in collections:
            stats = client.get_collection_stats(config.ZILLIZ_COLLECTION)
            row_count = stats.get("row_count", "?")
            print(f"  Row count  : {row_count}")

            desc = client.describe_collection(config.ZILLIZ_COLLECTION)
            print(f"  Fields     : {[f['name'] for f in desc.get('fields', [])]}")
        else:
            print(f"  ⚠️  Collection '{config.ZILLIZ_COLLECTION}' NOT found")

    except Exception as e:
        print(f"  ❌ STATUS  : UNREACHABLE")
        print(f"  Error      : {e}")


if __name__ == "__main__":
    ping_qdrant()
    ping_zilliz()
    print(f"\n{'═'*60}")
    print("  DONE")
    print(f"{'═'*60}\n")
