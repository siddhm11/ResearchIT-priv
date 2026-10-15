"""ResearchIT embedding bake-off: GPU encodes (private; reads a private dataset).

Same text formats and 512-token budget as the local runs:
  bge_m3_full    title + "\n\n" + abstract   (production separator), all 86,787 papers
  others         prefix + title + ". " + abstract, the 47,339-paper round-2 subset
sentence-transformers BGE-M3 was checked locally to equal FlagEmbedding (cosine 1.0).
"""
import glob, json, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5", "einops"], check=False)
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

src = glob.glob("/kaggle/input/**/texts.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(src)]
r2 = set(json.load(open(glob.glob("/kaggle/input/**/round2_ids.json", recursive=True)[0])))
print("papers", len(rows), "round2", len(r2), "gpu", torch.cuda.get_device_name(0), flush=True)
JOBS = [("gte_modernbert", "Alibaba-NLP/gte-modernbert-base", "", ". ", False, "r2"),
        ("nomic_v15", "nomic-ai/nomic-embed-text-v1.5", "search_document: ", ". ", True, "r2"),
        ("qwen3_06b", "Qwen/Qwen3-Embedding-0.6B", "", ". ", False, "r2"),
        ("bge_m3_full", "BAAI/bge-m3", "", "\n\n", False, "all")]
for name, hf, prefix, sep, trc, scope in JOBS:
    try:
        sel = [r for r in rows if scope == "all" or r["id"] in r2]
        texts = [prefix + r["title"] + sep + r["abstract"] for r in sel]
        m = SentenceTransformer(hf, device="cuda", trust_remote_code=trc, model_kwargs={"torch_dtype": torch.float16})
        m.max_seq_length = 512
        t = time.time()
        E = m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        np.save(f"/kaggle/working/vec_{name}.npy", E.astype(np.float16))
        json.dump([r["id"] for r in sel], open(f"/kaggle/working/ids_{name}.json", "w"))
        print(name, E.shape, int(time.time() - t), "s", flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:  # keep going so one model's failure doesn't lose the others
        print("FAILED", name, repr(e)[:500], flush=True)
