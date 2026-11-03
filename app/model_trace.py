"""
Run the search models' forward passes without holding Python's GIL
(switch SEARCH_MODEL_TRACE, read once when each model loads).

Why: every PyTorch op called from Python releases the GIL and must take it
back before the next op. While any other thread in the process runs pure
Python (an arXiv HTML parse, a template render), each take-back waits up to
sys.getswitchinterval() (5 ms by default). BGE-M3 runs ~800 ops per query and
the cross-encoder ~1,500 per batch, so a search stalls for seconds while the
CPU sits idle. Measured on the laptop with one 2 MB page parsing in a loop
(2026-10-30, torch on 2 threads, one inference worker as on the Space):

                                    encode     cross-encoder (10 pairs)
    alone                            45 ms      138 ms
    + parse loop, today           5,172 ms    7,987 ms
    + parse loop, traced, 5 ms      123 ms      174 ms
    + parse loop, traced, 0.5 ms     50 ms      124 ms

A model traced with torch.jit runs its whole graph in C++ with the GIL
released, so it takes the GIL back a handful of times instead of hundreds.
compute_pool.configure_gil() shortens the wait for the few that remain
(GIL_SWITCH_INTERVAL_MS).

Same weights, same kernels: traced outputs were bit-identical to eager ones on
400 real search pairs and 40 query encodes. DeBERTa's masked softmax is a
custom autograd Function, which a trace keeps as a Python call (one per layer,
each needing the GIL); _PlainXSoftmax is the same arithmetic as plain tensor
ops for the trace only.

Each trace is checked against the eager model on probe inputs of several
lengths and batch sizes before it is used. A trace that fails or differs is
dropped and the model keeps running eagerly, so the worst case is today's
speed. Modes: off (eager), shadow (trace and check, report, keep eager),
on (trace, check, use).

Tracing prints TracerWarnings that are safe here and are silenced: DeBERTa
compares key and query lengths (always equal in self-attention, so the
recorded constant is right) and the softmax mask value is a dtype constant.
"""
from __future__ import annotations

import time
import types
import warnings

from app import features

SWITCH = "SEARCH_MODEL_TRACE"
MAX_DIFF = 1e-5          # traced vs eager, on probe inputs; measured 0.0

_status: dict[str, dict] = {}


def status() -> dict:
    """Per model: mode, whether the traced graph is in use, and why not."""
    return dict(_status)


def _report(name: str, **info) -> None:
    _status[name] = {"mode": features.mode(SWITCH), **info}
    print(f"[model_trace] {name}: {info}")


class _PlainXSoftmax:
    """transformers' DeBERTa-v2 XSoftmax.forward as plain tensor ops (inference only)."""

    @staticmethod
    def apply(input, mask, dim):
        import torch
        rmask = ~(mask.to(torch.bool))
        out = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        out = torch.softmax(out, dim)
        return out.masked_fill(rmask, 0)


def trace_encoder(bge) -> bool:
    """Swap BGE-M3's transformer body for a traced copy. FlagEmbedding keeps doing
    tokenisation, pooling and the sparse weights, so its outputs are unchanged.
    Returns True when the traced copy is in use."""
    mode = features.mode(SWITCH)
    if mode == "off":
        return False
    t0 = time.perf_counter()
    try:
        import torch

        body = bge.model.model

        class Body(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                return self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]

        class Traced(torch.nn.Module):
            """Stands in for the HF model inside FlagEmbedding's BGEM3ForInference."""

            def __init__(self, traced):
                super().__init__()
                self.traced = traced

            def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
                return types.SimpleNamespace(last_hidden_state=self.traced(input_ids, attention_mask))

        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            ex = bge.tokenizer(["example search query"], return_tensors="pt")
            traced = torch.jit.trace(Body(body).eval(), (ex["input_ids"], ex["attention_mask"]), check_trace=False)
            diff = 0.0
            probes = ["bert", "low rank adaptation of large language models",
                      "how do graph neural networks behave on heterophilous graphs " * 8]
            for text in probes:
                tok = bge.tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
                a = body(**tok, return_dict=True).last_hidden_state
                b = traced(tok["input_ids"], tok["attention_mask"])
                diff = max(diff, float((a - b).abs().max()))
        if diff > MAX_DIFF:
            _report("encoder", traced=False, reason=f"probe max diff {diff:.1e}")
            return False
        if mode == "on":
            bge.model.model = Traced(traced)
        _report("encoder", traced=mode == "on", probe_max_diff=diff,
                trace_ms=int((time.perf_counter() - t0) * 1000))
        return mode == "on"
    except Exception as e:
        _report("encoder", traced=False, reason=f"{type(e).__name__}: {str(e)[:200]}")
        return False


def trace_cross_encoder(ce) -> bool:
    """Swap the sentence-transformers CrossEncoder's HF classifier for a traced
    copy. Returns True when the traced copy is in use."""
    mode = features.mode(SWITCH)
    if mode == "off":
        return False
    t0 = time.perf_counter()
    try:
        import torch

        holder = ce[0]
        clf = holder.model
        tokenizer = ce.tokenizer
        uses_types = "token_type_ids" in tokenizer(["q"], ["d"], return_tensors="pt")

        class Body(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, token_type_ids):
                kw = {"token_type_ids": token_type_ids} if uses_types else {}
                return self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False, **kw)[0]

        class Traced(torch.nn.Module):
            def __init__(self, traced, config):
                super().__init__()
                self.traced = traced
                self.config = config

            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, return_dict=True, **kwargs):
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                return types.SimpleNamespace(logits=self.traced(input_ids, attention_mask, token_type_ids))

        def batch(pairs):
            enc = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True,
                            max_length=getattr(ce, "max_length", None) or 512, return_tensors="pt")
            return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))

        deberta = None
        if type(clf).__module__.endswith("modeling_deberta_v2"):
            from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            example = batch([["query", "Title: a\nAbstract: b c d"], ["query", "Title: e\nAbstract: f"]])
            original = deberta.XSoftmax if deberta else None
            if deberta:
                deberta.XSoftmax = _PlainXSoftmax
            try:
                traced = torch.jit.trace(Body(clf).eval(), example, check_trace=False)
            finally:
                if deberta:
                    deberta.XSoftmax = original
            python_ops = sum(1 for n in traced.inlined_graph.nodes() if n.kind() == "prim::PythonOp")
            long = "Title: Sentence embeddings\nAbstract: " + "BERT and RoBERTa set a new state of the art. " * 30
            diff = 0.0
            for pairs in ([["bert paper", "Title: BERT\nAbstract: We introduce BERT."]],
                          [["bert paper", long], ["bert paper", "Title: x\nAbstract: y"], ["graph networks", long[:300]]]):
                ids, mask, types_ = batch(pairs)
                a = clf(input_ids=ids, attention_mask=mask, return_dict=True,
                        **({"token_type_ids": types_} if uses_types else {})).logits
                b = traced(ids, mask, types_)
                diff = max(diff, float((a - b).abs().max()))
        if diff > MAX_DIFF:
            _report("cross_encoder", traced=False, reason=f"probe max diff {diff:.1e}")
            return False
        if mode == "on":
            holder.model = Traced(traced, clf.config)
        _report("cross_encoder", traced=mode == "on", probe_max_diff=diff, python_ops=python_ops,
                trace_ms=int((time.perf_counter() - t0) * 1000))
        return mode == "on"
    except Exception as e:
        _report("cross_encoder", traced=False, reason=f"{type(e).__name__}: {str(e)[:200]}")
        return False
