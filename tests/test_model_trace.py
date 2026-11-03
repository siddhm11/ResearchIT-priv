"""app/model_trace.py: traced search models must give the eager models' outputs
exactly, keep the eager model when they would not, and leave no Python call in
the cross-encoder's graph (each one needs the GIL)."""
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app import compute_pool, config, model_trace  # noqa: E402


class _Tok:
    """Character-level stand-in for a HF tokenizer: enough for the trace probes."""

    def __init__(self, limit=60):
        self.limit = limit

    def _ids(self, text):
        return [2] + [3 + (ord(c) % 90) for c in text][: self.limit - 2] + [2]

    def _batch(self, rows, pairs):
        n = max(len(r) for r in rows)
        out = {"input_ids": torch.tensor([r + [1] * (n - len(r)) for r in rows]),
               "attention_mask": torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows])}
        if pairs:
            out["token_type_ids"] = torch.tensor([[0] * a + [1] * (len(r) - a) + [0] * (n - len(r))
                                                  for r, a in pairs])
        return out

    def __call__(self, texts, others=None, **kwargs):
        if others is None:
            return self._batch([self._ids(t) for t in texts], None)
        rows, pairs = [], []
        for a, b in zip(texts, others):
            ia, ib = self._ids(a), self._ids(b)[1:]
            row = (ia + ib)[: self.limit]
            rows.append(row)
            pairs.append((row, min(len(ia), len(row))))
        return self._batch(rows, pairs)


class _Body:
    def __init__(self, model):
        self.model = model


class _Encoder:
    """Shape of FlagEmbedding's BGEM3FlagModel that model_trace touches."""

    def __init__(self):
        cfg = transformers.XLMRobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                                            num_attention_heads=2, intermediate_size=37,
                                            max_position_embeddings=80, pad_token_id=1)
        torch.manual_seed(0)
        self.model = _Body(transformers.XLMRobertaModel(cfg).eval())
        self.tokenizer = _Tok()


class _CrossEncoder:
    """Shape of sentence-transformers' CrossEncoder that model_trace touches."""

    def __init__(self):
        cfg = transformers.DebertaV2Config(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                                           num_attention_heads=2, intermediate_size=37,
                                           max_position_embeddings=80, relative_attention=True,
                                           position_buckets=16, pos_att_type=["p2c", "c2p"],
                                           type_vocab_size=2, num_labels=1)
        torch.manual_seed(0)
        self.holder = _Body(transformers.DebertaV2ForSequenceClassification(cfg).eval())
        self.tokenizer = _Tok()
        self.max_length = 60

    def __getitem__(self, i):
        return self.holder


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value):
        monkeypatch.setenv(model_trace.SWITCH, value)
    return set_mode


def _encode(bge, text):
    tok = bge.tokenizer([text])
    with torch.inference_mode():
        return bge.model.model(**tok, return_dict=True).last_hidden_state


def _score(ce, pairs):
    tok = ce.tokenizer([p[0] for p in pairs], [p[1] for p in pairs])
    with torch.inference_mode():
        return ce[0].model(**tok, return_dict=True).logits


TEXTS = ["bert", "low rank adaptation", "graph neural networks on heterophilous graphs " * 3]
PAIRS = [[["bert paper", "Title: BERT\nAbstract: deep bidirectional transformers"]],
         [["q", "short"], ["a longer query", "Title: t\nAbstract: " + "words " * 20], ["x", "y"]]]


def test_off_leaves_models_alone(mode):
    mode("off")
    bge, ce = _Encoder(), _CrossEncoder()
    body, clf = bge.model.model, ce[0].model
    assert model_trace.trace_encoder(bge) is False
    assert model_trace.trace_cross_encoder(ce) is False
    assert bge.model.model is body and ce[0].model is clf


def test_encoder_trace_matches_eager_at_every_length(mode):
    mode("on")
    bge = _Encoder()
    eager = [_encode(bge, t) for t in TEXTS]
    assert model_trace.trace_encoder(bge) is True
    assert model_trace.status()["encoder"]["traced"] is True
    for t, e in zip(TEXTS, eager):
        assert torch.equal(_encode(bge, t), e)


def test_cross_encoder_trace_matches_eager_and_has_no_python_ops(mode):
    mode("on")
    ce = _CrossEncoder()
    eager = [_score(ce, p) for p in PAIRS]
    assert model_trace.trace_cross_encoder(ce) is True
    assert model_trace.status()["cross_encoder"]["python_ops"] == 0
    for p, e in zip(PAIRS, eager):
        assert torch.equal(_score(ce, p), e)


def test_deberta_softmax_is_restored_after_tracing(mode):
    from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta
    mode("on")
    original = deberta.XSoftmax
    model_trace.trace_cross_encoder(_CrossEncoder())
    assert deberta.XSoftmax is original


def test_shadow_checks_but_keeps_eager(mode):
    mode("shadow")
    bge, ce = _Encoder(), _CrossEncoder()
    body, clf = bge.model.model, ce[0].model
    assert model_trace.trace_encoder(bge) is False
    assert model_trace.trace_cross_encoder(ce) is False
    assert bge.model.model is body and ce[0].model is clf
    assert model_trace.status()["encoder"]["probe_max_diff"] == 0.0
    assert model_trace.status()["cross_encoder"]["mode"] == "shadow"


def test_a_trace_that_differs_is_not_used(mode, monkeypatch):
    mode("on")
    monkeypatch.setattr(model_trace, "MAX_DIFF", -1.0)
    bge = _Encoder()
    body = bge.model.model
    assert model_trace.trace_encoder(bge) is False
    assert bge.model.model is body
    assert "probe max diff" in model_trace.status()["encoder"]["reason"]


def test_a_trace_that_fails_is_not_used(mode):
    mode("on")
    bge = _Encoder()
    bge.tokenizer = None            # tracing raises
    body = bge.model.model
    assert model_trace.trace_encoder(bge) is False
    assert bge.model.model is body
    assert model_trace.status()["encoder"]["reason"].startswith("TypeError")


def test_switch_interval_follows_config(monkeypatch):
    before = sys.getswitchinterval()
    try:
        monkeypatch.setattr(config, "GIL_SWITCH_INTERVAL_MS", 0.5)
        assert compute_pool.configure_gil() == pytest.approx(0.5)
        assert compute_pool.stats()["gil_switch_interval_ms"] == pytest.approx(0.5)
        monkeypatch.setattr(config, "GIL_SWITCH_INTERVAL_MS", 0.0)
        assert compute_pool.configure_gil() == pytest.approx(0.5)   # 0 leaves it alone
    finally:
        sys.setswitchinterval(before)
