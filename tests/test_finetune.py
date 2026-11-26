"""Fine-tuning tests with a tiny randomly initialised BERT + locally trained WordPiece tokenizer
(no downloads, runs in seconds). They check the contract, determinism and fold handling, not accuracy."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from common.data_loader import load_test, load_train  # noqa: E402
from common.prediction_io import validate_probs  # noqa: E402
from models.finetune import multitask as mt  # noqa: E402
from models.finetune.calibration import apply_temperature, crossfit_confidence, fit_temperature  # noqa: E402


_TOK = {}


def _tiny_tokenizer():
    """Built once: WordPiece training can order tied tokens differently between calls."""
    if "tok" in _TOK:
        return _TOK["tok"]
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = ["your order is out for delivery", "bkash pin pathan account block", "you won tk 5000 pay fee",
              "part-time job join telegram", "parcel held pay fee", "eid sale 30% off", "আপনার পার্সেল",
              "query: s r"]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=200, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    _TOK["tok"] = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                          cls_token="[CLS]", sep_token="[SEP]")
    return _TOK["tok"]


def _tiny_backbone(cfg):
    from transformers import BertConfig, BertModel

    tok = _tiny_tokenizer()
    torch.manual_seed(cfg.seed)
    enc = BertModel(BertConfig(vocab_size=tok.vocab_size + 4, hidden_size=32, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=64, max_position_embeddings=128))
    return tok, enc


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(mt, "load_backbone", _tiny_backbone)
    return mt.FTConfig(epochs=2, batch_size=4, eval_batch_size=5, max_len=32)


def test_contract_shapes_and_determinism(tiny_data, tiny):
    _, cfg = tiny_data
    tr, te = load_train(cfg), load_test(cfg)
    p1 = mt.MultiTaskFT(tiny, device="cpu").fit(tr).predict(te)
    p2 = mt.MultiTaskFT(tiny, device="cpu").fit(tr).predict(te)
    for t in "ABC":
        validate_probs(t, p1[t], len(te))
        np.testing.assert_allclose(p1[t], p2[t], atol=1e-6)


def test_predict_restores_row_order(tiny_data, tiny):
    _, cfg = tiny_data
    tr = load_train(cfg)
    m = mt.MultiTaskFT(tiny, device="cpu").fit(tr)
    full = m.predict(tr)
    rev = m.predict(tr.iloc[::-1])
    np.testing.assert_allclose(full["A"], rev["A"][::-1], atol=1e-5)


def test_augmentation_and_frozen_embeddings(tiny_data, tiny):
    _, cfg = tiny_data
    tr = load_train(cfg)
    c = mt.FTConfig(**{**tiny.__dict__, "augment": "corrupt", "augment_copies": 1})
    m = mt.MultiTaskFT(c, device="cpu")
    aug = None
    m.tok, _ = _tiny_backbone(c)
    aug = m._augment(tr)
    assert len(aug) == 2 * len(tr)
    assert (aug.raw_text.iloc[len(tr):].to_numpy() != tr.raw_text.to_numpy()).any()
    m.fit(tr)
    assert not any(p.requires_grad for p in m.net.encoder.embeddings.parameters())


def test_build_text_keeps_conversation_structure():
    c = mt.FTConfig()
    t = mt.build_text([("S", "hi"), ("R", "what?")], "S: hi ||| R: what?", c, "</s>")
    assert t == "query: S: hi </s> R: what?"
    assert mt.build_text([(None, "Your OTP")], "Your OTP", c, "</s>") == "query: Your OTP"


def test_calibration_crossfit():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 60)
    logits = rng.normal(size=(60, 6)) * 3
    logits[np.arange(60), y] += 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    T = fit_temperature(p, y)
    assert 0.2 <= T <= 5
    pa = apply_temperature(p, T)
    assert (pa.argmax(1) == p.argmax(1)).all()
    np.testing.assert_allclose(pa.sum(1), 1)
    conf, temps = crossfit_confidence(p, y, np.arange(60) % 3)
    assert conf.shape == (60,) and len(temps) == 3 and ((conf > 0) & (conf <= 1)).all()
