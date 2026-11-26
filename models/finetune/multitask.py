"""Shared multilingual encoder + three heads (Task A, B, C). Plugs into common.stress via make_fit_predict.

    from models.finetune.multitask import FTConfig, MultiTaskFT, make_fit_predict
    report, oof = run_stress(make_fit_predict(FTConfig()), train, repeats=False)

Determinism: seeds are reset at the start of every fit; data order uses a seeded generator.
On CPU runs are bit-reproducible; on GPU, cuDNN kernels can add tiny run-to-run noise.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data_loader import split_turns
from common.labels import N_A, N_B, N_C, TECHNIQUES
from common.seed import set_seed


@dataclass
class FTConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    text_view: str = "raw_text"          # tokenizer handles case; normalized_text is also possible
    prefix: str = "query: "              # e5 models expect a prefix
    max_len: int = 128
    epochs: int = 15
    batch_size: int = 16
    lr_encoder: float = 5e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    freeze_embeddings: bool = True       # 96M of 118M params are the 250k-token embedding table
    loss_weights: tuple = (1.0, 0.5, 0.5)  # A, B, C
    class_weighted: bool = True
    grad_clip: float = 1.0
    eval_batch_size: int = 64
    seed: int = 42
    fp16: bool = True                     # only used on CUDA
    augment: str = "none"                 # "none" | "corrupt"
    augment_copies: int = 2
    local_files_only: bool = False
    extra: dict = field(default_factory=dict)


def build_text(turns, raw: str, cfg: FTConfig, sep: str) -> str:
    """Conversations keep speaker tags and turn order; single messages are used as-is."""
    if len(turns) > 1:
        body = f" {sep} ".join(f"{s}: {t}" if s else t for s, t in turns)
    else:
        body = raw
    return cfg.prefix + body


def load_backbone(cfg: FTConfig):
    """Returns (tokenizer, encoder). Tests monkeypatch this to use a tiny offline model."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
    enc = AutoModel.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
    return tok, enc


class MultiTaskNet(nn.Module):
    def __init__(self, encoder, dropout: float):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Linear(h, N_A)
        self.head_b = nn.Linear(h, N_B)
        self.head_c = nn.Linear(h, N_C)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        z = self.drop(pooled)
        return self.head_a(z), self.head_b(z), self.head_c(z)


def _class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=k).astype(float)
    w = np.where(counts > 0, (len(y) / (k * np.maximum(counts, 1))) ** 0.5, 0.0)  # sqrt-balanced
    return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)


def _pos_weight(Y: np.ndarray) -> torch.Tensor:
    pos = Y.sum(0).astype(float)
    neg = len(Y) - pos
    return torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, 5.0), dtype=torch.float32)


class MultiTaskFT:
    name = "ft_e5small_mt"

    def __init__(self, cfg: FTConfig | None = None, device: str | None = None):
        self.cfg = cfg or FTConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fit_seconds = None

    # ---------- data ----------
    def _texts(self, df: pd.DataFrame) -> list[str]:
        sep = getattr(self.tok, "sep_token", None) or "|"
        if self.cfg.text_view == "raw_text":
            return [build_text(t, r, self.cfg, sep) for t, r in zip(df["turns"], df["raw_text"])]
        return [self.cfg.prefix + t for t in df[self.cfg.text_view]]

    def _encode(self, texts):
        return self.tok(texts, truncation=True, max_length=self.cfg.max_len, padding=True, return_tensors="pt")

    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.augment == "none":
            return df
        if self.cfg.augment == "corrupt":
            from common.stress import corrupt_df

            copies = [df] + [corrupt_df(df, seed=1000 + i) for i in range(self.cfg.augment_copies)]
            out = pd.concat(copies, ignore_index=True)
            out["turns"] = [split_turns(r) for r in out["raw_text"]]  # rebuild turns from corrupted text
            return out
        raise ValueError(f"unknown augment {self.cfg.augment!r}")

    # ---------- training ----------
    def fit(self, df: pd.DataFrame):
        c = self.cfg
        set_seed(c.seed)
        t0 = time.time()
        self.tok, enc = load_backbone(c)
        self.net = MultiTaskNet(enc, c.dropout).to(self.device)
        df = self._augment(df)
        texts = self._texts(df)
        yA = df["label"].to_numpy(int)
        yB = df[TECHNIQUES].to_numpy(np.float32)
        yC = df["action"].to_numpy(int)

        wA = _class_weights(yA, N_A).to(self.device) if c.class_weighted else None
        wC = _class_weights(yC, N_C).to(self.device) if c.class_weighted else None
        pw = _pos_weight(yB).to(self.device)

        if c.freeze_embeddings:
            for p in self.net.encoder.embeddings.parameters():
                p.requires_grad = False
        enc_params = [p for p in self.net.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in self.net.named_parameters() if not n.startswith("encoder.")]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": c.lr_encoder},
                                 {"params": head_params, "lr": c.lr_head}], weight_decay=c.weight_decay)
        n = len(texts)
        steps_per_epoch = math.ceil(n / c.batch_size)
        total = steps_per_epoch * c.epochs
        warm = max(1, int(c.warmup_ratio * total))
        def lr_lambda(s):  # linear warmup, then linear decay to 0
            return (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        use_amp = c.fp16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        gen = torch.Generator().manual_seed(c.seed)
        wl = c.loss_weights
        self.net.train()
        for _ in range(c.epochs):
            order = torch.randperm(n, generator=gen).numpy()
            for i in range(0, n, c.batch_size):
                idx = order[i:i + c.batch_size]
                b = self._encode([texts[j] for j in idx])
                ids, am = b["input_ids"].to(self.device), b["attention_mask"].to(self.device)
                ta = torch.tensor(yA[idx], device=self.device)
                tb = torch.tensor(yB[idx], device=self.device)
                tc = torch.tensor(yC[idx], device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    la, lb, lc = self.net(ids, am)
                la, lb, lc = la.float(), lb.float(), lc.float()
                loss = (wl[0] * F.cross_entropy(la, ta, weight=wA)
                        + wl[1] * F.binary_cross_entropy_with_logits(lb, tb, pos_weight=pw)
                        + wl[2] * F.cross_entropy(lc, tc, weight=wC))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
        self.net.eval()
        self.fit_seconds = round(time.time() - t0, 1)
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict(self, df: pd.DataFrame, return_logits: bool = False) -> dict:
        c = self.cfg
        texts = self._texts(df)
        # sort by length so batches pad less (big win at 500k rows); restore order afterwards
        order = np.argsort([len(t) for t in texts], kind="stable")
        outs = {"A": [], "B": [], "C": []}
        use_amp = c.fp16 and self.device.type == "cuda"
        for i in range(0, len(texts), c.eval_batch_size):
            idx = order[i:i + c.eval_batch_size]
            b = self._encode([texts[j] for j in idx])
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                la, lb, lc = self.net(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
            outs["A"].append(la.float().cpu()); outs["B"].append(lb.float().cpu()); outs["C"].append(lc.float().cpu())
        if not texts:
            return {"A": np.zeros((0, N_A)), "B": np.zeros((0, N_B)), "C": np.zeros((0, N_C))}
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        L = {k: torch.cat(v)[torch.from_numpy(inv)] for k, v in outs.items()}
        res = {"A": F.softmax(L["A"], -1).double().numpy(), "B": torch.sigmoid(L["B"]).double().numpy(),
               "C": F.softmax(L["C"], -1).double().numpy()}
        if return_logits:
            res["logits_A"] = L["A"].double().numpy()
        return res

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["loss_weights"] = list(d["loss_weights"])
        return {"model": self.name, **d}


def make_fit_predict(cfg: FTConfig | None = None, timings: list | None = None):
    """Adapter for common.stress. Optionally records per-fold fit times."""
    def fit_predict(train_df, valid_df):
        m = MultiTaskFT(cfg).fit(train_df)
        if timings is not None:
            timings.append(m.fit_seconds)
        return m.predict(valid_df)
    return fit_predict
