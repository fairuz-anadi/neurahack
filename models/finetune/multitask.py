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
    synthetic_path: str | None = None     # labeled CSV (id,text,label,techniques,action) added to every fit
    synthetic_max: int | None = None      # cap on synthetic rows per fit (seeded sample)
    real_repeat: int = 1                  # repeat real rows so synthetic data does not swamp them
    pseudo_path: str | None = None        # teacher pseudo-labels for test ids (Kaggle only)
    pseudo_max: int | None = None         # cap on pseudo-labeled rows per fit (seeded sample)
    pseudo_min_conf: float = 0.0          # drop teacher labels below this label confidence
    external_path: str | None = None      # public external data with full or partial (masked) labels
    external_max: int | None = None       # cap on external rows per fit (seeded sample)
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


def _target_masks(df: pd.DataFrame, yA: np.ndarray):
    """Task A allowed-class mask (n,6) and known-flags for Task B / C. Rows without explicit masks are fully
    labeled (real, synthetic, pseudo); external partial rows carry a_mask / b_known / c_known columns."""
    n = len(df)
    amask = np.zeros((n, N_A), dtype=np.float32)
    col = df["a_mask"] if "a_mask" in df.columns else pd.Series([None] * n)
    for i, (y, mk) in enumerate(zip(yA, col)):
        if isinstance(mk, (list, tuple, np.ndarray)):
            amask[i] = np.asarray(mk, dtype=np.float32)
        else:
            amask[i, int(y)] = 1.0
    bk = df["b_known"].fillna(True).astype(bool).to_numpy() if "b_known" in df.columns else np.ones(n, bool)
    ck = df["c_known"].fillna(True).astype(bool).to_numpy() if "c_known" in df.columns else np.ones(n, bool)
    return amask, bk, ck


def _masked_loss(la, lb, lc, ta, tb, tc, amask, bk, ck, wA, wC, pw, wl, device):
    """Task A: -log P(true class in allowed set) (equals weighted CE for fully labeled rows).
    Task B / C: only rows whose labels are known contribute."""
    m = torch.as_tensor(amask, device=device)
    logp = F.log_softmax(la, -1)
    li = -torch.logsumexp(logp + torch.log(m.clamp_min(1e-12)), -1)
    if wA is not None:
        wi = torch.where(ta >= 0, wA[ta.clamp(min=0)], torch.ones_like(li))
    else:
        wi = torch.ones_like(li)
    loss = wl[0] * (wi * li).sum() / wi.sum()
    bkt = torch.as_tensor(bk, device=device)
    if bkt.any():
        per = F.binary_cross_entropy_with_logits(lb, tb, pos_weight=pw, reduction="none").mean(1)
        loss = loss + wl[1] * per[bkt].mean()
    ckt = torch.as_tensor(ck, device=device)
    if ckt.any():
        loss = loss + wl[2] * F.cross_entropy(lc[ckt], tc[ckt].clamp(min=0), weight=wC)
    return loss


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

    def _add_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        parts = [df] * max(1, c.real_repeat)
        if c.synthetic_path:
            from models.finetune.synthetic import load_synthetic

            syn = load_synthetic(c.synthetic_path)
            if c.synthetic_max and len(syn) > c.synthetic_max:
                syn = syn.sample(c.synthetic_max, random_state=c.seed)
            parts.append(syn)
        if c.pseudo_path:
            from models.finetune.synthetic import load_pseudo

            ps = load_pseudo(c.pseudo_path, c.pseudo_min_conf)
            if c.pseudo_max and len(ps) > c.pseudo_max:
                ps = ps.sample(c.pseudo_max, random_state=c.seed)
            parts.append(ps)
        if c.external_path:
            from models.finetune.synthetic import load_external

            ex = load_external(c.external_path)
            if c.external_max and len(ex) > c.external_max:
                ex = ex.sample(c.external_max, random_state=c.seed)
            parts.append(ex)
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else df

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
        df = self._augment(self._add_synthetic(df))
        self.n_train_rows = len(df)
        texts = self._texts(df)
        yA = df["label"].to_numpy(int)
        yB = df[TECHNIQUES].to_numpy(np.float32)
        yC = df["action"].to_numpy(int)
        amask, bk, ck = _target_masks(df, yA)
        partial = bool((yA < 0).any() or (~bk).any() or (~ck).any())
        knownA = yA >= 0

        wA = _class_weights(yA[knownA], N_A).to(self.device) if c.class_weighted else None
        wC = _class_weights(yC[ck], N_C).to(self.device) if c.class_weighted else None
        pw = _pos_weight(yB[bk]).to(self.device)

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
                if not partial:
                    loss = (wl[0] * F.cross_entropy(la, ta, weight=wA)
                            + wl[1] * F.binary_cross_entropy_with_logits(lb, tb, pos_weight=pw)
                            + wl[2] * F.cross_entropy(lc, tc, weight=wC))
                else:
                    loss = _masked_loss(la, lb, lc, ta, tb, tc, amask[idx], bk[idx], ck[idx], wA, wC, pw, wl,
                                        self.device)
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
    def predict(self, df: pd.DataFrame, return_logits: bool = False, progress_every: int = 50_000) -> dict:
        """Batched inference. Rows are sorted by length to reduce padding and restored afterwards.
        On CUDA out-of-memory the batch size is halved and the batch retried."""
        c = self.cfg
        texts = self._texts(df)
        if not texts:
            return {"A": np.zeros((0, N_A)), "B": np.zeros((0, N_B)), "C": np.zeros((0, N_C))}
        order = np.argsort([len(t) for t in texts], kind="stable")
        outs = {"A": [], "B": [], "C": []}
        use_amp = c.fp16 and self.device.type == "cuda"
        bs, i, next_report, t0 = c.eval_batch_size, 0, progress_every, time.time()
        while i < len(texts):
            idx = order[i:i + bs]
            try:
                b = self._encode([texts[j] for j in idx])
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    la, lb, lc = self.net(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                print(f"[predict] CUDA OOM -> eval batch size {bs}", flush=True)
                continue
            outs["A"].append(la.float().cpu()); outs["B"].append(lb.float().cpu()); outs["C"].append(lc.float().cpu())
            i += len(idx)
            if len(texts) >= 2 * progress_every and i >= next_report:
                print(f"[predict] {i}/{len(texts)} rows, {i / (time.time() - t0):.0f} rows/s", flush=True)
                next_report += progress_every
        self.last_eval_batch_size = bs
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        L = {k: torch.cat(v)[torch.from_numpy(inv)] for k, v in outs.items()}
        res = {"A": F.softmax(L["A"], -1).double().numpy(), "B": torch.sigmoid(L["B"]).double().numpy(),
               "C": F.softmax(L["C"], -1).double().numpy()}
        if return_logits:
            res["logits_A"] = L["A"].double().numpy()
        return res

    def release(self):
        """Free the network (and GPU memory) once predictions are taken."""
        import gc

        self.peak_gpu_mb = (round(torch.cuda.max_memory_allocated() / 2**20) if self.device.type == "cuda" else None)
        self.net = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def config_dict(self) -> dict:
        d = asdict(self.cfg)
        d["loss_weights"] = list(d["loss_weights"])
        return {"model": self.name, **d}


def make_fit_predict(cfg: FTConfig | None = None, timings: list | None = None):
    """Adapter for common.stress. Optionally records per-fold fit times."""
    def fit_predict(train_df, valid_df):
        m = MultiTaskFT(cfg).fit(train_df)
        pred = m.predict(valid_df)
        m.release()
        if timings is not None:
            timings.append(m.fit_seconds)
        return pred
    return fit_predict
