"""Classical baseline: char_wb + word TF-IDF -> Logistic Regression for Tasks A, B (6 binary), C.

CPU only, deterministic. Vectorizers are fit inside each training fold (no leakage).
Handles classes / techniques that are absent from a training fold:
  - multiclass: missing classes get probability 0, then rows are renormalized
  - binary technique with a single class in the fold: constant probability = fold prevalence
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common.labels import N_A, N_B, N_C, TECHNIQUES


@dataclass
class TfidfLRConfig:
    text_view: str = "normalized_text"
    char_ngram: tuple = (2, 5)
    word_ngram: tuple = (1, 2)
    char_max_features: int = 200_000
    word_max_features: int = 50_000
    C: float = 4.0
    class_weight: str | None = "balanced"
    max_iter: int = 2000
    seed: int = 42
    predict_chunk: int = 50_000


class TfidfLR:
    name = "tfidf_lr"

    def __init__(self, cfg: TfidfLRConfig | None = None):
        self.cfg = cfg or TfidfLRConfig()

    # ---- features ----
    def _fit_vec(self, texts):
        c = self.cfg
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c.char_ngram), sublinear_tf=True,
                                  min_df=1, max_features=c.char_max_features, dtype=np.float32)
        self.vw = TfidfVectorizer(analyzer="word", ngram_range=tuple(c.word_ngram), sublinear_tf=True,
                                  min_df=1, max_features=c.word_max_features, token_pattern=r"(?u)\b\w+\b",
                                  dtype=np.float32)
        return hstack([self.vc.fit_transform(texts), self.vw.fit_transform(texts)]).tocsr()

    def _vec(self, texts):
        return hstack([self.vc.transform(texts), self.vw.transform(texts)]).tocsr()

    def _lr(self):
        c = self.cfg
        return LogisticRegression(C=c.C, class_weight=c.class_weight, max_iter=c.max_iter, random_state=c.seed)

    # ---- training ----
    def fit(self, df: pd.DataFrame):
        X = self._fit_vec(df[self.cfg.text_view].tolist())
        self.mA = self._lr().fit(X, df["label"].to_numpy())
        self.mC = self._lr().fit(X, df["action"].to_numpy())
        self.mB = []
        for t in TECHNIQUES:
            y = df[t].to_numpy()
            self.mB.append(self._lr().fit(X, y) if len(np.unique(y)) == 2 else float(y.mean()))
        return self

    @staticmethod
    def _full_proba(model, X, n_classes):
        P = np.zeros((X.shape[0], n_classes))
        P[:, model.classes_.astype(int)] = model.predict_proba(X)
        return P / P.sum(1, keepdims=True)

    def _predict_X(self, X) -> dict:
        B = np.column_stack([m.predict_proba(X)[:, 1] if not isinstance(m, float) else np.full(X.shape[0], m)
                             for m in self.mB])
        return {"A": self._full_proba(self.mA, X, N_A), "B": B, "C": self._full_proba(self.mC, X, N_C)}

    def predict(self, df: pd.DataFrame) -> dict:
        texts = df[self.cfg.text_view].tolist()
        parts = [self._predict_X(self._vec(texts[i:i + self.cfg.predict_chunk]))
                 for i in range(0, len(texts), self.cfg.predict_chunk)]
        if not parts:
            return {"A": np.zeros((0, N_A)), "B": np.zeros((0, N_B)), "C": np.zeros((0, N_C))}
        return {k: np.vstack([p[k] for p in parts]) for k in ("A", "B", "C")}

    def config_dict(self) -> dict:
        return {"model": self.name, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.cfg).items()}}


def make_fit_predict(cfg: TfidfLRConfig | None = None):
    """Adapter for common.stress.run_cv / run_stress."""
    def fit_predict(train_df, valid_df):
        return TfidfLR(cfg).fit(train_df).predict(valid_df)
    return fit_predict
