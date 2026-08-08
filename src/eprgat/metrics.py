"""Evaluation metrics for extreme-imbalance event classification.

F1 is reported (as requested) but never alone: with ~1% positives, F1 at a
fixed 0.5 threshold is nearly meaningless, so we also report AUROC, AUPRC
(the headline metric for rare-event detection), precision/recall at the
validation-tuned threshold, MCC, and precision@K with K = number of actual
positives (the SOC "how much must I investigate" view).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


@dataclass
class Metrics:
    n: int
    n_pos: int
    pos_rate: float
    auroc: float
    auprc: float
    baseline_ap: float            # AUPRC of the always-positive predictor
    threshold: float
    f1: float
    precision: float
    recall: float
    f1_at_050: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision_at_k: float         # K = n_pos (investigation workload)
    lift_at_k: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def search_threshold(y: np.ndarray, proba: np.ndarray, policy: str = "f1",
                     min_recall: float = 0.90) -> float:
    """Choose the operating threshold on a (validation) set.

    policy='f1'     : threshold maximising F1 — the balanced operating point.
    policy='recall' : the HIGHEST threshold still achieving recall >= min_recall
      — the recall-first operating point for when missing a true positive is
      costlier than investigating a false one. Applied identically to every
      model being compared (RGAT and the MLP baseline alike).
    """
    if policy not in ("f1", "recall"):
        raise ValueError(f"unknown threshold policy '{policy}'")
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, proba)
    if policy == "recall":
        # sklearn: thresholds ascending, recall non-increasing along the
        # arrays, artificial (precision=1, recall=0) point appended last.
        rec = rec[:-1]
        ok = np.where(rec >= min_recall)[0]        # rec[0] == 1.0 always
        if len(ok) == 0:
            return float(thr[0])
        return float(thr[int(ok[-1])])             # highest thr meeting floor
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    f1 = f1[:-1]                                   # thr array is shorter by 1
    if len(f1) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def compute_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> Metrics:
    pred = (proba >= threshold).astype(np.int64)
    tn, fp, fn, tp = (confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
                      + [0, 0, 0, 0])[:4]
    pos_rate = float(y.mean()) if len(y) else 0.0
    auroc = float(roc_auc_score(y, proba)) if 0 < y.sum() < len(y) else float("nan")
    auprc = float(average_precision_score(y, proba)) if 0 < y.sum() < len(y) else float("nan")
    k = max(1, int(y.sum()))
    order = np.argsort(-proba)
    precision_at_k = float(y[order[:k]].mean())
    return Metrics(
        n=int(len(y)), n_pos=int(y.sum()), pos_rate=pos_rate,
        auroc=auroc, auprc=auprc, baseline_ap=pos_rate,
        threshold=float(threshold),
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1_at_050=float(f1_score(y, (proba >= 0.5).astype(int), zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        precision_at_k=precision_at_k,
        lift_at_k=float(precision_at_k / pos_rate) if pos_rate > 0 else float("nan"))


def format_metrics(m: Metrics, title: str = "") -> str:
    head = f"── {title} " if title else ""
    lines = [
        f"{head}{'─' * max(0, 66 - len(head))}",
        f"  events={m.n:,}  positives={m.n_pos:,}  pos_rate={100*m.pos_rate:.2f}%",
        f"  ranking:   AUROC={m.auroc:.4f}   AUPRC={m.auprc:.4f}   "
        f"(baseline AP={m.baseline_ap:.4f}, lift x{(m.auprc/max(m.baseline_ap,1e-12)):.1f})",
        f"  @thr={m.threshold:.3f}:  F1={m.f1:.4f}   P={m.precision:.4f}   "
        f"R={m.recall:.4f}   MCC={m.mcc:.4f}",
        f"  @thr=0.50:  F1={m.f1_at_050:.4f}",
        f"  confusion: TN={m.tn:,}  FP={m.fp:,}  FN={m.fn:,}  TP={m.tp:,}",
        f"  triage:    precision@K(=n_pos)={m.precision_at_k:.4f}  "
        f"(lift x{m.lift_at_k:.1f} over random)",
        "─" * 66,
    ]
    return "\n".join(lines)
