"""Misclassification audit tooling for Argus Vision (Phase 6).

Callable from 05_evaluation.ipynb to produce, from (y_true, y_pred, y_prob):
  * the confusion matrix (raw + row-normalized) -> CSV (+ PNG when matplotlib is present),
  * a per-class precision / recall / F1 / support / PR-AUC table -> CSV + console,
  * the top-N highest-confidence WRONG predictions -> CSV.

Dependency-light by design: sklearn / pandas / matplotlib are USED when available (Kaggle)
but every metric has a stdlib fallback, and CSVs are written with the stdlib ``csv`` module,
so the tooling runs end-to-end on synthetic data in a minimal environment (PNG is simply
skipped when matplotlib is absent). This is the same sklearn-optional pattern as splits.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Optional, Sequence

CLASS_NAMES_DEFAULT = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

try:
    import numpy as _np  # noqa: F401
    from sklearn.metrics import average_precision_score as _sk_ap

    _HAS_SK = True
except Exception:  # pragma: no cover - minimal env
    _HAS_SK = False

try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# --------------------------------------------------------------------------- #
# Core metrics (stdlib; sklearn used only for the official PR-AUC when present)
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> list[list[int]]:
    """Raw confusion matrix as an ``n_classes x n_classes`` list (rows=true, cols=pred)."""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def normalize_rows(cm: list[list[int]]) -> list[list[float]]:
    """Row-normalize a confusion matrix (each cell = fraction of that true class)."""
    out = []
    for row in cm:
        s = sum(row)
        out.append([(c / s) if s else 0.0 for c in row])
    return out


def _average_precision(y_bin: Sequence[int], scores: Sequence[float]) -> float:
    """One-vs-rest average precision. Uses sklearn when available, else a stdlib
    implementation (mean precision at the rank of each positive — equals
    sklearn.average_precision_score for tie-free continuous scores)."""
    pos = sum(1 for v in y_bin if v)
    if pos == 0:
        return float("nan")
    if _HAS_SK:
        try:
            return float(_sk_ap(list(y_bin), list(scores)))
        except Exception:
            pass
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, start=1):
        if y_bin[i]:
            tp += 1
            ap += tp / rank
    return ap / pos


def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[Sequence[float]],
    class_names: Sequence[str],
) -> list[dict]:
    """Per-class precision/recall/F1/support/PR-AUC plus macro & weighted rows."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
    rows: list[dict] = []
    total = len(y_true)
    for c in range(n):
        tp = cm[c][c]
        support = sum(cm[c])
        precision = tp / col_sums[c] if col_sums[c] else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        y_bin = [1 if int(t) == c else 0 for t in y_true]
        scores = [float(row[c]) for row in y_prob]
        pr_auc = _average_precision(y_bin, scores)
        rows.append({
            "class": class_names[c], "precision": precision, "recall": recall,
            "f1": f1, "support": support, "pr_auc": pr_auc,
        })

    def _avg(key: str, weighted: bool) -> float:
        vals = [(r[key], r["support"]) for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not vals:
            return float("nan")
        if weighted:
            sw = sum(w for _, w in vals)
            return sum(v * w for v, w in vals) / sw if sw else 0.0
        return sum(v for v, _ in vals) / len(vals)

    for label, weighted in (("macro_avg", False), ("weighted_avg", True)):
        rows.append({
            "class": label,
            "precision": _avg("precision", weighted),
            "recall": _avg("recall", weighted),
            "f1": _avg("f1", weighted),
            "support": total,
            "pr_auc": _avg("pr_auc", weighted),
        })
    return rows


def balanced_accuracy(rows: list[dict], class_names: Sequence[str]) -> float:
    """Mean per-class recall = balanced accuracy."""
    recalls = [r["recall"] for r in rows if r["class"] in class_names]
    return sum(recalls) / len(recalls) if recalls else 0.0


def top_confident_wrong(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[Sequence[float]],
    image_paths: Sequence[str],
    class_names: Sequence[str],
    lesion_ids: Optional[Sequence] = None,
    n: int = 100,
) -> list[dict]:
    """The N most-confident WRONG predictions, highest predicted-prob first."""
    wrong = []
    for i in range(len(y_true)):
        t, p = int(y_true[i]), int(y_pred[i])
        if t == p:
            continue
        conf = float(y_prob[i][p])
        wrong.append((conf, i))
    wrong.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for rank, (conf, i) in enumerate(wrong[:n], start=1):
        probs = {class_names[c]: round(float(y_prob[i][c]), 4) for c in range(len(class_names))}
        row = {
            "rank": rank,
            "image_path": image_paths[i],
            "true_class": class_names[int(y_true[i])],
            "predicted_class": class_names[int(y_pred[i])],
            "confidence": round(conf, 4),
            "all_class_probabilities": json.dumps(probs),
            "lesion_id": ("" if lesion_ids is None else lesion_ids[i]),
        }
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Writers + orchestrator
# --------------------------------------------------------------------------- #
def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _save_confusion_png(cm: list[list[int]], class_names: Sequence[str], path: str) -> bool:
    if not _HAS_MPL:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = normalize_rows(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (row-normalized)")
    for r in range(len(class_names)):
        for c in range(len(class_names)):
            ax.text(c, r, str(cm[r][c]), ha="center", va="center",
                    color="white" if norm[r][c] > 0.5 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


def run_audit(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[Sequence[float]],
    image_paths: Sequence[str],
    class_names: Sequence[str] = tuple(CLASS_NAMES_DEFAULT),
    lesion_ids: Optional[Sequence] = None,
    out_dir: str = ".",
    top_n: int = 100,
) -> dict:
    """Run the full misclassification audit; write CSVs (+PNG), print a summary, return results."""
    class_names = list(class_names)
    n = len(class_names)
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, n)
    norm = normalize_rows(cm)
    metrics = per_class_metrics(y_true, y_pred, y_prob, class_names)
    bal_acc = balanced_accuracy(metrics, class_names)
    wrong = top_confident_wrong(y_true, y_pred, y_prob, image_paths, class_names, lesion_ids, top_n)

    # ---- write files ----
    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    _write_csv(cm_csv, ["true\\pred", *class_names],
               [[class_names[r], *cm[r]] for r in range(n)])
    cmn_csv = os.path.join(out_dir, "confusion_matrix_normalized.csv")
    _write_csv(cmn_csv, ["true\\pred", *class_names],
               [[class_names[r], *[round(v, 4) for v in norm[r]]] for r in range(n)])
    pcm_csv = os.path.join(out_dir, "per_class_metrics.csv")
    _write_csv(pcm_csv, ["class", "precision", "recall", "f1", "support", "pr_auc"],
               [[m["class"], round(m["precision"], 4), round(m["recall"], 4),
                 round(m["f1"], 4), m["support"],
                 ("" if isinstance(m["pr_auc"], float) and math.isnan(m["pr_auc"]) else round(m["pr_auc"], 4))]
                for m in metrics])
    tw_csv = os.path.join(out_dir, f"top{top_n}_confident_wrong.csv")
    _write_csv(tw_csv,
               ["rank", "image_path", "true_class", "predicted_class", "confidence",
                "all_class_probabilities", "lesion_id"],
               [[w["rank"], w["image_path"], w["true_class"], w["predicted_class"],
                 w["confidence"], w["all_class_probabilities"], w["lesion_id"]] for w in wrong])
    png_ok = _save_confusion_png(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))

    # ---- console summary ----
    macro = next(m for m in metrics if m["class"] == "macro_avg")
    weighted = next(m for m in metrics if m["class"] == "weighted_avg")
    print(f"=== Misclassification audit: {len(y_true)} samples, {n} classes ===")
    print(f"balanced accuracy = {bal_acc:.4f} | macro-F1 = {macro['f1']:.4f} | "
          f"macro PR-AUC = {macro['pr_auc']:.4f} | weighted-F1 = {weighted['f1']:.4f}")
    print(f"{'class':<8}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}{'pr_auc':>9}")
    for m in metrics:
        pra = "  nan" if isinstance(m["pr_auc"], float) and math.isnan(m["pr_auc"]) else f"{m['pr_auc']:.4f}"
        print(f"{m['class']:<8}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
              f"{m['support']:>9}{pra:>9}")
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>6}" for c in class_names))
    for r in range(n):
        print(f"{class_names[r]:<8}" + "".join(f"{cm[r][c]:>6}" for c in range(n)))
    print(f"top-{top_n} confident-wrong (preview):")
    for w in wrong[:5]:
        print(f"   #{w['rank']} {os.path.basename(str(w['image_path']))}: "
              f"true={w['true_class']} pred={w['predicted_class']} conf={w['confidence']:.3f}")
    print("written: " + ", ".join(os.path.basename(p) for p in [cm_csv, cmn_csv, pcm_csv, tw_csv]
                                  + (["confusion_matrix.png"] if png_ok else [])))
    if not png_ok:
        print("  (confusion_matrix.png skipped: matplotlib not installed)")

    return {
        "confusion_matrix": cm, "confusion_matrix_normalized": norm,
        "per_class_metrics": metrics, "balanced_accuracy": bal_acc,
        "top_confident_wrong": wrong,
        "files": {"confusion_csv": cm_csv, "confusion_norm_csv": cmn_csv,
                  "per_class_csv": pcm_csv, "top_wrong_csv": tw_csv, "png": png_ok},
    }
