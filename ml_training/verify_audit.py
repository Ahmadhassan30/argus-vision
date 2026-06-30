"""Verification tool for Phase 6 (misclassification audit) — RUN ON KAGGLE or LOCALLY.

Proves that audit.py runs end-to-end on a synthetic 2531-sample dataset,
producing confusion matrices, per-class metrics (including PR-AUC), and the
top confident wrong predictions CSV, validating schema and data sorting.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from audit import run_audit, CLASS_NAMES_DEFAULT


def verify_audit() -> None:
    # 1. Generate 2531 synthetic samples matching the classes
    n_samples = 2531
    random.seed(42)

    y_true = [random.randint(0, 7) for _ in range(n_samples)]
    y_pred = []
    y_prob = []
    
    for yt in y_true:
        # Generate random probability distribution
        logits = [random.random() for _ in range(8)]
        # Boost true class slightly so prediction is somewhat meaningful
        if random.random() < 0.6:
            logits[yt] += 1.0
        
        # Softmax
        exp_logits = [math.exp(l) for l in logits]
        sum_exp = sum(exp_logits)
        prob = [e / sum_exp for e in exp_logits]
        y_prob.append(prob)
        y_pred.append(prob.index(max(prob)))

    image_paths = [f"/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_{i:07d}.jpg" for i in range(n_samples)]
    lesion_ids = [f"IL_{random.randint(1000, 2000)}" if random.random() < 0.9 else None for _ in range(n_samples)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 2. Run the audit
        res = run_audit(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            image_paths=image_paths,
            class_names=CLASS_NAMES_DEFAULT,
            lesion_ids=lesion_ids,
            out_dir=tmp_dir,
            top_n=100
        )

        # 3. Verify outputs are computed and files are written
        assert "balanced_accuracy" in res
        assert "confusion_matrix" in res
        assert "per_class_metrics" in res
        assert "top_confident_wrong" in res

        expected_files = [
            "confusion_matrix.csv",
            "confusion_matrix_normalized.csv",
            "per_class_metrics.csv",
            "top100_confident_wrong.csv"
        ]

        for ef in expected_files:
            file_path = os.path.join(tmp_dir, ef)
            assert os.path.exists(file_path), f"Expected audit file {ef} was not written to {tmp_dir}"
            assert os.path.getsize(file_path) > 0, f"File {ef} is empty"

        # 4. Check sorting of the top wrong predictions
        wrong_predictions = res["top_confident_wrong"]
        assert len(wrong_predictions) <= 100, f"Expected at most 100 wrong predictions, got {len(wrong_predictions)}"
        
        # Confidences should be descending
        confidences = [w["confidence"] for w in wrong_predictions]
        assert confidences == sorted(confidences, reverse=True), "Top wrong predictions are not sorted by confidence descending"

    print("[PASS] audit verification passed: confusion matrix, per-class metrics, and top-wrong list parsed successfully.")


def main() -> None:
    verify_audit()
    print("ALL AUDIT VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
