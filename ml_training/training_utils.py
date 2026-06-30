"""Decoupled-training and logit-adjustment utilities for Argus Vision (Phase 4).

Implements the two imbalance techniques used by the agent training notebooks, kept
here as small, independently-testable pieces (the notebooks orchestrate the two
stages using their own per-epoch train/eval functions):

* **Decoupled training (Kang et al., 2020):** Stage A learns representations with
  instance-balanced sampling (plain DataLoader, no WeightedRandomSampler) and an
  *unweighted* loss; Stage B freezes the backbone and retrains ONLY the classifier
  head with class-balanced sampling, correcting the decision boundary without
  distorting the learned features. :func:`freeze_all_but_classifier` performs the
  freeze (via the timm ``get_classifier`` API, which works for both EfficientNet-B4
  and ViT-B/16), and :func:`snapshot_frozen_params` / :func:`assert_frozen_unchanged`
  let the notebook PROVE the freeze actually held (params bit-identical, requires_grad
  False) after Stage B.
* **Logit adjustment (Menon et al., 2021):** :func:`apply_logit_adjustment` adds
  ``tau * log(prior)`` to the logits before the loss, where ``prior`` is the EMPIRICAL
  (un-rebalanced) class frequency. Toggleable independently of decoupled training.
"""

from __future__ import annotations

import os
import types
from typing import Optional, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Logit adjustment (Menon et al., 2021)
# --------------------------------------------------------------------------- #
def class_priors_from_counts(class_counts: Sequence[float]) -> list[float]:
    """Empirical class priors P(y=c) = count_c / total (NOT the rebalanced weights).

    Args:
        class_counts: Per-class training-set counts, canonical order.

    Returns:
        A list of priors that sum to 1.0.
    """
    counts = [float(c) for c in class_counts]
    total = sum(counts)
    if total <= 0:
        n = len(counts)
        return [1.0 / n] * n
    return [c / total for c in counts]


def apply_logit_adjustment(
    logits: torch.Tensor,
    class_priors: Sequence[float] | torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Train-time logit adjustment: ``logits + tau * log(prior)``.

    Added BEFORE the loss so the model learns a prior-corrected decision boundary
    (Menon et al., "Long-tail learning via logit adjustment", ICLR 2021). At inference
    the RAW logits are used (no adjustment), which then favour rare classes.

    Args:
        logits: ``(N, C)`` raw class scores.
        class_priors: length-``C`` empirical class frequencies (see
            :func:`class_priors_from_counts`).
        tau: Adjustment strength (1.0 is the standard value).

    Returns:
        The adjusted ``(N, C)`` logits.
    """
    log_priors = torch.log(
        torch.as_tensor(class_priors, device=logits.device, dtype=logits.dtype) + 1e-12
    )
    return logits + tau * log_priors


class LogitAdjustedLoss(nn.Module):
    """Wrap a base loss so train-time logit adjustment is applied to the logits first.

    Lets the notebooks toggle logit adjustment WITHOUT touching their per-epoch training
    loop: just pass ``LogitAdjustedLoss(base_criterion, priors, tau)`` as the criterion.
    Only the training loss is adjusted; evaluation uses the raw model logits.

    Args:
        base_criterion: e.g. a :class:`losses.FocalLoss` instance.
        class_priors: empirical class priors (see :func:`class_priors_from_counts`).
        tau: adjustment strength.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        class_priors: Sequence[float],
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.register_buffer("_log_priors", torch.log(torch.as_tensor(list(class_priors), dtype=torch.float32) + 1e-12))
        self.tau = tau

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self._log_priors.to(logits.device, logits.dtype)
        return self.base_criterion(adjusted, targets)


# --------------------------------------------------------------------------- #
# Decoupled training — Stage B backbone freeze + verification
# --------------------------------------------------------------------------- #
def freeze_all_but_classifier(model: nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the final classifier head (Stage B).

    Uses ``model.get_classifier()`` (the timm API, valid for EfficientNet-B4 and
    ViT-B/16) to locate the head. After this call, ONLY the head trains.

    Args:
        model: The timm classification model.

    Returns:
        ``(trainable_params, total_params)`` parameter counts.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.get_classifier().parameters():
        param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_backbone_bn(model: nn.Module) -> int:
    """Also freeze BatchNorm *running statistics* during Stage B (canonical cRT).

    Freezing parameters alone is not enough for a truly fixed backbone: BatchNorm's
    running_mean / running_var are BUFFERS (not params), so they keep updating whenever
    the module is in train mode. Because the reused per-epoch loop calls ``model.train(True)``,
    this sets every BatchNorm module to ``eval`` AND overrides its ``train`` method to a
    no-op so ``model.train(True)`` can no longer re-enable it — keeping running stats fixed
    through Stage B. No-op for LayerNorm-only models (ViT-B/16 has no running stats).

    Args:
        model: The model whose BatchNorm running stats should be frozen.

    Returns:
        The number of BatchNorm modules frozen (0 for a pure-LayerNorm model).
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    n = 0
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()
            module.train = types.MethodType(lambda self, mode=True: self, module)
            n += 1
    return n


def snapshot_frozen_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone the values of all currently-frozen params (call AFTER freezing).

    Returns:
        ``{name: detached clone}`` for every param with ``requires_grad == False``.
    """
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }


def assert_frozen_unchanged(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> int:
    """Prove the freeze held: frozen params are still frozen AND bit-identical.

    Compares every snapshotted param's current value to its pre-Stage-B clone with
    ``torch.equal`` (exact), and re-checks ``requires_grad is False``.

    Args:
        model: The model after Stage B training.
        snapshot: The dict returned by :func:`snapshot_frozen_params` before Stage B.

    Raises:
        AssertionError: if any frozen param changed value or became trainable.

    Returns:
        The number of frozen params verified.
    """
    changed: list[str] = []
    unfroze: list[str] = []
    for name, param in model.named_parameters():
        if name in snapshot:
            if param.requires_grad:
                unfroze.append(name)
            if not torch.equal(param.detach(), snapshot[name]):
                changed.append(name)
    assert not unfroze, f"Params unexpectedly trainable in Stage B: {unfroze[:5]}"
    assert not changed, (
        f"Frozen backbone params CHANGED during Stage B (freeze failed): {changed[:5]} "
        f"({len(changed)} total)"
    )
    return len(snapshot)


# --------------------------------------------------------------------------- #
# Checkpoint / resume reliability (Phase 5)
# --------------------------------------------------------------------------- #
def save_resumable(
    path: str,
    stage: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    scaler: Optional[object] = None,
    best_auc: float = -1.0,
    epochs_no_improve: int = 0,
) -> None:
    """Save a FULL resumable training state (separate from the best-weights file).

    Writes model + optimizer + scheduler + AMP-scaler state plus the early-stopping
    bookkeeping so an interrupted stage can continue exactly where it stopped. The
    production best-weights checkpoint (a plain ``model.state_dict()``) is written
    separately and stays clean, so the backend's loader needs no changes.
    """
    payload = {
        "stage": stage,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_auc": float(best_auc),
        "epochs_no_improve": int(epochs_no_improve),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the resume file


def load_resumable(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    scaler: Optional[object] = None,
    map_location: Optional[object] = None,
) -> Optional[dict]:
    """Restore a resumable state in place; return resume bookkeeping or None.

    If ``path`` exists, loads model/optimizer/scheduler/scaler state into the passed
    objects and returns ``{start_epoch, best_auc, epochs_no_improve, stage}`` where
    ``start_epoch = saved_epoch + 1``. Returns ``None`` when there is nothing to resume.
    """
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return {
        "start_epoch": int(ckpt["epoch"]) + 1,
        "best_auc": float(ckpt["best_auc"]),
        "epochs_no_improve": int(ckpt["epochs_no_improve"]),
        "stage": ckpt.get("stage"),
    }
