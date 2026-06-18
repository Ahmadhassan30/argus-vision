"""Classification loss functions for Argus Vision agent training.

Provides two losses used when training the EfficientNet-B4 (Agent A) and
ViT-B/16 (Agent B) classifiers on the heavily imbalanced ISIC-8 task:

* :class:`FocalLoss` - down-weights well-classified examples and supports a
  per-class ``alpha`` weighting tensor (typically the inverse-square-root
  frequency weights produced by
  :meth:`dataset.ISICDataset.compute_class_weights`).
* :class:`LabelSmoothingCrossEntropy` - a standard label-smoothing
  cross-entropy regularizer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting.

    Implements ``loss = (1 - p_t) ** gamma * CE`` where ``p_t`` is the model's
    probability assigned to the ground-truth class. A larger ``gamma`` places
    more relative emphasis on hard, misclassified examples.

    Args:
        gamma: Focusing parameter ``>= 0``. ``gamma == 0`` reduces the loss to
            (weighted) cross-entropy.
        alpha: Optional per-class weight tensor of shape ``(NUM_CLASSES,)``,
            passed through to ``F.cross_entropy`` as its ``weight`` argument.
        reduction: One of ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be one of 'mean', 'sum', 'none'; got {reduction!r}."
            )
        self.gamma: float = gamma
        self.reduction: str = reduction
        # Register alpha as a buffer so it moves with .to(device) / .cuda().
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            logits: Unnormalized class scores of shape ``(N, NUM_CLASSES)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            The reduced loss, a scalar tensor unless ``reduction == "none"`` in
            which case a tensor of shape ``(N,)`` is returned.
        """
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing regularization.

    Smooths the one-hot target distribution by redistributing ``smoothing``
    probability mass uniformly across all classes, which discourages
    over-confident predictions and tends to improve calibration.

    Args:
        smoothing: Smoothing factor in ``[0, 1)``. ``0`` recovers standard
            cross-entropy.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1); got {smoothing}.")
        self.smoothing: float = smoothing
        self.confidence: float = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Args:
            logits: Unnormalized class scores of shape ``(N, NUM_CLASSES)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            A scalar loss tensor (mean over the batch).
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Negative log-likelihood of the true class.
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform smoothing term: mean negative log-prob over all classes.
        smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
