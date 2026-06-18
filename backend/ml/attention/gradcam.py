"""Grad-CAM++ saliency for the EfficientNet-B4 agent (Agent A).

This module computes a 224x224 class-discriminative saliency map for a timm
EfficientNet-B4 backbone using the Grad-CAM++ algorithm from the
``pytorch_grad_cam`` package (pinned as ``grad-cam==1.5.0``).

The target convolutional layer is resolved robustly so the same code path works
for both a fine-tuned Argus checkpoint and the ImageNet-pretrained fallback that
the system uses when no checkpoint is present.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGE_SIZE: int = 224


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """Resolve a robust Grad-CAM++ target layer for a timm EfficientNet-B4.

    The spec calls for ``model.blocks[-1].bn3`` as the ideal target. We honor
    that intent while falling back gracefully so the function also works against
    the ImageNet-pretrained fallback backbone (whose module names may differ).

    Resolution order:

    1. ``model.blocks[-1]`` (the final inverted-residual stage; if it exposes a
       ``bn3`` sub-module we target that, matching the spec's stated intent).
    2. ``model.conv_head`` (the 1x1 head convolution present on timm EffNets).
    3. The last ``nn.BatchNorm2d`` or ``nn.Conv2d`` module discovered by walking
       every sub-module of the model.

    Args:
        model: The (timm) EfficientNet-B4 module to introspect.

    Returns:
        The selected target ``nn.Module`` for Grad-CAM++ hooking.

    Raises:
        RuntimeError: If no convolutional or batch-norm layer can be found.
    """
    # 1. Final block (optionally its bn3, per spec intent).
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        last_block = blocks[-1]
        bn3 = getattr(last_block, "bn3", None)
        if isinstance(bn3, nn.Module):
            return bn3
        return last_block

    # 2. Head convolution.
    conv_head = getattr(model, "conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head

    # 3. Last BatchNorm2d / Conv2d found by walking the module tree.
    fallback: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
            fallback = module
    if fallback is not None:
        return fallback

    raise RuntimeError(
        "Could not resolve a Grad-CAM++ target layer: no blocks, conv_head, "
        "Conv2d, or BatchNorm2d module was found on the supplied model."
    )


def compute_gradcam_plusplus(
    model: nn.Module,
    tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Compute a Grad-CAM++ saliency map for ``target_class``.

    Args:
        model: The EfficientNet-B4 classifier (Agent A backbone).
        tensor: A pre-processed input batch of shape ``(1, 3, 224, 224)`` on the
            same device as ``model``.
        target_class: The class index (0..7 for ISIC-8) to explain.

    Returns:
        A ``224x224`` ``float32`` ``np.ndarray`` with values in ``[0, 1]``.
    """
    target_layer = _resolve_target_layer(model)

    # Clear any stale gradients before building the CAM.
    model.zero_grad()

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)

    # Clear gradients again so we do not leak state back into the caller.
    model.zero_grad()

    # grad-cam returns shape (batch, H, W); take the first sample. Values are
    # already min-max scaled into [0, 1] by the library.
    heatmap = np.asarray(grayscale_cam[0], dtype=np.float32)

    if heatmap.shape != (IMAGE_SIZE, IMAGE_SIZE):
        heatmap = cv2.resize(
            heatmap,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.ascontiguousarray(np.clip(heatmap, 0.0, 1.0), dtype=np.float32)
