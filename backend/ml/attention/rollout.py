"""Attention rollout for the ViT-B/16 agent (Agent B).

Implements the attention-rollout method of Abnar & Zuidema (2020) for a timm
Vision Transformer (``vit_base_patch16_224``-style backbone).

timm's ``Attention`` module does not, by default, expose the post-softmax
attention matrix: when ``fused_attn`` is enabled it dispatches to
``F.scaled_dot_product_attention`` which fuses the softmax into a single kernel,
and even on the slow path the attention tensor is a local variable. To capture a
reliable attention map we therefore:

1. Disable ``fused_attn`` on every block's attention module.
2. Register a forward hook on each ``block.attn`` module that re-derives the
   per-head softmax attention from the hook's *input* tensor ``x`` using the
   module's own ``qkv`` projection, ``num_heads`` and ``scale``.

The captured per-layer attention is averaged over heads, residual-augmented
(``0.5 * (A + I)``) and row-normalized, then the layers are multiplied together
(the "rollout"). The CLS-token row over the 196 patch tokens is reshaped to a
``14x14`` grid and upsampled to ``224x224``.

All hooks are always removed and ``fused_attn`` is always restored in a
``finally`` block. If anything goes wrong the function logs a warning and falls
back to a centered Gaussian saliency so the debate pipeline can still run.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGE_SIZE: int = 224
GRID_SIZE: int = 14  # 224 / 16 patch size -> 14x14 = 196 patch tokens.
NUM_PATCH_TOKENS: int = GRID_SIZE * GRID_SIZE


def _centered_gaussian(size: int = IMAGE_SIZE, sigma_frac: float = 0.25) -> np.ndarray:
    """Build a centered isotropic Gaussian saliency map in ``[0, 1]``.

    Used as a graceful fallback when attention capture fails so the rest of the
    pipeline continues to receive a well-formed heatmap.

    Args:
        size: Output side length in pixels.
        sigma_frac: Standard deviation as a fraction of ``size``.

    Returns:
        A ``size x size`` ``float32`` array normalized to ``[0, 1]``.
    """
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords)
    sigma = max(sigma_frac, 1e-6)
    gaussian = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma * sigma))
    gaussian = gaussian.astype(np.float32)
    span = float(gaussian.max() - gaussian.min())
    if span < 1e-12:
        return np.zeros((size, size), dtype=np.float32)
    return ((gaussian - gaussian.min()) / span).astype(np.float32)


def compute_attention_rollout(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Compute an attention-rollout saliency map for a timm ViT.

    Args:
        model: The ViT-B/16 classifier (Agent B backbone). Must expose
            ``model.blocks[i].attn`` modules with ``qkv``, ``num_heads`` and
            ``scale`` attributes (the standard timm ``Attention`` layout).
        tensor: A pre-processed input batch of shape ``(1, 3, 224, 224)`` on the
            same device as ``model``.

    Returns:
        A ``224x224`` ``float32`` ``np.ndarray`` with values in ``[0, 1]``. On
        failure this is a centered Gaussian fallback.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    captured: Dict[int, torch.Tensor] = {}
    original_fused: Dict[int, bool] = {}

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        logger.warning(
            "compute_attention_rollout: model has no .blocks; using Gaussian "
            "fallback saliency."
        )
        return _centered_gaussian()

    def _make_hook(layer_idx: int):
        """Create a forward hook that re-derives softmax attention for a block."""

        def _hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            try:
                x = inputs[0]
                batch, num_tokens, dim = x.shape
                num_heads = int(module.num_heads)
                head_dim = dim // num_heads
                # timm scale may be a float; recompute robustly if absent.
                scale = getattr(module, "scale", None)
                if scale is None:
                    scale = head_dim ** -0.5

                qkv = module.qkv(x)  # (B, N, 3 * dim)
                qkv = qkv.reshape(batch, num_tokens, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                q, k = qkv[0], qkv[1]

                attn = (q @ k.transpose(-2, -1)) * float(scale)  # (B, heads, N, N)
                attn = attn.softmax(dim=-1)
                # Average over heads -> (B, N, N); detach + move to CPU float32.
                captured[layer_idx] = attn.mean(dim=1).detach().to(torch.float32).cpu()
            except Exception as exc:  # pragma: no cover - defensive capture.
                logger.warning(
                    "compute_attention_rollout: failed to capture attention for "
                    "block %d: %s",
                    layer_idx,
                    exc,
                )

        return _hook

    try:
        # Disable fused attention and register hooks on every block's attn.
        for idx, block in enumerate(blocks):
            attn_module = getattr(block, "attn", None)
            if attn_module is None or not hasattr(attn_module, "qkv"):
                continue
            original_fused[idx] = bool(getattr(attn_module, "fused_attn", False))
            if hasattr(attn_module, "fused_attn"):
                attn_module.fused_attn = False
            handles.append(attn_module.register_forward_hook(_make_hook(idx)))

        if not handles:
            logger.warning(
                "compute_attention_rollout: no attention modules with a qkv "
                "projection were found; using Gaussian fallback saliency."
            )
            return _centered_gaussian()

        # Forward pass to populate the captured attention maps.
        with torch.no_grad():
            model(tensor)

        if not captured:
            logger.warning(
                "compute_attention_rollout: no attention matrices were captured "
                "during the forward pass; using Gaussian fallback saliency."
            )
            return _centered_gaussian()

        # Assemble per-layer attention in block order.
        layer_indices = sorted(captured.keys())
        first_attn = captured[layer_indices[0]]
        num_tokens = first_attn.shape[-1]
        identity = torch.eye(num_tokens, dtype=torch.float32)

        rollout = torch.eye(num_tokens, dtype=torch.float32)
        for idx in layer_indices:
            attn = captured[idx][0]  # (N, N) for the single batch element.
            # Residual augmentation then row-normalize.
            aug = 0.5 * attn + 0.5 * identity
            aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # Sequential matrix product across layers.
            rollout = aug @ rollout

        # CLS token is index 0; take its row over the patch-token columns,
        # dropping the CLS column itself.
        cls_attention = rollout[0, 1:]
        if cls_attention.shape[0] < NUM_PATCH_TOKENS:
            logger.warning(
                "compute_attention_rollout: expected at least %d patch tokens "
                "but got %d; using Gaussian fallback saliency.",
                NUM_PATCH_TOKENS,
                cls_attention.shape[0],
            )
            return _centered_gaussian()

        grid = cls_attention[:NUM_PATCH_TOKENS].reshape(GRID_SIZE, GRID_SIZE)
        grid_np = grid.numpy().astype(np.float32)

        heatmap = cv2.resize(
            grid_np,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)

        span = float(heatmap.max() - heatmap.min())
        if span < 1e-12:
            return _centered_gaussian()
        heatmap = (heatmap - heatmap.min()) / span
        return np.ascontiguousarray(np.clip(heatmap, 0.0, 1.0), dtype=np.float32)

    except Exception as exc:
        logger.warning(
            "compute_attention_rollout: unexpected failure (%s); using Gaussian "
            "fallback saliency.",
            exc,
        )
        return _centered_gaussian()

    finally:
        # Always remove hooks and restore the original fused_attn flags.
        for handle in handles:
            handle.remove()
        for idx, was_fused in original_fused.items():
            attn_module = getattr(blocks[idx], "attn", None)
            if attn_module is not None and hasattr(attn_module, "fused_attn"):
                attn_module.fused_attn = was_fused
