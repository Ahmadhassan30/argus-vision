"""Canonical image transforms for Argus Vision — the single source of truth.

Every stage that turns a dermoscopy image into a model tensor must use the SAME
pipeline, or training and inference silently disagree. This module defines the two
canonical pipelines and is imported by the training notebooks and `dataset.py`:

* :func:`get_train_transform` — stochastic augmentation for representation learning.
* :func:`get_eval_transform`  — the DETERMINISTIC validation / test / inference
  pipeline: ``Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)``.

The production backend (`backend/services/image_service.py`) lives in a separate
deployable package and cannot import this module across the Docker boundary, so it
mirrors :func:`get_eval_transform` exactly (same ``Resize(256) -> CenterCrop(224)``)
— see the cross-reference comment there. Keeping the resize at 256 (not 224) before
the 224 centre-crop is the fix that makes inference preprocessing identical to
training-time evaluation.
"""

from __future__ import annotations

from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

#: Shorter-side resize applied BEFORE the square crop (must exceed IMAGE_SIZE so
#: the centre-crop keeps a consistent margin of context, matching training).
RESIZE_SIZE: int = 256

#: Canonical RandomResizedCrop area scale. Reconciles the old divergence between
#: the notebooks (0.7) and dataset.py (0.8) onto a single value: 0.7.
RANDOM_RESIZED_CROP_SCALE: tuple[float, float] = (0.7, 1.0)


def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Stochastic training augmentation pipeline (representation learning).

    Args:
        image_size: Output square size in pixels (default :data:`config.IMAGE_SIZE`).

    Returns:
        A ``torchvision.transforms.Compose`` yielding a normalized
        ``(3, image_size, image_size)`` tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomResizedCrop(image_size, scale=RANDOM_RESIZED_CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic eval / test / inference pipeline.

    ``Resize(256) -> CenterCrop(image_size) -> ToTensor -> Normalize(ImageNet)``.
    This is the canonical transform the production backend mirrors.

    Args:
        image_size: Output square size in pixels (default :data:`config.IMAGE_SIZE`).

    Returns:
        A ``torchvision.transforms.Compose`` yielding a normalized
        ``(3, image_size, image_size)`` tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
