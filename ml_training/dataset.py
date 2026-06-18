"""ISIC-2019 dataset, transforms and class-balancing utilities for Argus Vision.

This module provides :class:`ISICDataset`, a ``torch.utils.data.Dataset`` that
understands both common ISIC label layouts:

* a wide one-hot layout where each ISIC-8 class has its own column
  (the canonical ISIC-2019 ``*_GroundTruth.csv`` shape), and
* a long layout with a single ``label`` column holding either an integer class
  index or the class name string.

In both cases the labels are normalized to integer indices that match the
contract ordering defined in :data:`config.ISIC_CLASSES`.

It also exposes the training and validation augmentation pipelines via
:func:`get_train_transforms` and :func:`get_val_transforms`, and helpers for
handling the heavy class imbalance present in dermoscopy datasets:
:meth:`ISICDataset.compute_class_weights` and
:meth:`ISICDataset.make_weighted_sampler`.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ISIC_CLASSES,
    NUM_CLASSES,
)

# Case-insensitive lookup from class name to contract index.
_CLASS_TO_IDX: dict[str, int] = {name.upper(): idx for idx, name in enumerate(ISIC_CLASSES)}


def get_train_transforms() -> transforms.Compose:
    """Build the training-time augmentation pipeline.

    The pipeline resizes the image, takes a randomized crop and applies a set
    of photometric and geometric augmentations appropriate for dermoscopy
    imagery (which has no canonical orientation), followed by ImageNet
    normalization and a light random-erasing regularizer.

    Returns:
        A ``torchvision.transforms.Compose`` producing a normalized
        ``torch.Tensor`` of shape ``(3, IMAGE_SIZE, IMAGE_SIZE)``.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Build the deterministic validation/inference transform pipeline.

    Returns:
        A ``torchvision.transforms.Compose`` producing a normalized
        ``torch.Tensor`` of shape ``(3, IMAGE_SIZE, IMAGE_SIZE)``.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# Module-level instantiations kept for convenience / backward compatibility.
train_transforms: transforms.Compose = get_train_transforms()
val_transforms: transforms.Compose = get_val_transforms()


class ISICDataset(Dataset):
    """Dataset over ISIC-2019 dermoscopy images for ISIC-8 classification.

    Args:
        csv_path: Path to the label CSV. The CSV must contain an ``image``
            column with the (extension-less or extension-bearing) image
            filename. Labels may be provided either as one-hot columns named
            after the ISIC-8 classes, or as a single ``label`` column holding
            an integer index or class-name string.
        image_dir: Directory containing the image files referenced by the CSV.
        transform: A callable applied to each PIL image; typically the result
            of :func:`get_train_transforms` or :func:`get_val_transforms`.
        split: Free-form split name (e.g. ``"train"``, ``"val"``, ``"test"``)
            retained for bookkeeping and logging.

    Attributes:
        labels: A list of integer class indices aligned with ``self.images``.
        images: A list of image filenames aligned with ``self.labels``.
    """

    #: Common image extensions to probe when the CSV stores extension-less names.
    _IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform,  # noqa: ANN001 - torchvision Compose / callable, kept generic.
        split: str = "train",
    ) -> None:
        self.csv_path: str = csv_path
        self.image_dir: str = image_dir
        self.transform = transform
        self.split: str = split

        frame: pd.DataFrame = pd.read_csv(csv_path)
        if "image" not in frame.columns:
            raise ValueError(
                f"CSV at {csv_path!r} must contain an 'image' column; "
                f"found columns {list(frame.columns)}."
            )

        self.images: list[str] = frame["image"].astype(str).tolist()
        self.labels: list[int] = self._extract_labels(frame)

        if len(self.images) != len(self.labels):
            raise ValueError(
                "Mismatched image/label counts after parsing "
                f"({len(self.images)} images vs {len(self.labels)} labels)."
            )

    def _extract_labels(self, frame: pd.DataFrame) -> list[int]:
        """Normalize the label representation in ``frame`` to integer indices.

        Supports a wide one-hot layout (one column per ISIC-8 class) and a long
        layout (single ``label`` column of integers or class-name strings).

        Args:
            frame: The raw CSV contents.

        Returns:
            A list of integer class indices in ``[0, NUM_CLASSES)``.

        Raises:
            ValueError: If neither supported layout can be detected, or if a
                label value cannot be mapped to a valid class index.
        """
        # Detect one-hot layout: every ISIC class name present as a column.
        column_upper: dict[str, str] = {col.upper(): col for col in frame.columns}
        onehot_columns: list[str] = [
            column_upper[name.upper()] for name in ISIC_CLASSES if name.upper() in column_upper
        ]
        if len(onehot_columns) == NUM_CLASSES:
            onehot = frame[onehot_columns].to_numpy()
            return onehot.argmax(axis=1).astype(int).tolist()

        # Otherwise expect a single label column (integer or class-name string).
        label_column: Optional[str] = None
        for candidate in ("label", "Label", "LABEL", "target", "diagnosis", "dx"):
            if candidate in frame.columns:
                label_column = candidate
                break
        if label_column is None:
            raise ValueError(
                "Could not detect labels: expected either one-hot class columns "
                f"{ISIC_CLASSES} or a single label column. Found columns "
                f"{list(frame.columns)}."
            )

        labels: list[int] = []
        for raw_value in frame[label_column].tolist():
            labels.append(self._coerce_label(raw_value))
        return labels

    @staticmethod
    def _coerce_label(raw_value: object) -> int:
        """Coerce a single raw label value to an integer class index.

        Accepts integers, integer-like floats, numeric strings and class-name
        strings (case-insensitive).

        Args:
            raw_value: A label cell from the CSV.

        Returns:
            The integer class index in ``[0, NUM_CLASSES)``.

        Raises:
            ValueError: If the value cannot be mapped to a valid class index.
        """
        # Class-name string (e.g. "MEL").
        if isinstance(raw_value, str):
            key = raw_value.strip().upper()
            if key in _CLASS_TO_IDX:
                return _CLASS_TO_IDX[key]
            # Numeric string (e.g. "3" or "3.0").
            try:
                idx = int(float(key))
            except ValueError as exc:
                raise ValueError(f"Unrecognized label value {raw_value!r}.") from exc
        else:
            # Numeric (int / float / numpy scalar).
            try:
                idx = int(float(raw_value))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unrecognized label value {raw_value!r}.") from exc

        if not 0 <= idx < NUM_CLASSES:
            raise ValueError(
                f"Label index {idx} out of range [0, {NUM_CLASSES}) for value {raw_value!r}."
            )
        return idx

    def _resolve_path(self, image_name: str) -> str:
        """Resolve a CSV image entry to an on-disk path.

        Tries the name verbatim first, then probes common extensions when the
        CSV stores extension-less filenames.

        Args:
            image_name: The raw filename from the ``image`` column.

        Returns:
            An absolute or relative filesystem path to the image.
        """
        direct_path = os.path.join(self.image_dir, image_name)
        if os.path.isfile(direct_path):
            return direct_path

        _, ext = os.path.splitext(image_name)
        if ext == "":
            for candidate_ext in self._IMAGE_EXTENSIONS:
                candidate = os.path.join(self.image_dir, image_name + candidate_ext)
                if os.path.isfile(candidate):
                    return candidate

        # Fall back to the direct path; PIL will raise a clear error if missing.
        return direct_path

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Load and transform a single sample.

        Args:
            idx: Index into the dataset.

        Returns:
            A tuple ``(image_tensor, label_int, image_path)`` where
            ``image_tensor`` is the transformed image, ``label_int`` is the
            integer class index and ``image_path`` is the resolved file path.
        """
        image_path = self._resolve_path(self.images[idx])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)
        label_int: int = int(self.labels[idx])
        return image_tensor, label_int, image_path

    def _class_counts(self) -> torch.Tensor:
        """Compute per-class sample counts as a float tensor of size NUM_CLASSES."""
        counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        for label in self.labels:
            counts[label] += 1.0
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-square-root frequency class weights.

        Weights are ``1 / sqrt(count)`` per class, normalized so their mean is
        ``1.0``. Classes that are absent from the split receive a weight of
        zero (their inverse frequency is undefined). The result is suitable as
        the ``alpha`` argument of :class:`losses.FocalLoss` or the ``weight``
        argument of ``nn.CrossEntropyLoss``.

        Returns:
            A ``torch.FloatTensor`` of shape ``(NUM_CLASSES,)``.
        """
        counts = self._class_counts()
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        present = counts > 0
        weights[present] = 1.0 / torch.sqrt(counts[present])

        mean_present = weights[present].mean() if present.any() else torch.tensor(1.0)
        if mean_present > 0:
            weights = weights / mean_present
        return weights.to(dtype=torch.float32)

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a :class:`WeightedRandomSampler` for class-balanced sampling.

        Each sample is assigned a weight inversely proportional to its class
        frequency, so that rare classes are oversampled during training to the
        same expected per-class rate.

        Returns:
            A ``WeightedRandomSampler`` with ``num_samples == len(self)`` and
            replacement enabled.
        """
        counts = self._class_counts()
        # Per-class sampling weight: inverse frequency (guard against div-by-zero).
        per_class_weight = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        present = counts > 0
        per_class_weight[present] = 1.0 / counts[present]

        sample_weights = torch.tensor(
            [per_class_weight[label].item() for label in self.labels],
            dtype=torch.float64,
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.labels),
            replacement=True,
        )
