"""Image preprocessing and heatmap rendering utilities for Argus Vision.

These are pure functions (no shared state) used by the ML pipeline to turn an
uploaded image into a normalised model tensor, and to render Grad-CAM / attention
heatmaps as base64-encoded PNG overlays suitable for transport over the API and
WebSocket layers.

ImageNet normalisation and a 224x224 input size are used throughout, matching the
shared model contract.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

#: Target square input size (pixels) for both models.
IMAGE_SIZE = 224
#: Shorter-side resize applied BEFORE the centre-crop. MUST be 256 (not 224) so the
#: inference pipeline is identical to the training-time evaluation transform
#: (ml_training/transforms.get_eval_transform): Resize(256) -> CenterCrop(224).
#: Resizing straight to 224 cropped away the margin of context the models were
#: trained on, causing a silent train/inference distribution shift.
RESIZE_SIZE = 256
#: ImageNet channel means used for normalisation.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
#: ImageNet channel standard deviations used for normalisation.
IMAGENET_STD = [0.229, 0.224, 0.225]

#: Reusable preprocessing transform: resize, centre-crop, tensorise, normalise.
#: Mirrors ml_training/transforms.get_eval_transform exactly (cross-package: the
#: backend Docker image cannot import ml_training, so the definition is duplicated
#: but kept byte-for-byte equivalent): Resize(256) -> CenterCrop(224).
_TRANSFORM = T.Compose(
    [
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """Load an image and produce a model-ready tensor plus the original RGB image.

    The image is opened with PIL and converted to RGB. The original (full
    resolution) RGB image is returned unchanged for downstream display and
    heatmap overlay use, while a normalised ``[1, 3, 224, 224]`` tensor is
    produced for model inference.

    Args:
        image_path: Filesystem path to the source image.

    Returns:
        A tuple ``(tensor, original_pil)`` where ``tensor`` has shape
        ``[1, 3, 224, 224]`` (batch dimension included) and ``original_pil`` is
        the source image converted to RGB.
    """
    with Image.open(image_path) as opened:
        original_pil = opened.convert("RGB")

    tensor: torch.Tensor = _TRANSFORM(original_pil).unsqueeze(0)
    return tensor, original_pil


def heatmap_to_b64(heatmap_array: np.ndarray, original_image: Image.Image) -> str:
    """Render a heatmap as a jet-colormapped overlay on the original image.

    The original image is resized to ``224x224`` and the ``[0, 1]`` heatmap is
    resized to match, converted to an 8-bit jet colormap, and alpha-blended over
    the image. The resulting RGB image is PNG-encoded and returned as a raw
    base64 string (no ``data:`` URI prefix).

    Args:
        heatmap_array: 2D array of activation values, expected (but not required)
            to lie in ``[0, 1]``. Values are clipped to ``[0, 1]`` before use.
        original_image: The original RGB PIL image to overlay onto.

    Returns:
        A base64-encoded PNG string of the blended RGB overlay.
    """
    # Original image -> 224x224 RGB ndarray, then to BGR for OpenCV operations.
    base_rgb = np.array(original_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    # Normalise / clip heatmap to [0, 1], resize to match, scale to uint8.
    heatmap = np.asarray(heatmap_array, dtype=np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    if heatmap.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)

    # Jet colormap (OpenCV returns BGR) then blend with the base image.
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(colored_bgr, 0.4, base_bgr, 0.6, 0.0)

    # Convert back to RGB before PNG-encoding so colours are correct for viewers.
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return _encode_rgb_png_b64(overlay_rgb)


def array_to_b64(numpy_array: np.ndarray) -> str:
    """Render a 2D float array as a standalone jet-colormapped PNG (base64).

    The array is min-max normalised to ``[0, 255]`` (a constant array maps to all
    zeros), colormapped with jet, PNG-encoded, and returned as a raw base64
    string (no ``data:`` URI prefix).

    Args:
        numpy_array: 2D array of float values to visualise.

    Returns:
        A base64-encoded PNG string of the colormapped array.
    """
    arr = np.asarray(numpy_array, dtype=np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    span = arr_max - arr_min
    if span > 0.0:
        normalised = (arr - arr_min) / span
    else:
        normalised = np.zeros_like(arr)
    arr_uint8 = (normalised * 255.0).astype(np.uint8)

    # Jet colormap returns BGR; convert to RGB for correct viewer rendering.
    colored_bgr = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return _encode_rgb_png_b64(colored_rgb)


def _encode_rgb_png_b64(rgb_array: np.ndarray) -> str:
    """PNG-encode an RGB uint8 ndarray and return raw base64.

    Args:
        rgb_array: ``[H, W, 3]`` uint8 array in RGB channel order.

    Returns:
        A base64-encoded PNG string (no ``data:`` URI prefix).
    """
    image = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
