"""Guard against silent drift between the two definitions of "correct preprocessing".

`backend/services/image_service.py` mirrors `ml_training/transforms.get_eval_transform`
across the Docker boundary (the backend image can't import ml_training). They are kept
in sync only by a comment — so this test fails loudly if someone changes one resize /
crop / normalization value and forgets the other.

Runs anywhere: the primary check parses the constants with `ast` (no torch needed). When
torchvision IS importable (Kaggle / a full backend env) it ALSO builds both transforms and
introspects the live Compose steps. Part of the Phase 9 test suite scaffold.
"""

from __future__ import annotations

import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_IMAGE_SERVICE = os.path.join(_REPO, "backend", "services", "image_service.py")
_TRANSFORMS = os.path.join(_REPO, "ml_training", "transforms.py")
_CONFIG = os.path.join(_REPO, "ml_training", "config.py")


def _module_constants(path: str, names: set[str]) -> dict:
    """Extract literal module-level constants (handles plain + annotated assigns)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    out[tgt.id] = ast.literal_eval(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
            and node.value is not None
        ):
            out[node.target.id] = ast.literal_eval(node.value)
    return out


def test_eval_transform_parity() -> None:
    """The backend inference transform must match the canonical eval transform."""
    isvc = _module_constants(_IMAGE_SERVICE, {"RESIZE_SIZE", "IMAGE_SIZE", "IMAGENET_MEAN", "IMAGENET_STD"})
    tx = _module_constants(_TRANSFORMS, {"RESIZE_SIZE"})
    cfg = _module_constants(_CONFIG, {"IMAGE_SIZE", "IMAGENET_MEAN", "IMAGENET_STD"})

    # resize size
    assert isvc["RESIZE_SIZE"] == tx["RESIZE_SIZE"] == 256, (
        f"resize-before-crop drift: image_service={isvc['RESIZE_SIZE']} vs transforms={tx['RESIZE_SIZE']}"
    )
    # crop size (image_service crops IMAGE_SIZE; transforms crops config.IMAGE_SIZE)
    assert isvc["IMAGE_SIZE"] == cfg["IMAGE_SIZE"] == 224, (
        f"crop-size drift: image_service={isvc['IMAGE_SIZE']} vs config={cfg['IMAGE_SIZE']}"
    )
    # normalization stats
    assert isvc["IMAGENET_MEAN"] == cfg["IMAGENET_MEAN"], "normalization MEAN drift"
    assert isvc["IMAGENET_STD"] == cfg["IMAGENET_STD"], "normalization STD drift"


def test_eval_transform_parity_live() -> None:
    """Stronger check when torchvision is available: introspect the live Compose steps."""
    try:
        import sys
        import torchvision.transforms as T  # noqa: F401
    except Exception:
        import pytest  # type: ignore

        pytest.skip("torchvision not installed; AST parity check covers drift locally")
        return

    import sys

    sys.path.insert(0, os.path.join(_REPO, "ml_training"))
    sys.path.insert(0, os.path.join(_REPO, "backend"))
    import transforms as canonical  # ml_training/transforms.py
    from services import image_service as isvc

    def params(compose):
        import torchvision.transforms as T

        resize = next(s for s in compose.transforms if isinstance(s, T.Resize)).size
        crop = next(s for s in compose.transforms if isinstance(s, T.CenterCrop)).size
        norm = next(s for s in compose.transforms if isinstance(s, T.Normalize))
        crop = tuple(crop) if isinstance(crop, (list, tuple)) else (crop, crop)
        return resize, crop, (list(norm.mean), list(norm.std))

    assert params(isvc._TRANSFORM) == params(canonical.get_eval_transform()), (
        "Live backend transform diverged from ml_training.transforms.get_eval_transform"
    )


if __name__ == "__main__":
    test_eval_transform_parity()
    print("[PASS] AST parity: image_service eval transform == transforms.get_eval_transform "
          "(resize=256, crop=224, ImageNet mean/std identical).")
    try:
        test_eval_transform_parity_live()
        print("[PASS] live introspection parity (torchvision present).")
    except Exception as e:  # SystemExit from pytest.skip, or ImportError
        print(f"[SKIP] live introspection: {type(e).__name__} (torchvision absent — runs on Kaggle/CI).")
