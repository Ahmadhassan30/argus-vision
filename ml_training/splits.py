"""Lesion-grouped train/validation splitting for Argus Vision.

Every notebook that splits the ISIC-2019 data must do so **grouped by lesion**,
never by image: multiple dermoscopic images of the *same* lesion sharing a
`lesion_id` must land on the same side of the split, or validation scores are
inflated by leakage. This module is the single, shared implementation imported
by notebooks 01–05.

Design notes
------------
* The grouping unit is the **lesion**. Rows whose ``lesion_id`` is present and
  non-null are grouped by it; rows with a missing ``lesion_id`` fall back to
  using their own image id as a singleton group (an image that is its own group
  of one cannot leak). The fraction of rows in each case is printed loudly so the
  real coverage of the grouping protection is visible on Kaggle, not silent.
* The primary splitter is scikit-learn's :class:`StratifiedGroupKFold` (stratified
  *and* group-aware, so rare-class validation counts stay non-trivial), falling
  back to :class:`GroupShuffleSplit`. When scikit-learn is unavailable (e.g. a
  minimal local environment used only for logic verification), a deterministic
  pure-Python grouped splitter is used instead. **On Kaggle, sklearn is present,
  so the sklearn path is what actually runs the real split.**
* The metadata file ``ISIC_2019_Training_Metadata.csv`` (which carries
  ``lesion_id``) is NOT in the repo — it is mounted on Kaggle. :func:`attach_lesion_ids`
  discovers and merges it at runtime; if it is absent it adds an all-null
  ``lesion_id`` column with a loud warning (the split then degrades to image-level
  grouping, i.e. no lesion protection — but still leak-free by construction).

`label_col` is intentionally **required** (no default) so each call site passes
its own real column name (`01`/`04` use ``"_label"``, `02`/`05` use ``"label"``).
"""

from __future__ import annotations

import os
import random
from typing import Optional, Sequence

DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_SEED: int = 42

# Common column names seen across the notebooks / metadata files.
_IMAGE_COL_CANDIDATES = ("image", "image_id", "image_name", "img", "isic_id", "name")
_LESION_COL_CANDIDATES = ("lesion_id", "lesion", "lesionid")

# scikit-learn is the PRIMARY splitter and is ALWAYS used whenever it is installed.
# Detect it ONCE here via ImportError. The deterministic pure-Python fallback
# (_pure_python_group_split) is reachable ONLY when this import fails — i.e. sklearn is
# genuinely absent (a minimal local env). A *runtime* error inside a sklearn splitter is
# NEVER silently downgraded to the pure-Python path; it propagates (see lesion_grouped_indices).
try:
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal local envs
    StratifiedGroupKFold = None  # type: ignore[assignment,misc]
    GroupShuffleSplit = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Small helpers (stdlib only, so the core is verifiable without numpy/pandas).
# --------------------------------------------------------------------------- #
def _is_present(value) -> bool:
    """True if a lesion id is a real, usable value (not None/NaN/empty)."""
    if value is None:
        return False
    # NaN is the only value that is not equal to itself.
    if isinstance(value, float) and value != value:
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def _norm_image_id(value) -> str:
    """Normalize an image id for joining (strip directory + extension, lower)."""
    text = str(value).strip()
    text = os.path.basename(text)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if text.lower().endswith(ext):
            text = text[: -len(ext)]
            break
    return text.lower()


def build_group_keys(
    lesion_ids: Sequence,
    image_ids: Sequence,
) -> tuple[list[str], dict]:
    """Build the leak-proof grouping key for each row.

    Key is ``"L:<lesion_id>"`` when the lesion id is present, else ``"I:<image_id>"``
    (a singleton group). Returns ``(keys, stats)`` where ``stats`` records how many
    rows used each path.

    Args:
        lesion_ids: Per-row lesion id (may contain None/NaN for missing).
        image_ids: Per-row image id (used for the singleton fallback).

    Returns:
        A tuple of the per-row key list and a coverage-stats dict.
    """
    keys: list[str] = []
    n_present = 0
    n_missing = 0
    for lid, iid in zip(lesion_ids, image_ids):
        if _is_present(lid):
            keys.append("L:" + str(lid).strip())
            n_present += 1
        else:
            keys.append("I:" + _norm_image_id(iid))
            n_missing += 1
    total = len(keys)
    # Lesions with >1 image were the rows "at risk" under the old image-level split.
    lesion_to_count: dict[str, int] = {}
    for k in keys:
        if k.startswith("L:"):
            lesion_to_count[k] = lesion_to_count.get(k, 0) + 1
    at_risk_lesions = sum(1 for c in lesion_to_count.values() if c > 1)
    at_risk_images = sum(c for c in lesion_to_count.values() if c > 1)
    stats = {
        "total_rows": total,
        "rows_with_lesion_id": n_present,
        "rows_without_lesion_id": n_missing,
        "pct_with_lesion_id": (100.0 * n_present / total) if total else 0.0,
        "unique_groups": len(set(keys)),
        "unique_lesions": len(lesion_to_count),
        "at_risk_lesions": at_risk_lesions,
        "at_risk_images": at_risk_images,
    }
    return keys, stats


def _pure_python_group_split(
    keys: list[str],
    labels: Sequence,
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministic, stratified, group-aware split (stdlib only).

    Used only when scikit-learn is unavailable. Assigns whole groups (never
    splitting a lesion) to train/val, class by class, targeting ``test_size`` of
    each class's images for validation. Classes with a single group go to train
    (so no class loses all its training data); classes with >=2 groups keep at
    least one group on each side.
    """
    # group -> {idxs, label}
    groups: dict[str, dict] = {}
    for i, (k, lab) in enumerate(zip(keys, labels)):
        g = groups.setdefault(k, {"idxs": [], "label": lab})
        g["idxs"].append(i)

    by_label: dict[object, list[str]] = {}
    for k, info in groups.items():
        by_label.setdefault(info["label"], []).append(k)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for label, gks in sorted(by_label.items(), key=lambda kv: str(kv[0])):
        gks_sorted = sorted(gks)  # stable before shuffle
        rng.shuffle(gks_sorted)
        total_imgs = sum(len(groups[k]["idxs"]) for k in gks_sorted)
        target = test_size * total_imgs

        if len(gks_sorted) == 1:
            # Single lesion for this class — keep it in train.
            train_idx.extend(groups[gks_sorted[0]]["idxs"])
            continue

        acc = 0
        n_val_groups = 0
        for j, k in enumerate(gks_sorted):
            sz = len(groups[k]["idxs"])
            groups_left = len(gks_sorted) - j
            need_one_for_train = (groups_left + n_val_groups) <= len(gks_sorted) and (
                len(gks_sorted) - n_val_groups
            ) <= 1
            if acc < target and not need_one_for_train:
                val_idx.extend(groups[k]["idxs"])
                acc += sz
                n_val_groups += 1
            else:
                train_idx.extend(groups[k]["idxs"])
        if n_val_groups == 0:  # safety: ensure >=1 val group when >=2 groups
            moved = groups[gks_sorted[0]]["idxs"]
            for i in moved:
                train_idx.remove(i)
            val_idx.extend(moved)

    return sorted(train_idx), sorted(val_idx)


def lesion_grouped_indices(
    image_ids: Sequence,
    labels: Sequence,
    lesion_ids: Optional[Sequence] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
    verbose: bool = True,
) -> tuple[list[int], list[int], dict]:
    """Core grouped split over plain sequences (no pandas required).

    Shared by both the DataFrame split (:func:`get_lesion_grouped_split`) and the
    array-based call sites (NB03 image arrays, NB04 consensus-feature indices).

    Args:
        image_ids: Per-row image ids (singleton-group fallback key).
        labels: Per-row class labels (ints or strings).
        lesion_ids: Per-row lesion ids, or ``None`` to group purely by image id.
        test_size: Target validation fraction (by image count).
        random_state: Seed for deterministic shuffling.
        verbose: Print the coverage / class-balance summary.

    Returns:
        ``(train_idx, val_idx, stats)``.
    """
    n = len(image_ids)
    if lesion_ids is None:
        lesion_ids = [None] * n
    keys, stats = build_group_keys(lesion_ids, image_ids)

    train_idx: list[int]
    val_idx: list[int]
    method: str
    if _SKLEARN_AVAILABLE:
        # sklearn is installed → it is ALWAYS used. _pure_python_group_split is
        # unreachable in this branch.
        try:
            n_splits = max(2, round(1.0 / test_size))
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            tr, va = next(sgkf.split(list(range(n)), list(labels), groups=keys))
            method = f"StratifiedGroupKFold(n_splits={n_splits})"
        except Exception as exc:  # noqa: BLE001 - stratified split can fail on sparse classes
            # StratifiedGroupKFold can legitimately fail when a class has fewer groups
            # than n_splits. Fall back to GroupShuffleSplit — STILL sklearn, still
            # group-safe. Any failure HERE propagates: we never silently downgrade to the
            # pure-Python splitter while sklearn is installed.
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            tr, va = next(gss.split(list(range(n)), list(labels), groups=keys))
            method = (
                "GroupShuffleSplit (StratifiedGroupKFold unavailable for this data: "
                f"{type(exc).__name__})"
            )
        train_idx, val_idx = sorted(tr.tolist()), sorted(va.tolist())
    else:
        # sklearn genuinely absent → deterministic stdlib fallback (local verification only).
        train_idx, val_idx = _pure_python_group_split(
            keys, labels, test_size, random_state
        )
        method = "pure-python-grouped (sklearn NOT installed)"

    stats["method"] = method
    stats["n_train"] = len(train_idx)
    stats["n_val"] = len(val_idx)

    # Per-class train/val counts.
    train_set = set(train_idx)
    per_class: dict[object, list[int]] = {}
    for i, lab in enumerate(labels):
        row = per_class.setdefault(lab, [0, 0])
        if i in train_set:
            row[0] += 1
        else:
            row[1] += 1
    stats["per_class_counts"] = {k: tuple(v) for k, v in per_class.items()}

    # Hard guarantee: no group spans both sides.
    train_keys = {keys[i] for i in train_idx}
    val_keys = {keys[i] for i in val_idx}
    overlap = train_keys & val_keys
    stats["group_overlap"] = len(overlap)

    if verbose:
        _print_summary(stats)
    return train_idx, val_idx, stats


def _print_summary(stats: dict) -> None:
    """Print the coverage / leakage / class-balance summary."""
    print("── Lesion-grouped split ─────────────────────────────────────────")
    print(f"  method                : {stats.get('method')}")
    print(
        f"  rows                  : {stats['total_rows']}  "
        f"(train={stats.get('n_train', '?')}, val={stats.get('n_val', '?')})"
    )
    print(
        f"  lesion_id present     : {stats['rows_with_lesion_id']} "
        f"({stats['pct_with_lesion_id']:.1f}%)  |  "
        f"fallback to image-id  : {stats['rows_without_lesion_id']}"
    )
    print(
        f"  unique lesions        : {stats['unique_lesions']}  |  "
        f"unique groups: {stats['unique_groups']}"
    )
    print(
        f"  lesions w/ >1 image   : {stats['at_risk_lesions']} "
        f"({stats['at_risk_images']} images 'at risk' under the old image-level split)"
    )
    print(f"  group overlap (must be 0): {stats['group_overlap']}")
    pcc = stats.get("per_class_counts", {})
    if pcc:
        print("  per-class (train/val) :")
        for k in sorted(pcc, key=lambda x: str(x)):
            tr, va = pcc[k]
            print(f"      {k!s:>6}: train={tr:>6}  val={va:>5}")
    print("─────────────────────────────────────────────────────────────────")


def get_lesion_grouped_split(
    df,
    label_col: str,
    lesion_id_col: str = "lesion_id",
    image_col: Optional[str] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
    verbose: bool = True,
):
    """Return ``(train_df, val_df)`` with zero lesion overlap.

    Imported identically by every notebook that splits data. ``label_col`` is
    required — pass the column actually used by that notebook.

    Args:
        df: Source DataFrame (one row per image).
        label_col: Name of the integer/string label column (REQUIRED).
        lesion_id_col: Name of the lesion-id column (added by
            :func:`attach_lesion_ids`; may be all-null if no metadata is present).
        image_col: Image-id column; auto-detected from common names when ``None``.
        test_size: Target validation fraction.
        random_state: Deterministic seed.
        verbose: Print the coverage / class-balance summary.

    Returns:
        ``(train_df, val_df)``, each a reset-index copy carrying an added
        ``lesion_group`` column (the effective leak-proof grouping key).
    """
    if image_col is None:
        image_col = _detect_col(df, _IMAGE_COL_CANDIDATES)
    if image_col is None:
        raise ValueError(
            "Could not detect an image-id column; pass image_col explicitly. "
            f"Looked for {_IMAGE_COL_CANDIDATES}."
        )

    image_ids = list(df[image_col])
    labels = list(df[label_col])
    lesion_ids = list(df[lesion_id_col]) if lesion_id_col in df.columns else None

    train_idx, val_idx, stats = lesion_grouped_indices(
        image_ids, labels, lesion_ids, test_size, random_state, verbose
    )

    keys, _ = build_group_keys(
        lesion_ids if lesion_ids is not None else [None] * len(df), image_ids
    )
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    train_df["lesion_group"] = [keys[i] for i in train_idx]
    val_df["lesion_group"] = [keys[i] for i in val_idx]
    return train_df, val_df


def assert_no_lesion_leakage(
    train_df,
    val_df,
    lesion_id_col: str = "lesion_id",
    group_col: str = "lesion_group",
) -> None:
    """Hard assertion that no lesion (or group) spans train and val.

    Asserts on the effective ``lesion_group`` key when present (the correct,
    null-safe unit), and additionally on non-null raw ``lesion_id`` values.
    """
    if group_col in train_df.columns and group_col in val_df.columns:
        shared_groups = set(train_df[group_col]) & set(val_df[group_col])
        assert not shared_groups, (
            f"Lesion-group leakage detected: {len(shared_groups)} shared groups, "
            f"e.g. {list(shared_groups)[:5]}"
        )
    if lesion_id_col in train_df.columns and lesion_id_col in val_df.columns:
        tr = {v for v in train_df[lesion_id_col] if _is_present(v)}
        va = {v for v in val_df[lesion_id_col] if _is_present(v)}
        shared = tr & va
        assert not shared, (
            f"Lesion leakage detected between train and val splits: "
            f"{len(shared)} shared lesion_ids, e.g. {list(shared)[:5]}"
        )


def _detect_col(df, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate column present in ``df`` (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def attach_lesion_ids(
    df,
    image_col: Optional[str] = None,
    metadata_path: Optional[str] = None,
    search_roots: Sequence[str] = ("/kaggle/input", "/kaggle/working", "."),
    lesion_id_col: str = "lesion_id",
    verbose: bool = True,
):
    """Merge ``lesion_id`` into ``df`` from the ISIC-2019 metadata CSV.

    Discovers a metadata CSV (one containing a lesion-id column) under
    ``search_roots`` unless ``metadata_path`` is given, then left-joins
    ``lesion_id`` onto ``df`` by normalized image id. If no metadata is found, a
    loud warning is printed and an all-null ``lesion_id`` column is added so the
    downstream split still runs (degrading to leak-free image-level grouping).

    Returns the same DataFrame with a ``lesion_id`` column.
    """
    import pandas as pd  # local import: only needed when actually wiring notebooks

    if image_col is None:
        image_col = _detect_col(df, _IMAGE_COL_CANDIDATES)
    if image_col is None:
        raise ValueError("Could not detect the image-id column in df.")

    if metadata_path is None:
        metadata_path = _discover_metadata_csv(search_roots)

    if metadata_path is None:
        print(
            "⚠️  attach_lesion_ids: NO metadata CSV with a lesion_id column was "
            f"found under {list(search_roots)}.\n"
            "    lesion_id will be all-null and the split will degrade to "
            "IMAGE-LEVEL grouping (leak-free, but NO lesion protection).\n"
            "    Mount ISIC_2019_Training_Metadata.csv to enable lesion grouping."
        )
        df = df.copy()
        df[lesion_id_col] = None
        return df

    meta = pd.read_csv(metadata_path)
    meta_img = _detect_col(meta, _IMAGE_COL_CANDIDATES)
    meta_les = _detect_col(meta, _LESION_COL_CANDIDATES)
    if meta_img is None or meta_les is None:
        print(
            f"⚠️  attach_lesion_ids: metadata at '{metadata_path}' lacks an "
            f"image/lesion column (cols={list(meta.columns)}). Using image-level "
            "grouping."
        )
        df = df.copy()
        df[lesion_id_col] = None
        return df

    meta = meta[[meta_img, meta_les]].copy()
    meta["_img_key"] = meta[meta_img].map(_norm_image_id)
    meta = meta.drop_duplicates("_img_key")
    lookup = dict(zip(meta["_img_key"], meta[meta_les]))

    df = df.copy()
    df[lesion_id_col] = df[image_col].map(lambda x: lookup.get(_norm_image_id(x)))
    if verbose:
        n = len(df)
        present = int(sum(_is_present(v) for v in df[lesion_id_col]))
        print(
            f"attach_lesion_ids: merged lesion_id from '{os.path.basename(metadata_path)}' "
            f"— {present}/{n} ({100.0 * present / n:.1f}%) rows matched."
        )
    return df


def _discover_metadata_csv(search_roots: Sequence[str]) -> Optional[str]:
    """Find a CSV containing a lesion-id column under the given roots."""
    import csv

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.lower().endswith(".csv"):
                    continue
                if "metadata" not in name.lower() and "meta" not in name.lower():
                    # Cheap filter: still peek other csvs below if named oddly.
                    pass
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", newline="") as fh:
                        header = next(csv.reader(fh), [])
                    lower = {h.strip().lower() for h in header}
                    if any(c in lower for c in _LESION_COL_CANDIDATES):
                        return path
                except Exception:
                    continue
    return None
