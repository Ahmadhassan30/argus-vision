# ============================================================================
# Cell 2 — Install extras, imports, shared constants and discovery helpers
# ============================================================================
import sys, subprocess

# Kaggle pre-installs torch / torchvision / numpy / pandas / sklearn / matplotlib /
# seaborn / opencv-python / Pillow / tqdm. We add the extras THIS notebook needs.
# NOTE: Internet MUST be ON for these installs and for pretrained weight downloads.
print("Installing extras (Internet must be ON)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "timm"],
    check=False,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "grad-cam", "sentence-transformers", "netcal", "scipy"],
    check=False,
)
print("Extras installed (timm -U, grad-cam, sentence-transformers, netcal, scipy).")

import os
import re
import json
import glob
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- Shared contract
ISIC_CLASSES: List[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = ISIC_CLASSES  # alias used throughout
FULL_NAMES: Dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}
RISK_LEVELS: Dict[str, str] = {
    "MEL": "high", "NV": "low", "BCC": "high", "AK": "medium",
    "BKL": "low", "DF": "low", "VASC": "medium", "SCC": "high",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
NUM_CLASSES = 8
EMBEDDING_DIM = 384
FEATURE_DIM = 788  # 8 (pA) + 8 (pB) + 4 (spatial) + 384 (eA) + 384 (eB)

# Debate-trigger thresholds (identical to the backend settings).
TAU_JS = 0.25
TAU_ENTROPY = 0.8

# Backbone identifiers (MUST match the backend agents / NB01 + NB02).
AGENT_A_MODEL_NAME = "efficientnet_b4"
AGENT_B_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 32          # lower to 16 if OOM
NUM_WORKERS = 2

WORK_DIR = Path("/kaggle/working")
FIG_DIR = WORK_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Feature dimension: {FEATURE_DIM}")


# ---------------------------------------------------------------- discover_isic()
def discover_isic(root: str = "/kaggle/input") -> Tuple[str, str]:
    """Locate the ISIC-2019 ground-truth CSV and the image directory under /kaggle/input.

    The CSV is the .csv whose header contains ALL 8 ISIC class names. The image_dir
    is the directory containing the most ISIC_*.jpg/.jpeg files (case-insensitive).
    Robust to nested mirror folders (e.g. double-nested ISIC_2019_Training_Input/).
    """
    needed = set(c.upper() for c in ISIC_CLASSES)
    csv_path: Optional[str] = None
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                header = pd.read_csv(fpath, nrows=0)
            except Exception:
                continue
            cols = set(str(c).upper() for c in header.columns)
            if needed.issubset(cols):
                csv_path = fpath
                break
        if csv_path is not None:
            break

    # Find the directory holding the most ISIC_*.jpg/.jpeg images.
    best_dir: Optional[str] = None
    best_count = -1
    for dirpath, _dirs, files in os.walk(root):
        cnt = 0
        for fname in files:
            low = fname.lower()
            if low.startswith("isic_") and (low.endswith(".jpg") or low.endswith(".jpeg")):
                cnt += 1
        if cnt > best_count:
            best_count = cnt
            best_dir = dirpath
    image_dir = best_dir

    print(f"discover_isic -> csv_path = {csv_path}")
    print(f"discover_isic -> image_dir = {image_dir}  ({best_count} ISIC_*.jpg/.jpeg files)")
    assert csv_path is not None, "Ground-truth CSV with all 8 ISIC class columns not found under /kaggle/input."
    assert image_dir is not None and best_count > 0, "No ISIC_*.jpg image directory found under /kaggle/input."
    assert os.path.exists(csv_path) and os.path.isdir(image_dir)
    return csv_path, image_dir


# ---------------------------------------------------------------- find_file()
def find_file(filename_substring: str,
              search_roots: Tuple[str, ...] = ("/kaggle/input", "/kaggle/working")) -> Optional[str]:
    """Return the first path whose basename contains `filename_substring`, else None."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                if filename_substring in fname:
                    found = os.path.join(dirpath, fname)
                    print(f"find_file('{filename_substring}') -> {found}")
                    return found
    print(f"find_file('{filename_substring}') -> NOT FOUND under {search_roots}.")
    return None


CSV_PATH, IMAGE_DIR = discover_isic()
print("Setup complete.")
---
# ============================================================================
# Cell 3 — Models: Agent A (EfficientNet-B4), Agent B (ViT-B/16),
#          ConsensusMLP (+ temperature), and the argument encoder.
# ============================================================================
import timm


class ConsensusMLP(nn.Module):
    """Calibrated consensus classifier over the 788-dim debate feature vector.

    Maps [pA(8), pB(8), spatial_stats(4), eA(384), eB(384)] -> 8 logits via two
    hidden layers (Linear -> BatchNorm1d -> ReLU -> Dropout). A learnable scalar
    temperature divides the logits before softmax for calibration.
    """

    def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features) / self.temperature.clamp_min(1e-3)


def build_agent(model_name: str, ckpt_substring: str) -> Tuple[nn.Module, bool]:
    """Create a timm backbone and load its fine-tuned checkpoint if found."""
    ckpt = find_file(ckpt_substring)
    has_ckpt = ckpt is not None
    model = timm.create_model(model_name, pretrained=not has_ckpt, num_classes=NUM_CLASSES)
    if has_ckpt:
        state = torch.load(ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded {os.path.basename(ckpt)} (missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        print(f"WARNING: no checkpoint matching '{ckpt_substring}'; using ImageNet fallback.")
    return model.eval().to(DEVICE), has_ckpt


agent_a, A_HAS_CKPT = build_agent(AGENT_A_MODEL_NAME, "agent_a_best")
agent_b, B_HAS_CKPT = build_agent(AGENT_B_MODEL_NAME, "agent_b_best")

# ---- Consensus MLP + temperature -------------------------------------------
consensus = ConsensusMLP().to(DEVICE)
CONSENSUS_HAS_CKPT = False
consensus_ckpt = find_file("consensus_best")
if consensus_ckpt is not None:
    cstate = torch.load(consensus_ckpt, map_location=DEVICE)
    if isinstance(cstate, dict) and "state_dict" in cstate:
        cstate = cstate["state_dict"]
    consensus.load_state_dict(cstate, strict=False)
    CONSENSUS_HAS_CKPT = True
    print(f"Loaded consensus checkpoint: {os.path.basename(consensus_ckpt)}.")
else:
    print("WARNING: consensus_best.pth not found; consensus head is randomly initialised.")
consensus.eval()

# Temperature sidecar (consensus_temperature.json) — overrides the in-state value if present.
CONSENSUS_TEMPERATURE = float(consensus.temperature.detach().cpu().item())
temp_json = find_file("consensus_temperature")
if temp_json is not None:
    try:
        with open(temp_json, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        try:
            CONSENSUS_TEMPERATURE = float(content)
        except ValueError:
            tmeta = json.loads(content)
            if isinstance(tmeta, dict):
                CONSENSUS_TEMPERATURE = float(tmeta.get("temperature", CONSENSUS_TEMPERATURE))
            else:
                CONSENSUS_TEMPERATURE = float(tmeta)
        with torch.no_grad():
            consensus.temperature.fill_(CONSENSUS_TEMPERATURE)
        print(f"Applied temperature from sidecar: T={CONSENSUS_TEMPERATURE:.4f}")
    except Exception as exc:
        print(f"Could not read temperature sidecar ({exc}); using T={CONSENSUS_TEMPERATURE:.4f}.")
else:
    print(f"No temperature sidecar; using in-checkpoint T={CONSENSUS_TEMPERATURE:.4f}.")

# ---- Argument encoder (all-MiniLM-L6-v2, 384-dim) --------------------------
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


def encode_argument(text: str) -> np.ndarray:
    """Encode an argument string into a 384-d L2-normalised embedding."""
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec = encoder.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


print("Encoder loaded; embedding dim:", encode_argument("test argument").shape[0])
print("All models ready on", DEVICE)
---
# ============================================================================
# Cell 4 — ISIC dataframe + stratified test split + eval transform / loaders
# ============================================================================
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Read the one-hot ground-truth CSV; label = argmax over the 8 class columns.
gt = pd.read_csv(CSV_PATH)
image_col = "image" if "image" in gt.columns else gt.columns[0]
col_upper = {str(c).upper(): c for c in gt.columns}
onehot_cols = [col_upper[c.upper()] for c in ISIC_CLASSES]
labels_all = gt[onehot_cols].to_numpy().argmax(axis=1).astype(int)


def resolve_image_path(stem: str) -> str:
    """Resolve an ISIC image stem to a path inside IMAGE_DIR (tries .jpg/.jpeg)."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = os.path.join(IMAGE_DIR, str(stem) + ext)
        if os.path.isfile(p):
            return p
    return os.path.join(IMAGE_DIR, str(stem) + ".jpg")


records = []
for i in range(len(gt)):
    stem = str(gt.iloc[i][image_col])
    records.append({"image_path": resolve_image_path(stem),
                    "image_id": stem,
                    "label": int(labels_all[i])})
full_df = pd.DataFrame.from_records(records).reset_index(drop=True)

# Keep only rows whose image actually exists on disk.
exists_mask = full_df["image_path"].map(os.path.isfile)
n_missing = int((~exists_mask).sum())
if n_missing:
    print(f"Note: dropping {n_missing} rows whose image file was not found on disk.")
full_df = full_df[exists_mask].reset_index(drop=True)
print(f"Usable labelled images: {len(full_df):,}")

train_df, test_df = train_test_split(
    full_df, test_size=0.15, stratify=full_df["label"], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Train: {len(train_df):,} | Test (held-out 15%): {len(test_df):,}")

eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ISICEvalDataset(Dataset):
    """Returns (tensor, label_int, image_path) for the eval split."""

    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["image_path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as h:
            img = h.convert("RGB")
        return self.transform(img), int(self.labels[idx]), path


def load_tensor(image_path: str) -> torch.Tensor:
    """Load one image as a (1,3,224,224) eval tensor (used for single-image debate)."""
    with Image.open(image_path) as h:
        img = h.convert("RGB")
    return eval_transform(img).unsqueeze(0)


# Optional cap to keep the full-split evaluation tractable on a Kaggle session.
# Set MAX_EVAL_IMAGES = None to evaluate the entire test split.
MAX_EVAL_IMAGES = 2000
if MAX_EVAL_IMAGES is not None and len(test_df) > MAX_EVAL_IMAGES:
    eval_df = test_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, MAX_EVAL_IMAGES // NUM_CLASSES)),
                            random_state=SEED)
    ).reset_index(drop=True)
    print(f"Evaluating on a stratified subset of {len(eval_df):,} test images "
          f"(set MAX_EVAL_IMAGES=None for the full split).")
else:
    eval_df = test_df.copy()
    print(f"Evaluating on the full test split: {len(eval_df):,} images.")

eval_ds = ISICEvalDataset(eval_df, eval_transform)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
print("Eval loader ready.")
---
# ============================================================================
# Cell 5 — Attention (Grad-CAM++ / rollout), disagreement map M_delta, trigger,
#          cached-or-fallback debate arguments, and the 788-dim feature builder.
# ============================================================================
import cv2
from scipy.spatial.distance import jensenshannon
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_EPS = 1e-12
GRID_SIZE = 14
NUM_PATCH_TOKENS = GRID_SIZE * GRID_SIZE


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 0.0, None)
    return -float(np.sum(p * np.log2(p + _EPS)))


def js_divergence(pa: np.ndarray, pb: np.ndarray) -> float:
    d = jensenshannon(pa, pb, base=2)
    div = float(d) ** 2
    return div if np.isfinite(div) else 0.0


def trigger_fires(div: float, ent_a: float, ent_b: float) -> bool:
    return (div > TAU_JS) or (max(ent_a, ent_b) > TAU_ENTROPY)


@torch.no_grad()
def predict_probs(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    logits = model(tensor.to(DEVICE))
    return F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float64)


def gradcam_plusplus(model: nn.Module, tensor: torch.Tensor, target: int) -> np.ndarray:
    """224x224 Grad-CAM++ saliency map for Agent A (EfficientNet-B4), in [0,1]."""
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        last = blocks[-1]
        target_layer = getattr(last, "bn3", last)
    else:
        target_layer = getattr(model, "conv_head", None)
        if target_layer is None:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.Conv2d)):
                    target_layer = m
    model.zero_grad()
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=tensor.to(DEVICE), targets=[ClassifierOutputTarget(target)])
    model.zero_grad()
    hm = np.asarray(grayscale[0], dtype=np.float32)
    if hm.shape != (IMAGE_SIZE, IMAGE_SIZE):
        hm = cv2.resize(hm, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(hm, 0.0, 1.0).astype(np.float32)


def attention_rollout(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """224x224 attention-rollout saliency map for Agent B (ViT-B/16), in [0,1]."""
    handles: List[Any] = []
    captured: Dict[int, torch.Tensor] = {}
    original_fused: Dict[int, bool] = {}
    blocks = getattr(model, "blocks", None)

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            x = inputs[0]
            batch, num_tokens, dim = x.shape
            heads = int(module.num_heads)
            head_dim = dim // heads
            scale = getattr(module, "scale", head_dim ** -0.5)
            qkv = module.qkv(x).reshape(batch, num_tokens, 3, heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * float(scale)
            attn = attn.softmax(dim=-1)
            captured[layer_idx] = attn.mean(dim=1).detach().to(torch.float32).cpu()
        return hook

    try:
        for idx, block in enumerate(blocks):
            am = getattr(block, "attn", None)
            if am is None or not hasattr(am, "qkv"):
                continue
            original_fused[idx] = bool(getattr(am, "fused_attn", False))
            if hasattr(am, "fused_attn"):
                am.fused_attn = False
            handles.append(am.register_forward_hook(make_hook(idx)))
        with torch.no_grad():
            model(tensor.to(DEVICE))
        layer_indices = sorted(captured.keys())
        num_tokens = captured[layer_indices[0]].shape[-1]
        identity = torch.eye(num_tokens, dtype=torch.float32)
        rollout = torch.eye(num_tokens, dtype=torch.float32)
        for idx in layer_indices:
            attn = captured[idx][0]
            aug = 0.5 * attn + 0.5 * identity
            aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rollout = aug @ rollout
        cls_attention = rollout[0, 1:1 + NUM_PATCH_TOKENS]
        grid = cls_attention.reshape(GRID_SIZE, GRID_SIZE).numpy().astype(np.float32)
        hm = cv2.resize(grid, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        span = float(hm.max() - hm.min())
        if span < 1e-12:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return ((hm - hm.min()) / span).astype(np.float32)
    finally:
        for h in handles:
            h.remove()
        for idx, was_fused in original_fused.items():
            am = getattr(blocks[idx], "attn", None)
            if am is not None and hasattr(am, "fused_attn"):
                am.fused_attn = was_fused


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    span = float(arr.max() - arr.min())
    if span < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr.min()) / span).astype(np.float32)


def disagreement_map(heatmap_a: np.ndarray, heatmap_b: np.ndarray) -> np.ndarray:
    """The M_delta map: |normalised(A) - normalised(B)| in [0,1]."""
    return np.abs(_normalize_map(heatmap_a) - _normalize_map(heatmap_b)).astype(np.float32)


def compute_disagreement(heatmap_a: np.ndarray, heatmap_b: np.ndarray):
    """Contested-region (top-20%-mass) stats for both agents: mean/std/max."""
    norm_a = _normalize_map(heatmap_a)
    norm_b = _normalize_map(heatmap_b)
    combined = norm_a + norm_b
    threshold = float(np.percentile(combined, 80.0))
    mask = combined >= threshold
    if not bool(mask.any()):
        mask = np.ones_like(combined, dtype=bool)

    def region_stats(norm_map):
        sel = norm_map[mask]
        if sel.size == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        return {"mean": float(sel.mean()), "std": float(sel.std()), "max": float(sel.max())}

    return region_stats(norm_a), region_stats(norm_b)


# ---------------------------------------------------------------- Debate arguments
ISIC_CLASS_DESCRIPTIONS: Dict[str, str] = {
    "MEL": "Melanoma typically shows an atypical, broadened pigment network with irregular streaks, "
           "asymmetry of structure and colour, and frequent regression areas. A blue-white veil and "
           "chaotic vessels support malignancy.",
    "NV": "A melanocytic nevus is characterised by a symmetric, regularly spaced reticular or globular "
          "pattern with uniform colouration and a smooth transition to surrounding skin.",
    "BCC": "Basal cell carcinoma is defined by arborising (tree-like) vessels and blue-grey ovoid nests "
           "on a pigment-network-free background, with leaf-like areas and spoke-wheel structures.",
    "AK": "Actinic keratosis shows a 'strawberry' pattern: a red pseudo-network of dilated vessels around "
          "keratin-plugged follicular openings on a scaly, erythematous background.",
    "BKL": "Benign keratosis displays a cerebriform 'brain-like' surface with milia-like cysts and "
           "comedo-like openings, sharply demarcated borders and a stuck-on appearance.",
    "DF": "Dermatofibroma presents with a central white scar-like patch surrounded by a delicate peripheral "
          "pigment network and a homogeneous tan-brown ring.",
    "VASC": "Vascular lesions are recognised by sharply demarcated red, purple, or maroon lacunae separated "
            "by pale septa, with no melanocytic pigment network.",
    "SCC": "Squamous cell carcinoma shows central keratin masses, white circles around follicular openings, "
           "surface scale/ulceration, and looped or glomerular vessels at the periphery.",
}
_DELTA_PATTERN = re.compile(r"CONFIDENCE_DELTA\s*:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)

# Load a cached Groq transcript file if one was attached (preferred over live calls / fallback).
GROQ_CACHE: Dict[str, Dict[str, Any]] = {}
cache_path = find_file("groq_argument_cache")
if cache_path is None:
    cache_path = find_file("groq_cache_v2")
if cache_path is None:
    cache_path = find_file("groq_cache")
if cache_path is not None:
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            GROQ_CACHE = json.load(fh)
        print(f"Loaded {len(GROQ_CACHE)} cached debate transcripts from {os.path.basename(cache_path)}.")
    except Exception as exc:
        print(f"Could not read cached transcripts ({exc}); using deterministic fallback.")
else:
    print("No cached groq transcripts attached; using the deterministic offline fallback.")

# Optional live Groq client (only if GROQ_API_KEY is set as a Kaggle secret).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialised (live argument generation enabled).")
    except Exception as exc:
        print(f"Could not init Groq client ({exc}); using cache/fallback only.")


def _region_summary(stats: Dict[str, float]) -> str:
    return (f"In the contested region your attention map has mean {stats.get('mean', 0.0):.3f}, "
            f"std {stats.get('std', 0.0):.3f}, peak {stats.get('max', 0.0):.3f}.")


def _fallback_argument(pred: str, conf: float, stats: Dict[str, float], opponent: str) -> str:
    return (f"This lesion is most consistent with {FULL_NAMES[pred]} ({pred}) at {conf * 100:.1f}% "
            f"confidence. {ISIC_CLASS_DESCRIPTIONS[pred]} {_region_summary(stats)} These features are "
            f"inconsistent with {FULL_NAMES.get(opponent, opponent)} ({opponent}).")


def get_debate_transcript(image_id: str, pred_a: str, conf_a: float, pred_b: str, conf_b: float,
                          stats_a: Dict[str, float], stats_b: Dict[str, float]) -> Dict[str, str]:
    """Reuse the cached transcript for image_id if present; else deterministic fallback."""
    if image_id in GROQ_CACHE:
        return GROQ_CACHE[image_id]
    transcript = {
        "arg_a_r1": _fallback_argument(pred_a, conf_a, stats_a, pred_b),
        "arg_b_r1": _fallback_argument(pred_b, conf_b, stats_b, pred_a),
        "arg_a_r2": _fallback_argument(pred_a, conf_a, stats_a, pred_b) + "\nCONFIDENCE_DELTA: 0.0",
        "arg_b_r2": _fallback_argument(pred_b, conf_b, stats_b, pred_a) + "\nCONFIDENCE_DELTA: 0.0",
    }
    return transcript


def strip_delta(text: str) -> str:
    return _DELTA_PATTERN.sub("", text).strip()


def build_feature(pa, pb, stats_a, stats_b, emb_a, emb_b) -> np.ndarray:
    """Concatenate the 788-dim consensus feature: [pA, pB, spatial(4), eA, eB]."""
    spatial_stats = np.array([stats_a["mean"], stats_b["mean"], stats_a["std"], stats_b["std"]],
                             dtype=np.float32)
    feat = np.concatenate([pa.astype(np.float32), pb.astype(np.float32), spatial_stats,
                           emb_a.astype(np.float32), emb_b.astype(np.float32)])
    assert feat.shape == (FEATURE_DIM,), feat.shape
    return feat


print("Debate / attention / feature helpers ready.")
---
# ============================================================================
# Cell 6 — Run inference for all five configurations on the eval split.
# ============================================================================
from tqdm.auto import tqdm

# --- Detect an optional second checkpoint set for a DEEP ENSEMBLE -----------
deep_a_path = find_file("agent_a_seed")
deep_b_path = find_file("agent_b_seed")
DEEP_ENSEMBLE_AVAILABLE = (deep_a_path is not None) and (deep_b_path is not None)
deep_agent_a = deep_agent_b = None
if DEEP_ENSEMBLE_AVAILABLE:
    try:
        da = timm.create_model(AGENT_A_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        sa = torch.load(deep_a_path, map_location=DEVICE)
        da.load_state_dict(sa.get("state_dict", sa) if isinstance(sa, dict) else sa, strict=False)
        deep_agent_a = da.eval().to(DEVICE)
        db = timm.create_model(AGENT_B_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        sb = torch.load(deep_b_path, map_location=DEVICE)
        db.load_state_dict(sb.get("state_dict", sb) if isinstance(sb, dict) else sb, strict=False)
        deep_agent_b = db.eval().to(DEVICE)
        print("Deep-ensemble second checkpoints loaded.")
    except Exception as exc:
        print(f"Deep-ensemble load failed ({exc}); will SKIP gracefully.")
        DEEP_ENSEMBLE_AVAILABLE = False
else:
    print("Deep-ensemble checkpoints (agent_*_seed*) not attached -> SKIPPING that config "
          "gracefully (column mirrors the standard ensemble and is flagged in notes).")

CONFIG_NAMES = ["Agent A", "Agent B", "Standard Ensemble", "Deep Ensemble", "Argus (full)"]

# Accumulators: per-config predicted probability matrices + true labels + bookkeeping.
y_true: List[int] = []
image_paths: List[str] = []
proba = {name: [] for name in CONFIG_NAMES}

# Per-image debate bookkeeping (for the case studies + ablation in later cells).
debate_records: List[Dict[str, Any]] = []


@torch.no_grad()
def batch_probs(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    return F.softmax(model(x.to(DEVICE)), dim=1).detach().cpu().numpy().astype(np.float64)


def argus_predict(image_path: str, pa: np.ndarray, pb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the Argus predictive distribution + a debate record for one image."""
    div = js_divergence(pa, pb)
    ent_a, ent_b = shannon_entropy(pa), shannon_entropy(pb)
    fired = trigger_fires(div, ent_a, ent_b)
    rec: Dict[str, Any] = {
        "image_path": image_path, "image_id": Path(image_path).stem,
        "js": div, "ent_a": ent_a, "ent_b": ent_b, "fired": bool(fired),
        "pa": pa, "pb": pb,
    }
    if not fired:
        rec["consensus_prob"] = None
        return 0.5 * (pa + pb), rec

    tensor = load_tensor(image_path)
    hm_a = gradcam_plusplus(agent_a, tensor, int(pa.argmax()))
    hm_b = attention_rollout(agent_b, tensor)
    stats_a, stats_b = compute_disagreement(hm_a, hm_b)
    transcript = get_debate_transcript(
        rec["image_id"], CLASS_NAMES[int(pa.argmax())], float(pa.max()),
        CLASS_NAMES[int(pb.argmax())], float(pb.max()), stats_a, stats_b)
    emb_a = encode_argument(strip_delta(transcript["arg_a_r2"]))
    emb_b = encode_argument(strip_delta(transcript["arg_b_r2"]))
    feat = build_feature(pa, pb, stats_a, stats_b, emb_a, emb_b)

    consensus.eval()
    with torch.no_grad():
        ft = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
        cprob = F.softmax(consensus(ft), dim=1)[0].detach().cpu().numpy().astype(np.float64)
    rec.update({"stats_a": stats_a, "stats_b": stats_b, "transcript": transcript,
                "consensus_prob": cprob})
    return cprob, rec


# ---- Main loop: batched agent probs, then per-image Argus path -------------
print("Running five-configuration evaluation...")
for x, yb, paths in tqdm(eval_loader, total=len(eval_loader)):
    pA = batch_probs(agent_a, x)
    pB = batch_probs(agent_b, x)
    if DEEP_ENSEMBLE_AVAILABLE:
        pA2 = batch_probs(deep_agent_a, x)
        pB2 = batch_probs(deep_agent_b, x)

    for j in range(x.shape[0]):
        pa, pb = pA[j], pB[j]
        path = paths[j]
        y_true.append(int(yb[j]))
        image_paths.append(path)

        proba["Agent A"].append(pa)
        proba["Agent B"].append(pb)
        std_ens = 0.5 * (pa + pb)
        proba["Standard Ensemble"].append(std_ens)

        if DEEP_ENSEMBLE_AVAILABLE:
            proba["Deep Ensemble"].append(0.25 * (pa + pb + pA2[j] + pB2[j]))
        else:
            proba["Deep Ensemble"].append(std_ens)  # mirrors standard ensemble; flagged in notes

        argus_p, rec = argus_predict(path, pa, pb)
        proba["Argus (full)"].append(argus_p)
        debate_records.append(rec)

y_true = np.asarray(y_true, dtype=np.int64)
for name in CONFIG_NAMES:
    proba[name] = np.asarray(proba[name], dtype=np.float64)

n_fired = int(sum(r["fired"] for r in debate_records))
print(f"Done. Evaluated {len(y_true):,} images; debate trigger fired on "
      f"{n_fired:,} ({n_fired / max(len(y_true),1):.1%}).")
---
# ============================================================================
# Cell 7 — Metric helpers + the per-configuration metrics table.
# ============================================================================
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    from netcal.metrics import ECE as _NetcalECE
    _HAVE_NETCAL = True
except Exception:
    _HAVE_NETCAL = False


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the predicted top-class confidence. Uses netcal if available."""
    if _HAVE_NETCAL:
        try:
            return float(_NetcalECE(n_bins).measure(probs.astype(np.float64), labels.astype(int)))
        except Exception:
            pass
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def macro_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    present = np.unique(labels)
    if len(present) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs, multi_class="ovr",
                                   average="macro", labels=list(range(NUM_CLASSES))))
    except Exception:
        aucs = []
        for c in range(NUM_CLASSES):
            yc = (labels == c).astype(int)
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            aucs.append(roc_auc_score(yc, probs[:, c]))
        return float(np.mean(aucs)) if aucs else float("nan")


def metrics_for(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "Balanced Accuracy": float(balanced_accuracy_score(labels, preds)),
        "Macro AUC": macro_auc(probs, labels),
        "ECE": expected_calibration_error(probs, labels),
    }


rows = []
for name in CONFIG_NAMES:
    m = metrics_for(proba[name], y_true)
    note = ""
    if name == "Deep Ensemble" and not DEEP_ENSEMBLE_AVAILABLE:
        note = "SKIPPED (no 2nd checkpoints; mirrors Standard Ensemble)"
    rows.append({"Configuration": name, **m, "Note": note})

metrics_table = pd.DataFrame(rows).set_index("Configuration")
metrics_table_display = metrics_table.copy()
for c in ["Balanced Accuracy", "Macro AUC", "ECE"]:
    metrics_table_display[c] = metrics_table_display[c].map(lambda v: f"{v:.4f}")
metrics_table.to_csv(WORK_DIR / "metrics_full_split.csv")
print("Full-split metrics (saved to /kaggle/working/metrics_full_split.csv):")
metrics_table_display
---
# ============================================================================
# Cell 7b — Visualise the per-configuration metrics.
# ============================================================================
plot_df = metrics_table[["Balanced Accuracy", "Macro AUC", "ECE"]].astype(float)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, metric, better in zip(axes, plot_df.columns, ["higher", "higher", "lower"]):
    vals = plot_df[metric].values
    colors = sns.color_palette("viridis", len(vals))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"{metric}  ({better} is better)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Argus Vision — metrics by configuration (full test split)", y=1.03)
fig.tight_layout()
fig.savefig(FIG_DIR / "metrics_by_configuration.png", dpi=150, bbox_inches="tight")
plt.show()
---
# ============================================================================
# Cell 8 — D_hard metrics table.
# ============================================================================
hard_csv = find_file("hard_subset")
eval_index_by_stem = {Path(p).stem: i for i, p in enumerate(image_paths)}

hard_indices: List[int] = []
if hard_csv is not None:
    hard_df = pd.read_csv(hard_csv)
    path_col = "image_path" if "image_path" in hard_df.columns else hard_df.columns[0]
    for raw in hard_df[path_col].astype(str):
        stem = Path(raw).stem
        if stem in eval_index_by_stem:
            hard_indices.append(eval_index_by_stem[stem])
    hard_indices = sorted(set(hard_indices))
    print(f"hard_subset.csv has {len(hard_df)} rows; "
          f"{len(hard_indices)} of them are in the current eval split.")
else:
    print("hard_subset.csv not found; falling back to images where the live debate trigger fired.")

# Fallback: use the images where our own trigger fired.
if len(hard_indices) == 0:
    hard_indices = [i for i, r in enumerate(debate_records) if r["fired"]]
    print(f"Using {len(hard_indices)} trigger-fired images as D_hard.")

if len(hard_indices) >= 2 and len(np.unique(y_true[hard_indices])) >= 2:
    idx = np.asarray(hard_indices)
    yh = y_true[idx]
    hard_rows = []
    for name in CONFIG_NAMES:
        m = metrics_for(proba[name][idx], yh)
        note = "SKIPPED" if (name == "Deep Ensemble" and not DEEP_ENSEMBLE_AVAILABLE) else ""
        hard_rows.append({"Configuration": name, **m, "Note": note})
    hard_metrics_table = pd.DataFrame(hard_rows).set_index("Configuration")
    hard_metrics_table.to_csv(WORK_DIR / "metrics_hard_subset.csv")
    disp = hard_metrics_table.copy()
    for c in ["Balanced Accuracy", "Macro AUC", "ECE"]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    print(f"\nD_hard metrics over {len(idx)} images "
          f"(saved to /kaggle/working/metrics_hard_subset.csv):")
    display(disp)
else:
    print("Not enough hard-subset images (need >=2 with >=2 classes) to compute a D_hard table.")
    hard_metrics_table = None
---
# ============================================================================
# Cell 9 — Ablation: no-debate vs no-attention-features vs full Argus.
# ============================================================================
fired_recs = [r for r in debate_records if r["fired"] and r.get("consensus_prob") is not None]
print(f"Ablation runs over {len(fired_recs)} triggering images with computed debate features.")

if len(fired_recs) >= 2:
    # Rebuild the feature matrix for the triggering images (we have everything cached
    # on each record except the argument embeddings, which we re-derive deterministically
    # from the stored transcript — identical to the live path).
    feats_full, feats_nodebate, feats_noattn, ab_labels = [], [], [], []
    stem_to_label = {Path(p).stem: y_true[i] for i, p in enumerate(image_paths)}
    for r in tqdm(fired_recs, total=len(fired_recs), desc="ablation features"):
        pa, pb = r["pa"], r["pb"]
        stats_a, stats_b = r["stats_a"], r["stats_b"]
        emb_a = encode_argument(strip_delta(r["transcript"]["arg_a_r2"]))
        emb_b = encode_argument(strip_delta(r["transcript"]["arg_b_r2"]))
        zero_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        zero_stats = {"mean": 0.0, "std": 0.0, "max": 0.0}
        feats_full.append(build_feature(pa, pb, stats_a, stats_b, emb_a, emb_b))
        feats_nodebate.append(build_feature(pa, pb, stats_a, stats_b, zero_emb, zero_emb))
        feats_noattn.append(build_feature(pa, pb, zero_stats, zero_stats, emb_a, emb_b))
        ab_labels.append(int(stem_to_label[r["image_id"]]))

    ab_labels = np.asarray(ab_labels, dtype=np.int64)

    def consensus_probs(feat_mat: np.ndarray) -> np.ndarray:
        consensus.eval()
        with torch.no_grad():
            ft = torch.from_numpy(np.asarray(feat_mat, dtype=np.float32)).to(DEVICE)
            return F.softmax(consensus(ft), dim=1).detach().cpu().numpy().astype(np.float64)

    variants = {
        "Full Argus (debate + attention)": np.asarray(feats_full, dtype=np.float32),
        "No-debate (zeroed arg embeddings)": np.asarray(feats_nodebate, dtype=np.float32),
        "No-attention-features (zeroed spatial)": np.asarray(feats_noattn, dtype=np.float32),
    }
    ab_rows = []
    for vname, fmat in variants.items():
        p = consensus_probs(fmat)
        m = metrics_for(p, ab_labels)
        ab_rows.append({"Ablation variant": vname, **m})
    ablation_table = pd.DataFrame(ab_rows).set_index("Ablation variant")
    ablation_table.to_csv(WORK_DIR / "ablation_table.csv")
    disp = ablation_table.copy()
    for c in ["Balanced Accuracy", "Macro AUC", "ECE"]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    print("Ablation (saved to /kaggle/working/ablation_table.csv):")
    display(disp)
else:
    print("Too few triggering images to run the ablation.")
    ablation_table = None
---
# ============================================================================
# Cell 10 — Select the six case-study images.
# ============================================================================
stem_to_eval_idx = {Path(p).stem: i for i, p in enumerate(image_paths)}


def top_pred(prob_row: np.ndarray):
    return int(prob_row.argmax()), float(prob_row.max())


case_pool = []
for r in debate_records:
    i = stem_to_eval_idx[r["image_id"]]
    gt = int(y_true[i])
    a_pred = int(r["pa"].argmax()); b_pred = int(r["pb"].argmax())
    ens_pred = int(proba["Standard Ensemble"][i].argmax())
    argus_pred = int(proba["Argus (full)"][i].argmax())
    case_pool.append({
        "rec": r, "eval_idx": i, "gt": gt,
        "a_pred": a_pred, "b_pred": b_pred,
        "ens_pred": ens_pred, "argus_pred": argus_pred,
        "agree": a_pred == b_pred, "fired": r["fired"],
    })

selected = []
used = set()


def pick(predicate, count, label):
    chosen = []
    for c in case_pool:
        if len(chosen) >= count:
            break
        if c["eval_idx"] in used:
            continue
        if predicate(c):
            c2 = dict(c); c2["case_kind"] = label
            chosen.append(c2); used.add(c["eval_idx"])
    return chosen


# 1-2 agree-correct
selected += pick(lambda c: c["agree"] and c["a_pred"] == c["gt"], 2, "Agree-correct")
# 3-4 debate-helps: ensemble wrong, Argus right, and trigger fired
selected += pick(lambda c: c["fired"] and c["ens_pred"] != c["gt"] and c["argus_pred"] == c["gt"],
                 2, "Debate-helps")
# 5 debate-fails: trigger fired, Argus still wrong
selected += pick(lambda c: c["fired"] and c["argus_pred"] != c["gt"], 1, "Debate-fails")
# 6 melanoma (GT == MEL)
selected += pick(lambda c: c["gt"] == CLASS_NAMES.index("MEL"), 1, "Melanoma")

# Backfill to ensure we always show six cases even on degenerate splits.
fill_priority = sorted(case_pool, key=lambda c: (not c["fired"], not c["agree"]))
fi = 0
while len(selected) < 6 and fi < len(fill_priority):
    c = fill_priority[fi]; fi += 1
    if c["eval_idx"] in used:
        continue
    c2 = dict(c); c2["case_kind"] = "Additional"
    selected.append(c2); used.add(c["eval_idx"])

print(f"Selected {len(selected)} case studies:")
for c in selected:
    print(f"  [{c['case_kind']:13s}] {c['rec']['image_id']:>16s}  GT={CLASS_NAMES[c['gt']]:4s} "
          f"A={CLASS_NAMES[c['a_pred']]:4s} B={CLASS_NAMES[c['b_pred']]:4s} "
          f"Ens={CLASS_NAMES[c['ens_pred']]:4s} Argus={CLASS_NAMES[c['argus_pred']]:4s} "
          f"fired={c['fired']}")
---
# ============================================================================
# Cell 10b — Render each case study (image + 2 heatmaps + M_delta + texts + conf).
# ============================================================================
import textwrap


def denorm_for_show(image_path: str) -> np.ndarray:
    with Image.open(image_path) as h:
        img = h.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


def overlay(base_rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    hm = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32) / 255.0
    return np.clip(0.55 * base_rgb + 0.45 * cmap, 0, 1)


for c in selected:
    r = c["rec"]; i = c["eval_idx"]
    base = denorm_for_show(r["image_path"])
    tensor = load_tensor(r["image_path"])
    hm_a = gradcam_plusplus(agent_a, tensor, int(r["pa"].argmax()))
    hm_b = attention_rollout(agent_b, tensor)
    m_delta = disagreement_map(hm_a, hm_b)

    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(base); ax0.axis("off")
    ax0.set_title(f"Original\nGT = {CLASS_NAMES[c['gt']]} ({FULL_NAMES[CLASS_NAMES[c['gt']]]})", fontsize=9)
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(overlay(base, hm_a)); ax1.axis("off")
    ax1.set_title(f"Agent A Grad-CAM++\npred {CLASS_NAMES[c['a_pred']]} ({r['pa'].max():.2f})", fontsize=9)
    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(overlay(base, hm_b)); ax2.axis("off")
    ax2.set_title(f"Agent B attention\npred {CLASS_NAMES[c['b_pred']]} ({r['pb'].max():.2f})", fontsize=9)
    ax3 = fig.add_subplot(gs[0, 3]); im = ax3.imshow(m_delta, cmap="magma"); ax3.axis("off")
    ax3.set_title(r"Disagreement map $M_\Delta$", fontsize=9)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Confidence before (standard ensemble) vs after (Argus consensus).
    ens_p = proba["Standard Ensemble"][i]; argus_p = proba["Argus (full)"][i]
    ens_top, ens_conf = top_pred(ens_p); arg_top, arg_conf = top_pred(argus_p)

    axc = fig.add_subplot(gs[1, 0])
    axc.bar([0, 1], [ens_conf, arg_conf], color=["#6B7280", "#2563EB"])
    axc.set_xticks([0, 1]); axc.set_xticklabels(["before\n(ensemble)", "after\n(Argus)"], fontsize=8)
    axc.set_ylim(0, 1); axc.set_ylabel("top-class conf", fontsize=8)
    axc.set_title(f"before={CLASS_NAMES[ens_top]} -> after={CLASS_NAMES[arg_top]}", fontsize=8)
    for xi, v in zip([0, 1], [ens_conf, arg_conf]):
        axc.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    tr = r.get("transcript")
    arg_a_txt = strip_delta(tr["arg_a_r2"]) if tr else "(no debate — trigger did not fire)"
    arg_b_txt = strip_delta(tr["arg_b_r2"]) if tr else "(no debate — trigger did not fire)"
    axta = fig.add_subplot(gs[1, 1:3]); axta.axis("off")
    axta.text(0.0, 1.0, "Agent A argument:\n" + textwrap.fill(arg_a_txt, 78),
              va="top", ha="left", fontsize=7.2, family="monospace")
    axtb = fig.add_subplot(gs[1, 3]); axtb.axis("off")
    axtb.text(0.0, 1.0, "Agent B argument:\n" + textwrap.fill(arg_b_txt, 36),
              va="top", ha="left", fontsize=7.2, family="monospace")

    correct = "OK" if c["argus_pred"] == c["gt"] else "X"
    fig.suptitle(f"[{c['case_kind']}]  {r['image_id']}  |  JS={r['js']:.3f}  "
                 f"fired={r['fired']}  |  Argus={CLASS_NAMES[c['argus_pred']]} {correct}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"case_{c['case_kind'].lower().replace(' ', '_')}_{r['image_id']}.png",
                dpi=140, bbox_inches="tight")
    plt.show()

print("Case studies rendered and saved under /kaggle/working/figures/.")
---
# ============================================================================
# Cell 11 — Confusion matrix for the Argus (full) configuration.
# ============================================================================
from sklearn.metrics import confusion_matrix

argus_preds = proba["Argus (full)"].argmax(axis=1)
cm = confusion_matrix(y_true, argus_preds, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            cbar_kws={"label": "recall (row-normalised)"})
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Argus (full) — confusion matrix on the test split")
fig.tight_layout()
fig.savefig(FIG_DIR / "argus_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()