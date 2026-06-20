# Kaggle pre-installs torch, torchvision, numpy, pandas, scikit-learn, matplotlib,
# seaborn, opencv-python, Pillow, tqdm. We install the EXTRAS this notebook needs.
# INTERNET MUST BE ON (Settings -> Internet -> ON) for pip + pretrained weights + Groq.
import sys, subprocess

print("Installing extras (timm, grad-cam, sentence-transformers, netcal, groq)...")
print("NOTE: Internet must be ON for this to work (Settings -> Internet -> ON).")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-U", "timm",                 # -U so the ViT model id resolves
    "grad-cam",                   # Grad-CAM++ for Agent A
    "sentence-transformers",      # 384-d argument embeddings
    "netcal",                     # ECE metric
    "groq",                       # LLM argument generation
], check=False)

import os, json, math, time, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# --- Shared contract constants (identical to backend + other notebooks) ----
ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
NUM_CLASSES = 8

# Consensus feature contract (must match backend/ml/consensus/classifier.py).
FEATURE_DIM = 788
PROB_DIM = 8
SPATIAL_DIM = 4
EMBED_DIM = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Reproducibility.
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def discover_isic(root="/kaggle/input"):
    """Locate the ISIC-2019 ground-truth CSV and the training-image directory.

    The CSV is the .csv whose header contains ALL 8 ISIC class names. The image
    dir is the directory containing the most ISIC_*.jpg/.jpeg files. Robust to
    nested mirror folders (e.g. doubled ISIC_2019_Training_Input/).
    """
    csv_path = None
    class_set = set(ISIC_CLASSES)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                fp = os.path.join(dirpath, fn)
                try:
                    cols = set(pd.read_csv(fp, nrows=0).columns)
                except Exception:
                    continue
                if class_set.issubset(cols):
                    csv_path = fp
                    break
        if csv_path is not None:
            break

    # Image dir = directory with the most ISIC_*.jpg/.jpeg files.
    best_dir = None
    best_count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        count = 0
        for fn in filenames:
            low = fn.lower()
            if low.startswith("isic_") and (low.endswith(".jpg") or low.endswith(".jpeg")):
                count += 1
        if count > best_count:
            best_count = count
            best_dir = dirpath

    print("Discovered ground-truth CSV :", csv_path)
    print("Discovered image directory  :", best_dir, "(%d ISIC images)" % best_count)
    assert csv_path is not None and os.path.exists(csv_path), "Could not find ISIC ground-truth CSV under " + root
    assert best_dir is not None and best_count > 0, "Could not find an ISIC image directory under " + root
    return csv_path, best_dir


def find_file(filename_substring, search_roots=("/kaggle/input", "/kaggle/working")):
    """Return the first path whose basename contains the given substring."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if filename_substring in fn:
                    found = os.path.join(dirpath, fn)
                    print("find_file('%s') -> %s" % (filename_substring, found))
                    return found
    print("find_file('%s'): NOT FOUND under %s" % (filename_substring, list(search_roots)))
    return None

---
GROQ_MODEL = "llama-3.3-70b-versatile"
groq_api_key = None
groq_client = None

try:
    from kaggle_secrets import UserSecretsClient
    groq_api_key = UserSecretsClient().get_secret("GROQ_API_KEY")
except Exception as exc:
    print("Could not read GROQ_API_KEY Kaggle Secret (%s)." % exc)
    groq_api_key = None

if groq_api_key:
    try:
        from groq import Groq
        groq_client = Groq(api_key=groq_api_key)
        print("Groq client initialised. Arguments will be LLM-generated (model=%s)." % GROQ_MODEL)
    except Exception as exc:
        print("Groq SDK init failed (%s); falling back to deterministic argument text." % exc)
        groq_client = None
else:
    print("No GROQ_API_KEY available -> using DETERMINISTIC FALLBACK argument text (no LLM calls).")

---
import timm
from sentence_transformers import SentenceTransformer

AGENT_A_MODEL = "efficientnet_b4"
AGENT_B_MODEL = "vit_base_patch16_224.augreg_in21k_ft_in1k"


def load_agent(model_name, ckpt_substring):
    """Create a timm backbone with an 8-class head and load its checkpoint."""
    ckpt_path = find_file(ckpt_substring)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        res = model.load_state_dict(state, strict=False)
        print("Loaded %s from %s (missing=%d, unexpected=%d)" % (
            model_name, ckpt_path, len(res.missing_keys), len(res.unexpected_keys)))
    else:
        print("WARNING: no checkpoint for %s; using ImageNet-pretrained backbone "
              "with a random 8-class head (predictions NOT clinically meaningful)." % model_name)
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model.eval().to(DEVICE)
    return model


agent_a = load_agent(AGENT_A_MODEL, "agent_a_best")
agent_b = load_agent(AGENT_B_MODEL, "agent_b_best")

print("Loading sentence encoder all-MiniLM-L6-v2 ...")
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
_test_dim = len(sentence_encoder.encode("dimension probe", normalize_embeddings=True))
print("Sentence encoder embedding dim:", _test_dim)
assert _test_dim == EMBED_DIM, "Expected 384-d embeddings, got %d" % _test_dim

---
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32  # lower to 16 if you hit CUDA OOM
NUM_WORKERS = 2

CSV_PATH, IMAGE_DIR = discover_isic()

df = pd.read_csv(CSV_PATH)
# image-id column is named "image" in the ISIC-2019 ground truth.
id_col = "image" if "image" in df.columns else df.columns[0]
labels_all = df[ISIC_CLASSES].values.argmax(axis=1).astype(np.int64)
df = df.assign(_label=labels_all)
print("Total labelled rows:", len(df))
print("Class counts:", {ISIC_CLASSES[i]: int((labels_all == i).sum()) for i in range(NUM_CLASSES)})

# The SAME stratified split used across the project.
train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["_label"].values, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print("Train rows:", len(train_df), "Val rows:", len(val_df))

# Optionally restrict the TRAIN pool to Notebook 03's hard subset, if attached.
hard_csv = find_file("hard_subset.csv")
if hard_csv is not None:
    try:
        hard_df = pd.read_csv(hard_csv)
        hcol = "image" if "image" in hard_df.columns else hard_df.columns[0]
        hard_ids = set(hard_df[hcol].astype(str).tolist())
        before = len(train_df)
        train_df = train_df[train_df[id_col].astype(str).isin(hard_ids)].reset_index(drop=True)
        print("Restricted TRAIN pool to hard_subset.csv: %d -> %d rows" % (before, len(train_df)))
    except Exception as exc:
        print("Could not apply hard_subset.csv (%s); using full train split." % exc)


# val-style transform (deterministic) for trigger scan + feature building.
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# train-style augmentation transform (documented for completeness; the consensus
# head trains on extracted features, so feature extraction uses eval_tf).
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.1),
])


class ISICDataset(Dataset):
    """Reads the one-hot ISIC-2019 ground-truth CSV; label = argmax over 8 classes."""

    def __init__(self, frame, image_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image_id = str(row[id_col])
        path = os.path.join(self.image_dir, image_id + ".jpg")
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        label = int(row["_label"])
        return tensor, label, path


# --- Trigger (inlined from backend/ml/debate/trigger.py) -------------------
from scipy.spatial.distance import jensenshannon

TAU_JS = 0.25        # squared Jensen-Shannon divergence threshold
TAU_ENTROPY = 0.8   # Shannon entropy (bits) threshold
_EPS = 1e-12


def shannon_entropy(probs):
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    return -float(np.sum(p * np.log2(p + _EPS)))


def trigger_fired(pa, pb, tau_js=TAU_JS, tau_entropy=TAU_ENTROPY):
    pa = np.asarray(pa, dtype=np.float64).ravel()
    pb = np.asarray(pb, dtype=np.float64).ravel()
    jsd = jensenshannon(pa, pb, base=2)
    jsd = float(jsd) ** 2
    if not np.isfinite(jsd):
        jsd = 0.0
    ea = shannon_entropy(pa)
    eb = shannon_entropy(pb)
    fired = (jsd > tau_js) or (max(ea, eb) > tau_entropy)
    return fired, jsd, ea, eb


@torch.no_grad()
def agent_probs(model, tensor):
    """Return the 8-class softmax probability vector (numpy) for a (1,3,224,224) tensor."""
    logits = model(tensor.to(DEVICE))
    return F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float64)


# Scan the TRAIN pool and keep images where the trigger fires.
# Cap the number of fired images so a Kaggle session finishes comfortably; raise
# MAX_FIRED if you have time/credits to spare.
MAX_SCAN = 8000     # how many train rows to scan for triggers
MAX_FIRED = 1500     # cap on the number of debate samples to build features for

scan_df = train_df.iloc[:min(MAX_SCAN, len(train_df))].reset_index(drop=True)
scan_ds = ISICDataset(scan_df, IMAGE_DIR, eval_tf)

fired_records = []  # each: dict(image_id, path, label, pa, pb, jsd)
print("Scanning %d train images for trigger firing..." % len(scan_ds))
for i in tqdm(range(len(scan_ds))):
    tensor, label, path = scan_ds[i]
    tensor = tensor.unsqueeze(0)
    pa = agent_probs(agent_a, tensor)
    pb = agent_probs(agent_b, tensor)
    fired, jsd, ea, eb = trigger_fired(pa, pb)
    if fired:
        fired_records.append({
            "image_id": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "label": int(label),
            "pa": pa,
            "pb": pb,
            "jsd": float(jsd),
        })
    if len(fired_records) >= MAX_FIRED:
        break

print("Trigger fired on %d images (building consensus features for these)." % len(fired_records))
assert len(fired_records) > 0, "No images fired the trigger; lower TAU_JS / TAU_ENTROPY or raise MAX_SCAN."

---
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Grad-CAM++ for Agent A (target layer resolution mirrors backend) ------
def _resolve_target_layer(model):
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        last_block = blocks[-1]
        bn3 = getattr(last_block, "bn3", None)
        if isinstance(bn3, nn.Module):
            return bn3
        return last_block
    conv_head = getattr(model, "conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head
    fallback = None
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
            fallback = module
    if fallback is None:
        raise RuntimeError("No Grad-CAM++ target layer found.")
    return fallback


def compute_gradcam_plusplus(model, tensor, target_class):
    target_layer = _resolve_target_layer(model)
    model.zero_grad()
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(int(target_class))])
    model.zero_grad()
    heat = np.asarray(grayscale[0], dtype=np.float32)
    if heat.shape != (IMAGE_SIZE, IMAGE_SIZE):
        heat = cv2.resize(heat, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.clip(heat, 0.0, 1.0), dtype=np.float32)


# --- Attention rollout for Agent B (inlined from backend/ml/attention/rollout.py)
GRID_SIZE = 14
NUM_PATCH_TOKENS = GRID_SIZE * GRID_SIZE


def _centered_gaussian(size=IMAGE_SIZE, sigma_frac=0.25):
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    sigma = max(sigma_frac, 1e-6)
    g = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    span = float(g.max() - g.min())
    if span < 1e-12:
        return np.zeros((size, size), dtype=np.float32)
    return ((g - g.min()) / span).astype(np.float32)


def compute_attention_rollout(model, tensor):
    handles = []
    captured = {}
    original_fused = {}
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return _centered_gaussian()

    def _make_hook(layer_idx):
        def _hook(module, inputs, output):
            try:
                x = inputs[0]
                batch, num_tokens, dim = x.shape
                num_heads = int(module.num_heads)
                head_dim = dim // num_heads
                scale = getattr(module, "scale", None)
                if scale is None:
                    scale = head_dim ** -0.5
                qkv = module.qkv(x).reshape(batch, num_tokens, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                attn = (q @ k.transpose(-2, -1)) * float(scale)
                attn = attn.softmax(dim=-1)
                captured[layer_idx] = attn.mean(dim=1).detach().to(torch.float32).cpu()
            except Exception:
                pass
        return _hook

    try:
        for idx, block in enumerate(blocks):
            attn_module = getattr(block, "attn", None)
            if attn_module is None or not hasattr(attn_module, "qkv"):
                continue
            original_fused[idx] = bool(getattr(attn_module, "fused_attn", False))
            if hasattr(attn_module, "fused_attn"):
                attn_module.fused_attn = False
            handles.append(attn_module.register_forward_hook(_make_hook(idx)))
        if not handles:
            return _centered_gaussian()
        with torch.no_grad():
            model(tensor)
        if not captured:
            return _centered_gaussian()
        layer_indices = sorted(captured.keys())
        num_tokens = captured[layer_indices[0]].shape[-1]
        identity = torch.eye(num_tokens, dtype=torch.float32)
        rollout = torch.eye(num_tokens, dtype=torch.float32)
        for idx in layer_indices:
            attn = captured[idx][0]
            aug = 0.5 * attn + 0.5 * identity
            aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rollout = aug @ rollout
        cls_attention = rollout[0, 1:]
        if cls_attention.shape[0] < NUM_PATCH_TOKENS:
            return _centered_gaussian()
        grid = cls_attention[:NUM_PATCH_TOKENS].reshape(GRID_SIZE, GRID_SIZE).numpy().astype(np.float32)
        heat = cv2.resize(grid, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        span = float(heat.max() - heat.min())
        if span < 1e-12:
            return _centered_gaussian()
        heat = (heat - heat.min()) / span
        return np.ascontiguousarray(np.clip(heat, 0.0, 1.0), dtype=np.float32)
    except Exception:
        return _centered_gaussian()
    finally:
        for handle in handles:
            handle.remove()
        for idx, was_fused in original_fused.items():
            attn_module = getattr(blocks[idx], "attn", None)
            if attn_module is not None and hasattr(attn_module, "fused_attn"):
                attn_module.fused_attn = was_fused


# --- Disagreement / contested-region stats (inlined from disagreement.py) --
def _min_max_normalize(a):
    a = np.asarray(a, dtype=np.float32)
    mn, mx = float(a.min()), float(a.max())
    span = mx - mn
    if span < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / span).astype(np.float32)


def _region_stats(norm_map, mask):
    if not bool(mask.any()):
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    sel = norm_map[mask]
    return {"mean": float(sel.mean()), "std": float(sel.std()), "max": float(sel.max())}


def compute_disagreement(heatmap_a, heatmap_b):
    norm_a = _min_max_normalize(heatmap_a)
    norm_b = _min_max_normalize(heatmap_b)
    m_delta = np.abs(norm_a - norm_b).astype(np.float32)
    combined = norm_a + norm_b
    threshold = float(np.percentile(combined, 80.0))
    mask = combined >= threshold
    if not bool(mask.any()):
        mask = np.ones_like(combined, dtype=bool)
    return m_delta, _region_stats(norm_a, mask), _region_stats(norm_b, mask)

---
# --- Argument generation (inlined from backend/ml/debate/argument_gen.py) --
import re

CLASS_FULL_NAMES = {
    "MEL": "Melanoma", "NV": "Melanocytic Nevus", "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis", "BKL": "Benign Keratosis", "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion", "SCC": "Squamous Cell Carcinoma",
}
ISIC_CLASS_DESCRIPTIONS = {
    "MEL": "Melanoma typically shows an atypical broadened pigment network with irregular streaks, asymmetry of structure and colour, a blue-white veil and chaotic vessels.",
    "NV": "A melanocytic nevus shows a symmetric, regularly spaced reticular or globular pattern with uniform colour and a smooth transition to surrounding skin.",
    "BCC": "Basal cell carcinoma shows arborising (tree-like) vessels and blue-grey ovoid nests on a pigment-network-free background, with leaf-like and spoke-wheel structures.",
    "AK": "Actinic keratosis shows a strawberry pattern: a red pseudo-network of dilated vessels around keratin-plugged follicular openings on a scaly erythematous background.",
    "BKL": "Benign keratosis shows a cerebriform brain-like surface with milia-like cysts, comedo-like openings, sharply demarcated borders and a stuck-on appearance.",
    "DF": "Dermatofibroma shows a central white scar-like patch surrounded by a delicate peripheral pigment network, firm with a homogeneous tan-brown ring.",
    "VASC": "Vascular lesions show sharply demarcated red, purple or maroon lacunae separated by pale septa, with no melanocytic pigment network.",
    "SCC": "Squamous cell carcinoma shows central keratin masses, white circles around follicular openings, surface scale, and looped or glomerular vessels at the periphery.",
}
_SYSTEM_PERSONA = (
    "You are a board-certified dermatology AI in a structured adversarial debate about one "
    "dermoscopic skin-lesion image. You reason with the ABCDE rule and established dermoscopic "
    "criteria (pigment network, vessels, dots/globules, structureless zones, blue-white veil, "
    "milia-like cysts, lacunae, keratin). Argue concisely, cite specific visual evidence in the "
    "contested region, never invent findings contradicting the statistics, and never break character."
)
_DELTA_PATTERN = re.compile(r"CONFIDENCE_DELTA\s*:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)


def _full_name(c):
    return CLASS_FULL_NAMES.get(c, c)


def _describe(c):
    return ISIC_CLASS_DESCRIPTIONS.get(c, "No dermoscopic description available.")


def _region_summary(stats):
    return ("In the contested region your attention map has mean activation %.3f, standard "
            "deviation %.3f, and peak %.3f." % (stats.get("mean", 0.0), stats.get("std", 0.0), stats.get("max", 0.0)))


def build_prompt(agent_id, pred_class, confidence, region_stats, opp, opp_conf, rnd):
    user = (
        "You are Agent %s. You classify this dermoscopic lesion as %s (%s) with %.1f%% confidence.\n\n"
        "Clinical profile of %s: %s\n\n%s\n\n"
        "The opposing agent classifies the same lesion as %s (%s) with %.1f%% confidence.\n\n"
        "This is round %d of the debate. Argue in ONE focused paragraph why your diagnosis of %s "
        "is correct, grounding every claim in the dermoscopic evidence in the contested region and "
        "explaining why it favours %s over %s. No preamble, headings, or lists."
    ) % (agent_id, _full_name(pred_class), pred_class, confidence * 100, pred_class,
         _describe(pred_class), _region_summary(region_stats), _full_name(opp), opp,
         opp_conf * 100, rnd, pred_class, pred_class, opp)
    return [{"role": "system", "content": _SYSTEM_PERSONA}, {"role": "user", "content": user}]


def build_counter_prompt(agent_id, pred_class, original_arg, opponent_arg, confidence,
                         opp, opp_conf, region_stats):
    user = (
        "You are Agent %s, defending a diagnosis of %s (%s) at %.1f%% confidence.\n\n"
        "Clinical profile of %s: %s\n\n%s\n\n"
        "Your opening argument was:\n\"\"\"\n%s\n\"\"\"\n\n"
        "The opposing agent (predicting %s / %s at %.1f%%) argued:\n\"\"\"\n%s\n\"\"\"\n\n"
        "This is round 2. In ONE focused paragraph, directly rebut the opponent's strongest point "
        "using dermoscopic evidence from the contested region, then decide whether to HOLD or REVISE "
        "your confidence in %s. After the paragraph, on a final separate line, output exactly "
        "'CONFIDENCE_DELTA: <number>' where <number> is a float in [-0.3, 0.3]."
    ) % (agent_id, _full_name(pred_class), pred_class, confidence * 100, pred_class,
         _describe(pred_class), _region_summary(region_stats), original_arg.strip(),
         _full_name(opp), opp, opp_conf * 100, opponent_arg.strip(), pred_class)
    return [{"role": "system", "content": _SYSTEM_PERSONA}, {"role": "user", "content": user}]


def _fallback_argument(pred_class, confidence, region_stats, opp=None):
    contrast = ""
    if opp and opp != pred_class:
        contrast = " These features are inconsistent with %s (%s)." % (_full_name(opp), opp)
    return ("Based on the dermoscopic evidence, this lesion is most consistent with %s (%s) at "
            "%.1f%% confidence. %s %s The concentration of attention over these structures supports "
            "the %s diagnosis.%s") % (_full_name(pred_class), pred_class, confidence * 100,
            _describe(pred_class), _region_summary(region_stats), pred_class, contrast)


def generate_argument(messages, fallback_text):
    import time
    if groq_client is None:
        return fallback_text
    
    max_retries = 5
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=300)
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                return fallback_text
            # Add a tiny delay to naturally space out requests
            time.sleep(0.5)
            return content.strip()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate limit" in exc_str:
                sleep_time = backoff * (2 ** attempt)
                print("Rate limit (429) hit. Retrying in %.1fs (Attempt %d/%d)..." % (sleep_time, attempt + 1, max_retries))
                time.sleep(sleep_time)
            else:
                print("Groq call failed (%s); using fallback." % exc)
                return fallback_text
                
    print("Groq call failed after maximum retries; using fallback.")
    return fallback_text


def strip_delta(text):
    return _DELTA_PATTERN.sub("", text).strip() or text.strip()

---
# --- Argument generation (inlined from backend/ml/debate/argument_gen.py) --
import re
import json
import os
import numpy as np

CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}

ISIC_CLASS_DESCRIPTIONS = {
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

_SYSTEM_PERSONA = (
    "You are a board-certified dermatology AI in a structured adversarial debate about one "
    "dermoscopic skin-lesion image. You reason with the ABCDE rule and established dermoscopic "
    "criteria (pigment network, vessels, dots/globules, structureless zones, blue-white veil, "
    "milia-like cysts, lacunae, keratin). Argue concisely, cite specific visual evidence in the "
    "contested region, never invent findings contradicting the statistics, and never break character."
)

_DELTA_PATTERN = re.compile(r"CONFIDENCE_DELTA\s*:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)

def _full_name(c):
    return CLASS_FULL_NAMES.get(c, c)

def _describe(c):
    return ISIC_CLASS_DESCRIPTIONS.get(c, "No dermoscopic description available.")

def _region_summary(stats):
    return (f"In the contested region your attention map has mean {stats.get('mean', 0.0):.3f}, "
            f"std {stats.get('std', 0.0):.3f}, peak {stats.get('max', 0.0):.3f}.")

def _fallback_argument(pred_class, confidence, region_stats, opp=None):
    stats = region_stats or {}
    contrast = ""
    if opp and opp != pred_class:
        contrast = (
            f"These features are inconsistent with {_full_name(opp)} ({opp})."
        )
    return (
        f"This lesion is most consistent with {_full_name(pred_class)} ({pred_class}) at "
        f"{confidence * 100:.1f}% confidence. {_describe(pred_class)} "
        f"{_region_summary(stats)} {contrast}"
    )

# --- Instrumentation counters for Groq vs Fallback calls ---
total_gen_calls = 0
fallback_gen_calls = 0

def generate_argument(messages, fallback_text):
    global total_gen_calls, fallback_gen_calls, groq_client
    total_gen_calls += 1
    import time
    if groq_client is None:
        fallback_gen_calls += 1
        return fallback_text
    
    max_retries = 5
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=300)
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                fallback_gen_calls += 1
                return fallback_text
            # Add a tiny delay to naturally space out requests
            time.sleep(0.5)
            return content.strip()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate limit" in exc_str:
                print("WARNING: Groq Rate limit (429) hit! Disabling Groq API calls for the remainder of this run to fall back instantly.")
                groq_client = None
                fallback_gen_calls += 1
                return fallback_text
            else:
                print("WARNING: Groq call failed (%s); using fallback." % exc)
                fallback_gen_calls += 1
                return fallback_text
                
    print("WARNING: Groq call failed after maximum retries; using fallback.")
    fallback_gen_calls += 1
    return fallback_text

def strip_delta(text):
    return _DELTA_PATTERN.sub("", text).strip() or text.strip()

# --- Groq response cache (survives reruns; version v2) --------------
GROQ_CACHE_PATH = "/kaggle/working/groq_cache_v2.json"
if os.path.exists(GROQ_CACHE_PATH):
    with open(GROQ_CACHE_PATH, "r") as f:
        groq_cache = json.load(f)
    print("Loaded %d cached debate responses from %s" % (len(groq_cache), GROQ_CACHE_PATH))
    # Analyze the cache to count fallback vs real Groq call fraction
    total_cached = 0
    fallback_cached = 0
    for img_id, entry in groq_cache.items():
        text_a = entry.get("arg_a_r2", entry.get("text_a", ""))
        text_b = entry.get("arg_b_r2", entry.get("text_b", ""))
        for txt in [text_a, text_b]:
            if txt:
                total_cached += 1
                if txt.startswith("This lesion is most consistent with") or txt.startswith("Based on the dermoscopic evidence"):
                    fallback_cached += 1
    if total_cached > 0:
        pct_fb = (fallback_cached / total_cached) * 100.0
        print("Cache Wording Analysis: %d arguments in cache. Fallbacks: %d (%.1f%%), Groq: %d (%.1f%%)" % (
            total_cached, fallback_cached, pct_fb, total_cached - fallback_cached, 100.0 - pct_fb))
else:
    groq_cache = {}
    print("No existing groq_cache_v2.json found; starting fresh.")

def save_cache():
    with open(GROQ_CACHE_PATH, "w") as f:
        json.dump(groq_cache, f)

def debate_arguments(image_id, pa, pb, stats_a, stats_b):
    """Return (arg_a2, arg_b2), caching responses in the 4-key format."""
    if image_id in groq_cache:
        c = groq_cache[image_id]
        if "arg_a_r2" in c:
            return strip_delta(c["arg_a_r2"]), strip_delta(c["arg_b_r2"])
        elif "text_a" in c:
            # Handle older old concatenated format if present
            return c["text_a"], c["text_b"]

    pred_a = int(np.argmax(pa)); conf_a = float(pa[pred_a]); cls_a = ISIC_CLASSES[pred_a]
    pred_b = int(np.argmax(pb)); conf_b = float(pb[pred_b]); cls_b = ISIC_CLASSES[pred_b]

    # Round 1.
    fb_a1 = _fallback_argument(cls_a, conf_a, stats_a, cls_b)
    fb_b1 = _fallback_argument(cls_b, conf_b, stats_b, cls_a)
    arg_a1 = generate_argument(build_prompt("A", cls_a, conf_a, stats_a, cls_b, conf_b, 1), fb_a1)
    arg_b1 = generate_argument(build_prompt("B", cls_b, conf_b, stats_b, cls_a, conf_a, 1), fb_b1)

    # Round 2 (rebuttals).
    fb_a2 = _fallback_argument(cls_a, conf_a, stats_a, cls_b)
    fb_b2 = _fallback_argument(cls_b, conf_b, stats_b, cls_a)
    raw_a2 = generate_argument(
        build_counter_prompt("A", cls_a, arg_a1, arg_b1, conf_a, cls_b, conf_b, stats_a), fb_a2)
    raw_b2 = generate_argument(
        build_counter_prompt("B", cls_b, arg_b1, arg_a1, conf_b, cls_a, conf_a, stats_b), fb_b2)

    # Store in the standard 4-key format
    groq_cache[image_id] = {
        "arg_a_r1": arg_a1,
        "arg_b_r1": arg_b1,
        "arg_a_r2": raw_a2,
        "arg_b_r2": raw_b2,
    }
    save_cache()
    
    arg_a2 = strip_delta(raw_a2)
    arg_b2 = strip_delta(raw_b2)
    return arg_a2, arg_b2

def encode_arg(text):
    vec = sentence_encoder.encode(text, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.shape[0] != EMBED_DIM:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return vec

---
# --- Build the 788-d feature matrix and labels over all fired images -------
import os

REUSE_FEATURE_CACHE = False
cache_X_path = "/kaggle/working/consensus_features_X.npy"
cache_y_path = "/kaggle/working/consensus_features_y.npy"

# Reset instrumentation counters before feature building loop
total_gen_calls = 0
fallback_gen_calls = 0

if REUSE_FEATURE_CACHE and os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
    print("Loading cached consensus features from:", cache_X_path, "and", cache_y_path)
    X = np.load(cache_X_path)
    y = np.load(cache_y_path)
    print("Loaded X shape:", X.shape, "y shape:", y.shape)
else:
    X_list = []
    y_list = []

    # Reuse a deterministic eval tensor per image for Grad-CAM++ / rollout.
    feat_eval_ds = ISICDataset(
        pd.DataFrame({id_col: [r["image_id"] for r in fired_records],
                      "_label": [r["label"] for r in fired_records]}),
        IMAGE_DIR, eval_tf)

    print("Building consensus features for %d fired images..." % len(fired_records))
    for i, rec in enumerate(tqdm(fired_records)):
        tensor, label, _path = feat_eval_ds[i]
        tensor = tensor.unsqueeze(0).to(DEVICE)
        pa = rec["pa"]; pb = rec["pb"]

        # Grad-CAM++ (Agent A top class) and attention rollout (Agent B).
        cam_input = tensor.clone().requires_grad_(True)
        heat_a = compute_gradcam_plusplus(agent_a, cam_input, int(np.argmax(pa)))
        heat_b = compute_attention_rollout(agent_b, tensor)

        _m_delta, stats_a, stats_b = compute_disagreement(heat_a, heat_b)
        spatial = np.array([stats_a["mean"], stats_b["mean"], stats_a["std"], stats_b["std"]],
                           dtype=np.float32)

        text_a, text_b = debate_arguments(rec["image_id"], pa, pb, stats_a, stats_b)
        ea = encode_arg(text_a)
        eb = encode_arg(text_b)

        feat = np.concatenate([
            pa.astype(np.float32), pb.astype(np.float32), spatial, ea, eb
        ]).astype(np.float32)
        assert feat.shape[0] == FEATURE_DIM, "Feature dim %d != %d" % (feat.shape[0], FEATURE_DIM)
        X_list.append(feat)
        y_list.append(int(label))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    save_cache()
    print("Feature matrix X:", X.shape, " labels y:", y.shape)
    print("Cached debate responses:", len(groq_cache))
    np.save(cache_X_path, X)
    np.save(cache_y_path, y)
    print("Saved features to", cache_X_path, "and", cache_y_path)
    
    # Print real-time generation instrumentation report
    if total_gen_calls > 0:
        pct_fb_run = (fallback_gen_calls / total_gen_calls) * 100.0
        print("Feature Building Generation Wording Report:")
        print("Total arguments processed during this run: %d" % total_gen_calls)
        print("Generated fallbacks: %d (%.1f%%)" % (fallback_gen_calls, pct_fb_run))
        print("Real Groq API calls: %d (%.1f%%)" % (total_gen_calls - fallback_gen_calls, 100.0 - pct_fb_run))
    else:
        print("No dynamic arguments generated during this run (all loaded from cache).")

---
from torch.utils.data import TensorDataset

class ConsensusClassifier(nn.Module):
    """Calibrated fusion MLP. Architecture identical to backend/ml/consensus/classifier.py."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def logits(self, x):
        return self.mlp(x)

    def forward(self, x):
        t = torch.clamp(self.temperature, min=1e-2)
        return self.logits(x) / t


# --- Feature quality validation before training ----------------------------
if not np.isfinite(X).all():
    raise ValueError("Feature matrix X contains NaN or infinite values!")
if X.shape[1] != 788:
    raise ValueError("Feature matrix X second dimension is %d, expected 788!" % X.shape[1])
if len(X) == 0:
    raise ValueError("Feature matrix X is empty! No features to train on.")


# Stratified train/val split of the feature set.
feat_train_idx, feat_val_idx = train_test_split(
    np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)

Xtr = torch.tensor(X[feat_train_idx]); ytr = torch.tensor(y[feat_train_idx])
Xva = torch.tensor(X[feat_val_idx]);   yva = torch.tensor(y[feat_val_idx])
print("Consensus train:", Xtr.shape, " val:", Xva.shape)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, drop_last=False)
val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128, shuffle=False)

model = ConsensusClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

EPOCHS = 50
history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val_acc = -1.0
best_epoch = -1
best_state = None

# Early stopping configuration
patience = 8
min_delta = 0.001
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        # BatchNorm needs >1 sample; skip degenerate last batch of size 1.
        if xb.shape[0] < 2:
            continue
        out = model.logits(xb)  # train on raw logits (temperature == 1 here)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.shape[0]
    train_loss = running / max(1, len(Xtr))

    model.eval()
    vloss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model.logits(xb)
            vloss += criterion(out, yb).item() * xb.shape[0]
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.shape[0]
    val_loss = vloss / max(1, total)
    val_acc = correct / max(1, total)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Check for improvement
    if val_acc >= best_val_acc + min_delta:
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f" % (
            epoch + 1, EPOCHS, train_loss, val_loss, val_acc))
        
    if epochs_no_improve >= patience:
        print("Early stopping triggered at epoch %d. Best validation accuracy: %.4f at epoch %d." % (
            epoch + 1, best_val_acc, best_epoch))
        break

# Restore best weights.
if best_state is not None:
    model.load_state_dict(best_state)
print("Best val accuracy: %.4f" % best_val_acc)
print("Best epoch: %d" % best_epoch)

---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Consensus MLP loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
axes[1].plot(history["val_acc"], color="green")
axes[1].set_title("Consensus MLP val accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
plt.tight_layout()
os.makedirs("/kaggle/working/figures", exist_ok=True)
plt.savefig("/kaggle/working/figures/consensus_training_curves.png", dpi=120, bbox_inches="tight")
plt.show()

---
# Freeze the MLP; only the temperature parameter is optimised.
for p in model.mlp.parameters():
    p.requires_grad_(False)
model.temperature.requires_grad_(True)
model.eval()  # keep BatchNorm in eval mode using running stats

with torch.no_grad():
    val_logits = model.logits(Xva.to(DEVICE)).detach()
    val_targets = yva.to(DEVICE)

nll = nn.CrossEntropyLoss()
optimizer_t = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=100)


def _closure():
    optimizer_t.zero_grad()
    t = torch.clamp(model.temperature, min=1e-2)
    loss = nll(val_logits / t, val_targets)
    loss.backward()
    return loss


before = nll(val_logits, val_targets).item()
optimizer_t.step(_closure)
learned_temperature = float(torch.clamp(model.temperature, min=1e-2).detach().cpu().item())
with torch.no_grad():
    after = nll(val_logits / learned_temperature, val_targets).item()

print("Validation NLL before scaling: %.4f" % before)
print("Validation NLL after  scaling: %.4f" % after)
print("Learned temperature: %.4f" % learned_temperature)

---
from netcal.metrics import ECE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

with torch.no_grad():
    probs_before = F.softmax(val_logits, dim=1).cpu().numpy()
    probs_after = F.softmax(val_logits / learned_temperature, dim=1).cpu().numpy()
targets_np = yva.numpy()

ece_metric = ECE(bins=15)
ece_before = float(ece_metric.measure(probs_before, targets_np))
ece_after = float(ece_metric.measure(probs_after, targets_np))
print("ECE before temperature scaling: %.4f" % ece_before)
print("ECE after  temperature scaling: %.4f" % ece_after)

preds_after = probs_after.argmax(axis=1)
val_accuracy = accuracy_score(targets_np, preds_after)
print("Calibrated val accuracy: %.4f" % val_accuracy)

present = sorted(set(targets_np.tolist()) | set(preds_after.tolist()))
print("\nClassification report (validation):")
print(classification_report(
    targets_np, preds_after,
    labels=present,
    target_names=[ISIC_CLASSES[i] for i in present],
    zero_division=0,
))

cm = confusion_matrix(targets_np, preds_after, labels=list(range(NUM_CLASSES)))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ISIC_CLASSES, yticklabels=ISIC_CLASSES)
plt.title("Consensus MLP confusion matrix (val)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("/kaggle/working/figures/consensus_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()

# Reliability diagram (calibrated).
plt.figure(figsize=(6, 5))
conf = probs_after.max(axis=1)
correct = (preds_after == targets_np).astype(float)
bins = np.linspace(0, 1, 11)
idx = np.digitize(conf, bins) - 1
acc_per_bin, conf_per_bin, centers = [], [], []
for b in range(10):
    m = idx == b
    if m.sum() > 0:
        acc_per_bin.append(correct[m].mean())
        conf_per_bin.append(conf[m].mean())
        centers.append((bins[b] + bins[b + 1]) / 2)
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.plot(conf_per_bin, acc_per_bin, "o-", label="calibrated")
plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability diagram (calibrated)")
plt.legend(); plt.tight_layout()
plt.savefig("/kaggle/working/figures/consensus_reliability.png", dpi=120, bbox_inches="tight")
plt.show()

---
# Make sure the learned temperature is the value saved.
with torch.no_grad():
    model.temperature.copy_(torch.tensor([learned_temperature], dtype=torch.float32, device=model.temperature.device))

CKPT_PATH = "/kaggle/working/consensus_best.pth"
TEMP_PATH = "/kaggle/working/consensus_temperature.txt"

# Save as an envelope so the app can pick up the ECE alongside the weights; the
# app's ConsensusClassifier handles both {"state_dict": ..., "ece": ...} and a
# bare state_dict via strict=False.
state_dict = model.state_dict()  # includes mlp.* and temperature
torch.save({"state_dict": state_dict, "ece": ece_after}, CKPT_PATH)

with open(TEMP_PATH, "w") as f:
    f.write("%.6f\n" % learned_temperature)

print("Saved checkpoint:", CKPT_PATH)
print("Saved temperature:", TEMP_PATH, "->", learned_temperature)
print("state_dict keys:", list(state_dict.keys()))

# Sanity check: reload into a fresh model exactly as the app does (strict=False).
_check = ConsensusClassifier()
_loaded = torch.load(CKPT_PATH, map_location="cpu")
_sd = _loaded["state_dict"] if isinstance(_loaded, dict) and "state_dict" in _loaded else _loaded
_res = _check.load_state_dict(_sd, strict=False)
print("Reload check -> missing:", _res.missing_keys, " unexpected:", _res.unexpected_keys)
assert len(_res.missing_keys) == 0 and len(_res.unexpected_keys) == 0, "Checkpoint will not load cleanly in the app!"
print("Checkpoint loads cleanly into a fresh ConsensusClassifier (matches the app contract).")
