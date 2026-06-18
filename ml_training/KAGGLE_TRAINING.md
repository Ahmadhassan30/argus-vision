# Training Argus Vision models on Kaggle (free GPU)

This guide walks you through training every model in the Argus Vision pipeline on
[kaggle.com](https://www.kaggle.com) using their **free GPU** (NVIDIA T4 ×2 or
P100), then plugging the trained weights back into the Dockerized app.

The five notebooks in this folder are **Kaggle-native and self-contained**: they
read the dataset from `/kaggle/input/`, train on GPU with mixed precision, and
write checkpoints to `/kaggle/working/`. You do **not** need to upload any
helper `.py` files — each notebook inlines everything it needs.

---

## What you will produce

| Notebook | Trains / produces | Output file(s) in `/kaggle/working/` |
| --- | --- | --- |
| `01_train_agent_a.ipynb` | Agent A — EfficientNet-B4 classifier | `agent_a_best.pth` |
| `02_train_agent_b.ipynb` | Agent B — ViT-B/16 classifier | `agent_b_best.pth` |
| `03_build_hard_subset.ipynb` | The "hard" disagreement subset | `hard_subset.csv` |
| `04_train_consensus.ipynb` | Calibrated consensus MLP | `consensus_best.pth`, `consensus_temperature.txt` |
| `05_evaluation.ipynb` | Full evaluation report (tables + case studies) | figures / tables |

The three `.pth` files are what the backend loads. Drop them into
`backend/checkpoints/` and the app uses your trained models instead of the
ImageNet-pretrained fallback.

---

## The dataset you need to add on Kaggle

**Primary:** **ISIC 2019** — https://www.kaggle.com/datasets/andrewmvd/isic-2019

- 25,331 dermoscopic images, the exact 8 classes Argus uses
  (MEL, NV, BCC, AK, BKL, DF, VASC, SCC).
- Includes `ISIC_2019_Training_GroundTruth.csv` (one-hot labels) and the
  `ISIC_2019_Training_Input/` image folder.
- Mounts at `/kaggle/input/isic-2019/` once attached.

**Mirror (fallback):**
https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification

> The notebooks **auto-discover** the ground-truth CSV and the image folder
> anywhere under `/kaggle/input/` (via `discover_isic()`), so they work with
> either dataset and regardless of how the folders are nested. You don't have to
> hard-code any paths.

---

## One-time prerequisites

1. Create a free account at https://www.kaggle.com.
2. **Verify your phone number**: top-right avatar → **Settings** → **Phone
   Verification**. This is **required** to enable GPU and Internet in notebooks.

---

## Step-by-step: train Agent A (notebook 01)

> Repeat the same flow for notebook 02 (Agent B). They are independent and can be
> run in either order.

### 1. Create the notebook
- Go to https://www.kaggle.com/code → **New Notebook**.
- In the new notebook: **File → Import Notebook → Upload** and pick
  `ml_training/01_train_agent_a.ipynb` from this repo.
  *(Alternatively: clone this GitHub repo, or copy the cells in manually.)*

### 2. Attach the dataset
- Right panel → **Input → + Add Input** (or **Add Data**).
- Search **`isic-2019`**, find **andrewmvd/isic-2019**, click **+** to attach.
- It now appears under `/kaggle/input/isic-2019/`.

### 3. Turn on GPU and Internet
- Right panel → **Session options** (the **⋮ / Settings** / **three dots**):
  - **Accelerator → GPU T4 ×2** (or **GPU P100**).
  - **Internet → On**  ← needed for `pip install` and to download the
    pretrained backbone weights the first time.
- Without GPU, training will be far too slow; without Internet, the first cell
  (pip install + pretrained-weight download) will fail.

### 4. Run it
- **Run All** (▶▶), or run cells top-to-bottom.
- The first code cell pip-installs the extras (`timm`, `grad-cam`,
  `torchmetrics`), then everything else runs on GPU with mixed precision.
- Expected wall-clock on a T4: roughly **2–4 hours** for the full
  head-warmup + fine-tune schedule (depends on accelerator and epochs).
  - If you hit a **CUDA out-of-memory** error, lower `BATCH_SIZE` from `32` to
    `16` in the data cell and re-run.

### 5. Save the output (important!)
Kaggle deletes the interactive session's files when it times out. To keep
`agent_a_best.pth`:
- Click **Save Version** (top-right) → **Save & Run All (Commit)**.
- When the committed run finishes, its `/kaggle/working/` becomes a permanent
  **notebook output** you can download or attach to other notebooks.
- You can also grab the file immediately from the right panel **Output** tab →
  download `agent_a_best.pth`.

---

## Chaining notebooks 03 / 04 / 05 (they need the trained weights)

Notebooks 03–05 need the checkpoints produced by 01 and 02 (and 04's output for
05). On Kaggle you make one notebook's output available to another by **attaching
it as input**:

1. Open notebook 03 (import it, attach the `isic-2019` dataset, enable GPU + Internet as above).
2. Right panel → **Input → + Add Input → Your Work** (or **Notebook Output**) →
   select your committed **01_train_agent_a** and **02_train_agent_b** runs.
   - Their files now appear under `/kaggle/input/<your-notebook-slug>/`.
3. The notebooks locate the checkpoints automatically with `find_file("agent_a_best")`
   / `find_file("agent_b_best")` — searching both `/kaggle/input` and
   `/kaggle/working`. If a checkpoint isn't attached, the notebook prints a clear
   message and falls back to pretrained weights so it still runs.

Attachment summary:

| Notebook | Attach as input |
| --- | --- |
| 03 | `isic-2019` + outputs of 01 & 02 |
| 04 | `isic-2019` + outputs of 01 & 02 + **Groq secret** (below) |
| 05 | `isic-2019` + outputs of 01, 02, 04 + `hard_subset.csv` from 03 |

---

## Groq API key for notebook 04 (the consensus debate)

Notebook 04 calls the Groq LLM to generate the debate arguments. Add your key as
a **Kaggle Secret** so it isn't hard-coded:

1. Get a free key at https://console.groq.com/keys.
2. In the notebook: **Add-ons → Secrets → + Add a new secret**.
   - **Label:** `GROQ_API_KEY`
   - **Value:** your `gsk_...` key
   - Toggle it **attached** to this notebook.
3. The notebook reads it via:
   ```python
   from kaggle_secrets import UserSecretsClient
   GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
   ```
   If the secret is missing, the notebook falls back to deterministic argument
   text so it still completes (consensus quality is just lower).

> Notebook 04 **caches** every Groq response to
> `/kaggle/working/groq_cache.json`, keyed by image id, and skips anything
> already cached. This saves time/cost and survives re-runs. Attach this cache to
> notebook 05 to avoid re-calling Groq during evaluation.

---

## Getting the trained models back into the app

After committing notebooks 01, 02, and 04:

1. Download the three checkpoints from each notebook's **Output** tab:
   - `agent_a_best.pth`  (notebook 01)
   - `agent_b_best.pth`  (notebook 02)
   - `consensus_best.pth`  (notebook 04)
2. Place all three into the repo at:
   ```
   argus-vision/backend/checkpoints/
   ├── agent_a_best.pth
   ├── agent_b_best.pth
   └── consensus_best.pth
   ```
   (The filenames must match the defaults in `backend/.env.example` /
   `core/config.py`: `AGENT_A_CHECKPOINT`, `AGENT_B_CHECKPOINT`,
   `CONSENSUS_CHECKPOINT`.)
3. Start the stack:
   ```bash
   docker compose up --build
   ```
   On startup the backend finds the checkpoints and loads your trained models
   instead of the ImageNet fallback. Confirm via `GET http://localhost/api/health`
   → `"model_loaded": true`.

> `*.pth` files are git-ignored, so they won't be committed. Keep them locally or
> store them in your own release/artifact storage.

---

## Good-to-know about Kaggle GPU

- **Quota:** ~30 GPU-hours/week, and a single session runs up to ~12 hours
  (9 hours interactive). Train 01 and 02 in separate sessions if needed.
- **Save Version** is the reliable way to persist outputs — interactive sessions
  are ephemeral.
- **Mixed precision** (`torch.cuda.amp`) is already used in the training
  notebooks for speed and lower memory.
- **`/kaggle/working/` is the only writable directory**; `/kaggle/input/` is
  read-only.
- Increase epochs / unfreeze more layers for better accuracy if your quota
  allows; the defaults are a reasonable balance for a free session.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pip install` fails / pretrained download hangs | Internet is **Off** — enable it in Session options. |
| Cell 1 can't find the CSV/images | The dataset isn't attached, or a different ISIC set — re-attach `andrewmvd/isic-2019`; `discover_isic()` prints what it found. |
| `CUDA out of memory` | Lower `BATCH_SIZE` to 16 (or 8); restart the session to clear GPU memory. |
| `No accelerator` / training is extremely slow | Accelerator is set to **None/CPU** — switch to GPU T4 ×2. |
| Notebook 03/04/05 says a checkpoint is missing | Attach the committed outputs of 01/02 (and 04) via **Add Input → Your Work**. |
| Notebook 04 uses fallback arguments | `GROQ_API_KEY` secret not attached — add it under **Add-ons → Secrets**. |
| ViT model id not found in `timm` | The first cell installs `timm` with `-U`; make sure that cell ran (needs Internet). |
