# Argus Vision — Remediation Report

Living record of the gated remediation (Phases 0–9). **Honesty calibration:** published,
leakage-checked ISIC-2019 8-class results top out ~74–80% balanced accuracy even with heavy
ensembling; any >90% balanced-accuracy number here should be treated as a leakage/bug red flag,
not a win. Every numeric claim below comes from code actually run; results are labelled
**[verified-here]** (ran in this environment, synthetic/logic-only where noted) vs
**[pending-Kaggle]** (needs real ISIC data / GPU).

> No local image data and no torch/numpy/sklearn in the local env (by design), so Phase 0–8
> verification is synthetic/logic-only unless stated. Real-data verification happens on Kaggle.

---

## Phase 0 — Reconnaissance
Confirmed every described root cause (image-level split leakage; `Resize(224)` inference vs
`Resize(256)` training; `RandomResizedCrop` scale 0.8 vs 0.7; `1/sqrt` weighting; `strict=False`
loads without assertions; `scikit-learn==1.5.0` and unpinned `lightgbm`; no tests; Dockerfile runs
as root with `--reload`). **Blocker:** `ISIC_2019_Training_Metadata.csv` (with `lesion_id`) is not
in the repo and not referenced anywhere — data is mounted on Kaggle (`andrewmvd/isic-2019`). Resolved
by the user (will mount metadata on Kaggle).

## Phase 1 — Lesion-grouped split  [verified-here: synthetic]
- **NEW `splits.py`**: `get_lesion_grouped_split(df, label_col, ...)` (`label_col` required),
  `attach_lesion_ids`, `assert_no_lesion_leakage`, `lesion_grouped_indices` (array core).
  Partial-`lesion_id` coverage handled (present → group by lesion; missing → singleton image group),
  coverage % printed loudly. Primary = sklearn `StratifiedGroupKFold`→`GroupShuffleSplit`.
- **sklearn guarantee:** module-level `_SKLEARN_AVAILABLE` (`ImportError`) guard; the pure-Python
  fallback is reachable ONLY when sklearn is genuinely absent — a runtime error in a sklearn splitter
  now propagates, it is never silently downgraded. Proven at runtime (sklearn-present → SGKF used,
  pure-Python calls = 0; sklearn-absent → pure-Python only).
- Notebooks 01–05 wired to import & call `splits.py`; NB03 `val_dataset`→`mining_dataset`;
  NB04's consensus-feature split also lesion-grouped. `config.py` `PHASE2_MAX_EPOCHS=40`/`PHASE2_PATIENCE=8`
  canonicalised (anti-drift); NB01/02 read them.
- Evidence: synthetic 4160 imgs/2414 lesions → zero lesion/group overlap, exact partition, all rare
  classes retain val examples, 0.150 val fraction. **[pending-Kaggle]:** real coverage %, sklearn path.

## Phase 2 — Preprocessing unification  [verified-here: structural + exact geometry]
- **NEW `transforms.py`** (canonical `get_train_transform`/`get_eval_transform` = Resize(256)→
  CenterCrop(224)→ToTensor→Normalize). `image_service.py` `Resize(224)`→`Resize(256)`. `dataset.py`
  delegates (duplicate `Compose` removed; RandomResizedCrop scale 0.8→0.7). Notebooks 01–05 use the
  canonical functions; no inline `Compose` remains outside `transforms.py` + the backend mirror.
- Geometry proof: OLD inference retained 100% of the shorter side; NEW = training eval = 87.5% — the
  measured train/inference mismatch, now fixed. **[pending-Kaggle]:** pixel-checksum (needs torchvision).
- **Drift guard:** `backend/tests/test_transform_parity.py` asserts the backend transform and
  `transforms.get_eval_transform` share resize/crop/normalization (AST path runs anywhere; live
  torchvision introspection on Kaggle/CI). Proven to fail on simulated drift.

## Phase 3 — Effective-number weighting  [verified-here: 100k-draw sim]
- **NEW `weighting.py`** `effective_number_weights` (Cui et al.) — the SINGLE array driving both the
  sampler and the FocalLoss alpha. `config.EFFECTIVE_NUMBER_BETA`. `dataset.py` delegates (also fixed a
  latent bug: old sampler used 1/count while focal alpha used 1/sqrt). NB01/02 wired (same `class_weights`
  → sampler + alpha; `inv_sqrt` removed).
- **Key finding:** β=0.99 (as originally specified) is a near-no-op on ISIC (50:1 sampling imbalance,
  worse than 1/sqrt's 7.4:1) because `1/(1-β)=100` < every class count. Measured β sweep → **β=0.999
  chosen** (11.6:1, DF 0.94%→3.85%): a real rebalancing without the small-pool over-repetition /
  loss-double-counting risk of β=0.9999 (1.8:1). Reasoning documented in `config.py`.

## Phase 4 — Decoupled two-stage training  [verified-here: structural + logit math; freeze/smoke pending-Kaggle]
- **NEW `training_utils.py`**: `freeze_all_but_classifier` (timm `get_classifier`, both archs),
  `snapshot_frozen_params`/`assert_frozen_unchanged` (param-level bit-identical freeze proof),
  `class_priors_from_counts` (empirical), `apply_logit_adjustment` + `LogitAdjustedLoss`.
- **`config.py`**: `TRAINING_MODE` (joint|decoupled), `STAGE_A_EPOCHS`/`STAGE_B_EPOCHS`,
  `USE_LOGIT_ADJUSTMENT`/`LOGIT_ADJUSTMENT_TAU`.
- **NB01/NB02 fine-tune restructured**: decoupled = Stage A (full network, plain instance-balanced loader,
  UNWEIGHTED focal) → reload best → Stage B (freeze backbone, weighted sampler + weighted focal, head-only,
  then `assert_frozen_unchanged`). Joint path preserved verbatim. Both reuse the notebook's own
  `run_epoch`/`evaluate` via one shared `_run_stage` helper. Logit adjustment toggles the Stage-B / joint loss.
- Evidence: structural markers present in both (verified); logit-adjustment + empirical-priors math
  [verified-here]. `verify_decoupled.py` proves the freeze on the real EfficientNet-B4 + ViT-B/16
  (random tensors) — **[pending-Kaggle]** (needs torch). Full end-to-end smoke = run the notebook with small
  STAGE_*_EPOCHS on Kaggle. **Decoupled-vs-joint is NOT decided here — it requires a full Kaggle A/B run.**

## Phase 5 — Checkpoint / resume reliability  [verified-here: structural; LR-match pending-Kaggle]
- **`training_utils.py`**: `save_resumable` (atomic write via tmp+os.replace; model+optimizer+scheduler+
  AMP-scaler state + epoch + best_auc + epochs_no_improve) and `load_resumable` (restores all in place,
  returns `start_epoch = saved+1` and the counters).
- Wired into BOTH notebooks' shared `_run_stage` (so Stage A, Stage B AND joint each resume independently):
  the resume file is derived as `ckpt_path + ".resume"` (so each stage has its own), loaded at stage start,
  and re-saved every epoch; the loop now runs `range(start_epoch, ...)`. The production **best-weights-only**
  file (`agent_*_best.pth`, plain `state_dict`) is written separately and unchanged, so the backend loader
  needs no changes.
- **`verify_resume.py`** (Kaggle, ~1s CPU, no data): saves mid-cosine-schedule, reloads into FRESH
  optimizer/scheduler/scaler, asserts the LR equals the uninterrupted schedule at that epoch and that
  best_auc / epochs_no_improve are restored — the brief's exact check. **[pending-Kaggle]** (needs torch).

## Phase 6 — Misclassification audit tooling  [verified-here: synthetic end-to-end]
- **NEW `audit.py`** (sklearn/pandas/matplotlib-optional, stdlib fallbacks). `run_audit(y_true, y_pred,
  y_prob, image_paths, class_names, lesion_ids, out_dir, top_n)` writes:
  `confusion_matrix.csv` + `confusion_matrix_normalized.csv` (+ `.png` when matplotlib present),
  `per_class_metrics.csv` (precision/recall/f1/support/**PR-AUC** + macro/weighted rows — PR-AUC was
  missing before), and `top{N}_confident_wrong.csv` (`rank, image_path, true_class, predicted_class,
  confidence, all_class_probabilities` [JSON of all 8], `lesion_id`). Prints a console summary
  (balanced acc, macro-F1, macro PR-AUC, weighted-F1, per-class table, confusion matrix, top-5 preview).
- Evidence: `verify_audit.py` runs the whole pipeline on a 2531-sample synthetic fixture (stdlib only) —
  all CSV schemas validated, top-wrong rows are genuine errors sorted by confidence. On Kaggle the same
  call uses sklearn's `average_precision_score` + writes the PNG. **[pending-Kaggle]:** running it against
  the CURRENT pre-fix checkpoints on the lesion-grouped val split (the brief's baseline number).

## Phase 7 — Package Kaggle notebooks  [verified-here: structural]
- NB01/02 now have a top **RUN CONFIG** banner echoing the active `config.py` settings (IMAGE_SIZE,
  TRAINING_MODE, β, stage epochs, logit-adjust) + a resume-file detection print; auto-resume itself is
  handled inside `_run_stage` (Phase 5). The toggleable config block is `config.py` (single source).
- NB05 **auto-invokes the Phase-6 audit** at the end: `from audit import run_audit` + a final cell calling
  `run_audit(y_true, proba["Argus (full)"].argmax(1), proba["Argus (full)"], image_paths, ISIC_CLASSES,
  lesion_ids=…, out_dir="/kaggle/working")` (lesion ids mapped from `eval_df` in `image_paths` order).
- 03/04/05 already import the canonical `splits`/`transforms`/`weighting` modules (Phases 1–2);
  left structurally unchanged otherwise. Full ordered **Kaggle run plan** written below.

## Phase 8 — Small fixes  [verified-here: syntax]
- **`requirements.txt`**: `scikit-learn==1.5.0 → ==1.6.1` (pickle/scaler version safety); `lightgbm` was
  unpinned → `lightgbm==4.5.0`. **`Dockerfile`**: explicit `RUN pip install scikit-learn==1.6.1` so the
  serving sklearn matches the version that fit the consensus scaler/model, "permanently in the Dockerfile".
- **Asserts after every `load_state_dict(strict=False)`**: `agent_a.py`, `agent_b.py`, `consensus/classifier.py`
  now `assert not missing and not unexpected` (with the offending keys in the message) — a mismatched
  train/serve model is refused loudly instead of silently serving wrong predictions. (No allow-list needed:
  a clean checkpoint matches each architecture exactly.) All three still parse.

## Phase 9 — Backend / infra hardening  [verified-here: whole backend ast-parses clean + structural; execution pending-env]
Audited read-only across all 6 dimensions (6 agents), then applied contract-safe fixes (7 agents) — every
change reviewed + the full backend re-parsed clean. **Already-good (no change):** CORS read from env (not
wildcard), UUID-only save paths (no path traversal), **zero hardcoded secrets**, consistent `ArgusError`
hierarchy, no bare `except`, real `/health` dependency checks, all four ML deps exact-pinned.
- **9.1 input/errors:** `classify.py` — size rejected via `file.size` + bounded `read(max_bytes+1)` BEFORE
  buffering (memory-DoS); **corrupt-but-typed uploads now `Image.verify()` → 422** (⚠ flagged contract change,
  below); `pipeline is None` (503) checked BEFORE writing the temp file (no orphan); `create_job` failure cleans
  the temp file. WS `debate_stream.py` — Redis outage at connect now `close(1011)` (was an undefined abnormal
  close); 4004 not-found preserved. `main.py` 500 handler returns a generic `detail` (no `str(exc)` leak), shape
  intact.
- **9.2 resilience:** `job_service.py` Redis client now has `socket_connect_timeout`/`socket_timeout` (fail-fast,
  no hang) + native `Retry(ExponentialBackoff(), N)` on `ConnectionError/TimeoutError` (transient blips auto-retry).
  Degraded-mode (`PRETRAINED_FALLBACK` / untrained consensus) logs raised `warning → error` (unmissable).
  `pipeline.run` removes the temp image in a `finally`.
- **9.3 logging:** `job_id` now threaded through `classify.py` + per-stage `pipeline.py` logs; no `print()` in
  prod paths (confirmed).
- **9.4 security:** `allowed_origins_list` now drops a lone `"*"` so wildcard-with-credentials CORS can never be
  assembled; secrets/traversal confirmed clean.
- **9.5 tests:** NEW `backend/tests/test_features.py` (23-dim layout + the natural-log-vs-base-2 JS guard),
  `test_trigger.py` (JS/entropy fire rules incl. the OR-branch), `test_classify_route.py` (200 shape + 422 bad
  type + 422 corrupt), `__init__.py`, `requirements-dev.txt` (pytest, httpx).
- **9.6 config/deps:** magic numbers hoisted to `config.py` (TEMP_IMAGE_DIR [dedup across classify+jobs],
  JOB_TTL, WS keepalive/poll, HOST/PORT/version/estimated, Redis timeouts/retry); `pipeline.py` bbox now uses
  `image_service.RESIZE_SIZE/IMAGE_SIZE` (no drift on the `attention.bbox` contract); Dockerfile runs as a
  **non-root** user + dropped `--reload` (dev reload via a compose `command:` override); `lightgbm==4.5.0` pinned.

**Verification status:** the whole backend **ast-parses clean** and every fix was reviewed [verified-here].
Running the pytest suite, the corrupt-upload-4xx repro, and the health-check-failure demo need a live backend
(torch/fastapi/redis), which isn't available locally → **[pending-env]**. To run on a backend host:
`pip install -r backend/requirements.txt -r backend/requirements-dev.txt && pytest backend/tests -q`.

---

## 🔖 Flagged follow-ups (deferred — do NOT silently act on these)
1. **More aggressive β for Phase-4 Stage B only.** Stage B retrains only a *frozen-backbone* head, which
   carries far lower overfitting risk than full fine-tuning, so a higher β (more aggressive rebalancing)
   for Stage B's sampler may be justified — but that's a Phase-4 decision with its own evidence, not
   pre-decided. (Raised during Phase 3.)
2. **[RESOLVED]** ~~Freeze backbone BatchNorm running stats in Stage B.~~ Implemented `freeze_backbone_bn`
   (sets BN to eval + no-op `train` so `model.train(True)` can't re-enable it) and wired it into both
   notebooks' Stage B after `freeze_all_but_classifier`; `verify_decoupled.py` now also asserts every backbone
   BUFFER (BN running_mean/var) is bit-identical after Stage-B steps. No-op for ViT (LayerNorm). The backbone
   is now fully frozen (params + buffers) in Stage B.
2. **`transforms.py` as a properly shared package** between backend and ml_training (currently a comment-
   synced duplicate guarded by `test_transform_parity.py`). Real infra work → Phase 9.
3. **`dataset.py` unused `IMAGENET_MEAN/STD` imports** after transforms moved out — harmless; tidy later.
4. **`ml_training/debate_text_utils.py`** appears to be dead code (788-dim Groq contract removed) —
   candidate for removal in a future cleanup, not this pass.
5. **Phase-9 LOW items deferred (each contract-touching or low-value, left for a deliberate decision):**
   `health.py` returns HTTP 200 + `status:"ok"` even when `redis_connected/model_loaded` are false (a separate
   `/health/ready` returning 503 is the non-breaking option — don't mutate `/health`); add a `degraded` /
   `using_pretrained_fallback` health field so a random-head fallback is operationally visible; `datetime.utcnow()`
   → `datetime.now(timezone.utc)` (adds a `+00:00` offset to serialized timestamps — confirm the frontend parses
   it); keep a strong reference to the fire-and-forget `asyncio.create_task(pipeline.run(...))` (GC can cancel it);
   WS add a `receive()` task so an ungraceful disconnect is detected promptly (≤30s today); pin the Docker base to
   a digest for fully reproducible builds. Non-root Dockerfile + named-volume cache path **need a `docker compose
   up` smoke test** (volume-ownership init) — can't verify locally.

## ⚠️ External-contract touches (flagged, per the Phase-9 instruction)
- **Phase 9 — corrupt-upload path `200 → 422` (the one intentional observable change).** A corrupt/non-image
  file with a valid `image/png|jpeg` Content-Type previously returned `200 {"status":"queued"}` and only failed
  asynchronously (job `status="failed"` + an `error` WS event); it now fails synchronously with `422` via the
  SAME `ImageProcessingError` envelope (`{"error":"ImageProcessingError","detail":"Uploaded file is not a valid
  image."}`). **Why necessary:** Phase 9.1 explicitly requires "a corrupt/non-image file produces a clean 4xx",
  which the async-fail path did not provide; the 422 is also better UX (the upload page surfaces it immediately
  instead of navigating to a debate that then fails). The error-envelope SHAPE is unchanged. The new
  `test_classify_route.py` asserts the 422. If you prefer to keep the old `200`-then-async-fail contract, remove
  the `Image.verify()` block in `classify.py` and flip that test.
- The new WS `close(1011)` on a backend-lookup failure replaces a previously *undefined/abnormal* close — not a
  defined contract, so this is a strict improvement, not a contract change. `4004` (not-found) is unchanged.
- Phase 2 changed `image_service.py` preprocessing (Resize 224→256) — fixes a train/inference mismatch; model
  input stays 224×224, response/WS shapes unchanged (contract-safe).

## What to run on Kaggle — in order

**Prerequisites (one-time):** attach the `andrewmvd/isic-2019` dataset; **mount
`ISIC_2019_Training_Metadata.csv`** (with `lesion_id`) so the lesion-grouped split engages (else it warns
loudly and degrades to image-level grouping); and attach the `ml_training/` folder (or at least `splits.py`,
`transforms.py`, `weighting.py`, `config.py`, `training_utils.py`, `audit.py`) so the notebooks' bootstrap
`from splits import …` etc. resolve. Set Internet ON (pip + pretrained weights). The RUN CONFIG cell at the
top of NB01/02 echoes the active `config.py` settings and any resume files.

1. **Agent A — honest new baseline.** NB01 with `config.IMAGE_SIZE=224`, `TRAINING_MODE="joint"`,
   `USE_LOGIT_ADJUSTMENT=False`. This is the Phase-1/2/3 fixes WITHOUT decoupling. Evaluate with NB05 (the
   Phase-6 audit auto-runs). **Expect the balanced accuracy to be LOWER than the old leaky 73.8%** — the
   lesion-grouped split is genuinely harder; that's correct, not a regression. (>90% bal-acc = red flag.)
2. **Agent A — decoupled.** NB01 with `TRAINING_MODE="decoupled"`, same resolution. Compare per-class recall
   / PR-AUC on **DF/VASC/AK/SCC** against step 1 — this is the test of whether decoupled training earns its keep.
   *Decision gate: if VASC per-class recall is still below 0.30 after the decoupled run, do not proceed to Agent B — apply β=0.9999 for Stage B only and retrain Agent A first.*
3. **(Optional) logit adjustment.** `USE_LOGIT_ADJUSTMENT=True` on top of whichever mode won steps 1–2.
4. **(Only if 1–3 don't already clearly win, and GPU budget allows) resolution.** Re-run the best config at
   `config.IMAGE_SIZE=320` (320 before 380 — cheaper, test first). Note: ViT-B/16 (Agent B) is fixed at 224,
   so the resolution experiment is Agent A only.
5. **Agent B.** Only after Agent A shows real improvement, run NB02 with Agent A's winning config.
6. **Consensus.** Only after BOTH agents improve, rebuild `hard_subset.csv` (NB03) and retrain the consensus
   head (NB04). Run `verify_decoupled.py` and `verify_resume.py` once on Kaggle to confirm the freeze/resume tools.

- **β tuning note (Stage B):** the global `EFFECTIVE_NUMBER_BETA=0.999` is the safe default. **For Stage B
  specifically, test β=0.9999 over β=0.999** — since only the head trains there, overfitting risk from more
  aggressive rebalancing is lower, so more uniform sampling may help the rare classes without the calibration
  cost it would carry during full-network training.
