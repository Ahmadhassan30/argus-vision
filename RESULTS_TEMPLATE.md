# Argus Vision — Results Record (23-dim consensus rebuild)

This is a factual record of what was audited, what was changed, and the
before/after metrics. It is **not** a summary of successes only. After-values
are marked `[TO BE FILLED AFTER KAGGLE RUN]` because the notebooks were prepared
for the human to run on Kaggle; **no Kaggle output is simulated or reported
here**. Fill the placeholders from the actual notebook output, then this file
becomes `RESULTS.md`.

The notebooks (`01`–`05`) were edited for Kaggle and are ready to upload. Run
order: `01_train_agent_a` → `02_train_agent_b` → `03_build_hard_subset` →
`04_train_consensus` → `05_evaluation`, attaching each notebook's output to the
downstream ones (see each notebook's final markdown cell).

---

## 1. Original bug and the architectural decision

The consensus MLP previously collapsed to **18.0%** balanced accuracy due to a
feature-contract mismatch: the deterministic fallback-argument text was worded
differently between the training notebook (04) and the evaluation notebook (05),
so the 384-d sentence embeddings were out-of-distribution at eval time. Unifying
the wording restored balanced accuracy to **66.6%**.

The LLM-debate-text + 788-dim sentence-embedding architecture has now been
**removed entirely** and replaced with a **23-dimensional pure numerical** feature
vector. Reasons (established, not re-litigated here): Groq rate-limited after 2–3
images, making it unusable for the ~6,000–8,000 calls a training run needs; in
practice ~100% of the previous "arguments" were deterministic fallback templates,
not real LLM output; 788 features with ~1,500 samples is a 1.9:1
sample-to-feature ratio (statistically indefensible), whereas 23 features give
~65:1. The 23-dim vector retains the real information the debate produced.

**23-dim feature layout** (identical in `04`, `05`,
`backend/ml/debate/features.py`, and `ml_training/debate_text_utils.py`):

| idx | feature | meaning |
|-----|---------|---------|
| 0–7 | `pA` | Agent A softmax probabilities (MEL,NV,BCC,AK,BKL,DF,VASC,SCC) |
| 8–15 | `pB` | Agent B softmax probabilities (same order) |
| 16 | `js_div` | Jensen–Shannon divergence between pA and pB |
| 17 | `entropy_a` | Shannon entropy of pA (bits) |
| 18 | `entropy_b` | Shannon entropy of pB (bits) |
| 19 | `max_prob_delta` | max\|pA − pB\| over the 8 classes |
| 20 | `attn_iou` | IoU of the two attention maps thresholded at 0.5 (0.0 if unavailable) |
| 21 | `attn_entropy_a` | entropy of Agent A's normalized attention map (0.0 if unavailable) |
| 22 | `attn_entropy_b` | entropy of Agent B's normalized attention map (0.0 if unavailable) |

---

## 2. Phase 1 audit findings (read before editing)

The plan listed four candidate base-agent fixes (a)–(d). The actual notebook
code was read in full first; here is what was found, with evidence, and what was
therefore changed:

| Candidate fix | Audit finding (from the actual code) | Action |
|---------------|--------------------------------------|--------|
| (a) class-imbalance handling in the loss | **Already present.** Both agents use `FocalLoss(gamma=2.0, alpha=class_weights)` where `class_weights` = inverse-**sqrt**-frequency normalized to mean 1 (NB01 cell "FocalLoss ready", NB02 cell "FocalLoss"). | **Not changed.** Adding the plan's plain-CrossEntropy block would have *replaced* a working focal loss. |
| (b) balanced sampling | **Already present.** Both use a `WeightedRandomSampler` weighted by inverse-**sqrt** frequency (NB01/NB02 dataset cells). | **Not changed.** The plan's suggested sampler uses **raw** inverse frequency (`1/count`), which is *harsher* than what exists and is the exact scheme the project log says collapsed NV recall 0.86→0.08. Applying it would have been a regression, so it was deliberately avoided. |
| (c) training stopped before convergence | **Confirmed structural risk.** Phase-2 fine-tuning ran a **fixed 15 epochs** with `CosineAnnealingLR(T_max=15)` and **no early stopping**. (The committed notebooks contain no saved cell outputs, so convergence could not be read from logs — only the fixed-budget structure is verifiable.) | **Changed** (see §3). |
| (d) insufficient capacity unfrozen | Agent A unfreezes the last **2** EfficientNet-B4 stages + head; Agent B unfreezes the last **4** transformer blocks + norm + head (the depth the plan would recommend). The plan's threshold is <5M (B4) / <10M (ViT). The notebooks print the trainable-param count. | **Not changed**; the count is printed so the human can confirm. For Agent A the unfreeze depth is exposed as `PHASE2_UNFREEZE_BLOCKS` with a printed note to raise it if the count is < 5M. |

Established conclusion (unchanged): the base agents, not the consensus MLP or the
trigger thresholds, are the bottleneck.

---

## 3. What changed, by file

- **`01_train_agent_a.ipynb`** — Phase-2 fine-tune: fixed 15 epochs → up to
  `PHASE2_MAX_EPOCHS=40` with early stopping (`PATIENCE=8`) on validation
  macro-AUC; `CosineAnnealingLR` `T_max` updated to the new budget; unfreeze depth
  exposed as `PHASE2_UNFREEZE_BLOCKS=2` with a printed param-count note. Added a
  per-class validation report cell (balanced accuracy + per-class recall with a
  `<0.40` flag). Loss and sampler **unchanged**.
- **`02_train_agent_b.ipynb`** — same Phase-2 early-stopping change (two-param-group
  optimizer preserved) + per-class report cell. Loss, sampler, and unfreeze depth
  **unchanged**.
- **`03_build_hard_subset.ipynb`** — thresholds **unchanged** (`JS=0.25`,
  `ENTROPY=0.8`); added a cell printing the overall + per-class fire rate for
  direct comparison against the previous 98.7%.
- **`04_train_consensus.ipynb`** — removed all Groq / sentence-transformer /
  argument-generation / 788-dim code; builds the 23-dim feature matrix from
  Agent A/B softmax + Grad-CAM++/rollout attention maps; prints shape, NaN/Inf
  check, per-feature mean/std, and samples-per-class; fits `StandardScaler` on the
  **train split only** and saves `consensus_scaler.pkl` (+ `.json` sidecar); new
  MLP `23→128→64→8`; training = Adam(1e-3), `ReduceLROnPlateau` on val balanced
  accuracy (patience 10), early stopping (patience 20), max 200, batch 64,
  sqrt-inverse clamped class weights + label smoothing 0.1; temperature scaling +
  ECE retained.
- **`05_evaluation.ipynb`** — removed all Groq/embedding code; loads
  `consensus_scaler.pkl`/`.json`; Argus = calibrated consensus MLP on the
  standardized 23-dim vector for every image (attention features computed when the
  trigger fires, else 0.0); ablation rewritten to the 23-dim feature groups; case
  studies replaced debate text with a numeric feature summary. Added Phase 4
  (malignant/benign + MEL→NV), Phase 5 (abstention sweep + plot), Phase 6
  (bootstrap CIs), Phase 7 (error analysis).
- **`backend/ml/debate/features.py`** — **new**: canonical `extract_consensus_features`.
- **`backend/ml/debate/argument_gen.py`, `backend/ml/debate/encoder.py`** — **deleted**.
- **`backend/ml/consensus/classifier.py`** — 23-dim MLP; loads + applies the
  StandardScaler before the MLP; `predict(prob_a, prob_b, attn_map_a, attn_map_b)`.
- **`backend/ml/pipeline.py`** — removed Groq client, encoder, two-round debate,
  token streaming, probability nudging; flow is now agents → trigger → (attention
  if fired) → 23-dim feature → consensus → `debate=None`.
- **`backend/core/config.py`** — removed `GROQ_API_KEY`/`GROQ_MODEL`; added
  `CONSENSUS_SCALER`.
- **`backend/.env.example`**, **`backend/requirements.txt`** — removed Groq +
  sentence-transformers; added scikit-learn + joblib.
- **`ml_training/debate_text_utils.py`** — repurposed to hold only the canonical
  `extract_consensus_features` (old class-description / fallback-argument text
  removed).

Engineering deviations from the literal plan (documented for honesty):
1. The plan's NB04 snippet reads `row['prob_a']` from `hard_subset.csv`, but NB03
   never stored probability vectors. NB04 is self-contained and re-runs the agents
   to obtain `pA`/`pB` and the attention maps, so NB03 did **not** need a new
   probability-persistence contract.
2. The MLP attribute is `self.mlp` (not the plan's `self.net`) and keeps a learnable
   `temperature` — required to preserve temperature scaling and keep the checkpoint
   parameter names identical across NB04/NB05/backend.
3. The scaler is saved both as `consensus_scaler.pkl` (joblib, per the plan) and as
   a `consensus_scaler.json` `{mean, scale}` sidecar so the backend can standardize
   with numpy only if scikit-learn/joblib pickling is unavailable.
4. Argus runs the consensus MLP on **every** test image for eval↔serve consistency
   (the backend does the same). Caveat: the consensus MLP is trained on the
   fired/hard subset, so applying it to non-fired (easy) images is mildly
   out-of-distribution; the abstention layer (Phase 5) is the safety net. This is
   the same behavior the backend exhibits.

---

## 4. Phase results table

| Phase | Metric targeted | Before | After | Outcome |
|-------|----------------|--------|-------|---------|
| 1 — Agent A retrain | Balanced accuracy | 0.562 | [TO BE FILLED] | [TO BE FILLED] |
| 1 — Agent B retrain | Balanced accuracy | 0.700 | [TO BE FILLED] | [TO BE FILLED] |
| 1 — Agent A AK recall | per-class recall | 0.49 | [TO BE FILLED] | [TO BE FILLED] |
| 1 — Agent A SCC recall | per-class recall | 0.43 | [TO BE FILLED] | [TO BE FILLED] |
| 1 — majority-class guard | NV, BCC recall (must not drop >~10 pts) | NV high / BCC mid | [TO BE FILLED] | [TO BE FILLED — report worst-hit class] |
| 2 — minority data | extra DF/VASC/SCC/AK images | 0 added | see §6 | Not added; documented as a limitation |
| 3 — trigger fire rate | D_hard fire rate | 98.7% | [TO BE FILLED] | [TO BE FILLED] |
| 3 — Consensus (23-dim) | Argus balanced accuracy | 0.666 (788-dim) | [TO BE FILLED] | [TO BE FILLED] |
| 3 — Consensus ECE | calibrated ECE | 0.075 | [TO BE FILLED] | [TO BE FILLED] |
| 4 — Malignant recall | recall on {MEL,BCC,AK,SCC} | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| 4 — MEL→NV rate | MEL misclassified as NV | 0.31 | [TO BE FILLED] | [TO BE FILLED] |
| 5 — Abstention | selective bal acc at chosen τ | N/A | [TO BE FILLED] | [TO BE FILLED] |
| 6 — Bootstrap CI | 95% CI for Argus bal acc | N/A | [TO BE FILLED] | [TO BE FILLED] |

> Reminder when filling: for every reweighting/resampling change, report the
> single **worst-affected** class alongside the average, and state plainly if any
> majority class (NV, BCC) lost more than ~10 recall points.

---

## 5. Final headline numbers (fill after the Kaggle run)

| Metric | Argus (full) | Standard Ensemble | Deep Ensemble | Agent A | Agent B |
|--------|-------------|-------------------|---------------|---------|---------|
| Balanced accuracy | [TO BE FILLED] | 0.725 (prev) | [TO BE FILLED] | 0.562 (prev) | 0.700 (prev) |
| Balanced acc 95% CI | [TO BE FILLED] | [TO BE FILLED] | — | — | — |
| Plain accuracy | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Macro AUC | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| ECE (calibrated) | [TO BE FILLED] | 0.179 (prev) | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Malignant recall | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Malignant precision | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| MEL→NV rate | [TO BE FILLED] | [TO BE FILLED] | — | [TO BE FILLED] | [TO BE FILLED] |

- Abstention threshold recommended: **[TO BE FILLED]**
- Coverage at recommended threshold: **[TO BE FILLED]**
- Selective balanced accuracy at threshold: **[TO BE FILLED]**
- Bootstrap CIs overlap (Argus vs Ensemble)? **[TO BE FILLED — YES/NO]** (if YES,
  the difference is not statistically established — say so.)

---

## 6. Limitations (known before running)

- **DF (239) and VASC (253)** have very few images. Per-class recall for these
  will be noisy and may stay low regardless of loss weighting; their numbers carry
  wide uncertainty.
- **Phase 2 (extra minority data) was not performed.** Additional ISIC-2020 / full
  ISIC Archive data was not integrated: the current Kaggle pipeline is pinned to
  the `andrewmvd/isic-2019` dataset and the established stratified split, and no
  additional source was wired in or verified as accessible in that environment. No
  data was fabricated or simulated. This is a real limitation; the rebuild relies
  on Phase-1 loss/sampling (already present) + early stopping + stronger
  augmentation instead.
- **LLM debate text was removed.** The 23-dim numerical version does **not** test
  the original hypothesis that natural-language argument generation adds
  information beyond the softmax/attention statistics. The backend therefore no
  longer produces debate transcripts for the UI; the interpretability story is now
  the attention heatmaps + the numeric feature summary, not generated arguments.
  (The frontend `ArgumentStream` component will receive no debate data; updating it
  is out of scope for this change.)
- **Consensus applied out-of-distribution.** The consensus MLP is trained on the
  hard/fired subset but applied to every test/served image. With the current ~98.7%
  fire rate this is nearly moot; if Phase-1 retraining drops the fire rate, more
  non-fired (easy) images receive consensus predictions the head was not trained
  on. The abstention layer mitigates this.
- **No clinical validation.** This is a research prototype and must not be used for
  diagnostic decisions.
- **No Kaggle numbers are in this file.** Any "After"/headline value is a
  placeholder until the human runs the notebooks and pastes the real output.

---

## 7. Files changed (for reproducibility)

```
ml_training/01_train_agent_a.ipynb        early stopping + per-class report (loss/sampler unchanged)
ml_training/02_train_agent_b.ipynb        early stopping + per-class report (loss/sampler unchanged)
ml_training/03_build_hard_subset.ipynb    fire-rate-vs-98.7% breakdown cell (thresholds unchanged)
ml_training/04_train_consensus.ipynb      788-dim -> 23-dim pipeline, new MLP, scaler, training config
ml_training/05_evaluation.ipynb           23-dim pipeline + Phases 4-7 (malignant, abstention, CI, errors)
ml_training/debate_text_utils.py          repurposed -> canonical extract_consensus_features only
backend/ml/debate/features.py             NEW: canonical extract_consensus_features
backend/ml/debate/argument_gen.py         DELETED
backend/ml/debate/encoder.py              DELETED
backend/ml/consensus/classifier.py        23-dim MLP + StandardScaler at inference
backend/ml/pipeline.py                    removed Groq/debate; agents -> trigger -> attention -> 23-dim -> consensus
backend/core/config.py                    removed GROQ_*, added CONSENSUS_SCALER
backend/.env.example                      removed GROQ_*, added CONSENSUS_SCALER
backend/requirements.txt                  removed groq + sentence-transformers, added scikit-learn + joblib
backend/ml/attention/disagreement.py      docstring corrected 788-dim -> 23-dim (code unchanged)
docker-compose.yml                        removed GROQ_API_KEY/GROQ_MODEL env injection
.env.example                              rewritten: no Groq; documents CONSENSUS_SCALER
```

Deleted dead scratch dumps that contained the removed Groq/788 pipeline and would
have contradicted the migration: `nb4_code.py`, `nb4_code_utf8.py`,
`nb5_code_utf8.py`, and `ml_training/verify_debate_text.py` (the last referenced the
now-deleted `argument_gen.py` and would crash if run). Three generic, Groq-free
notebook-editing helpers remain at the repo root (`edit_notebooks.py`,
`edit_notebook_sqrt.py`, `dump_log.py`); they are stale one-offs not used by any
deliverable and can be deleted at the maintainer's discretion.

---

## 8. Verification performed (no Kaggle run)

The deliverables were checked statically (no Kaggle execution): every notebook
code cell and every backend module passes `ast.parse`; the `extract_consensus_features`
executable body is byte-identical across the four sites (file, backend, NB04, NB05);
the consensus MLP architecture and parameter names match across NB04/NB05/backend so
the checkpoint reloads with zero missing/unexpected keys; the StandardScaler is fit on
the training split only and applied before the MLP at every consuming site. An
adversarial multi-agent review (five per-area reviewers + a completeness critic)
returned 0 critical and 0 high findings; the issues it raised (a JS-display base
mismatch in NB05, stale Groq references in scratch files / `.env.example` /
`docker-compose.yml` / one docstring) were all fixed above. None of this substitutes
for actually running the notebooks on Kaggle and filling in §4–§5.
