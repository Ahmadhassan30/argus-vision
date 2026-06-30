"""Verification tool for Phase 4 (decoupled training) — RUN ON KAGGLE (needs torch).

Proves, on the ACTUAL EfficientNet-B4 and ViT-B/16 architectures (random tensors, no
images needed), that:
  1. freeze_all_but_classifier leaves ONLY the classifier head trainable;
  2. after a few optimizer steps on the head, every backbone parameter is BIT-IDENTICAL
     (the freeze genuinely held) — via snapshot_frozen_params + assert_frozen_unchanged;
  3. logit adjustment / LogitAdjustedLoss run and shift the loss as expected.

This is the "freeze actually took effect" verification the brief requires. The full
end-to-end image smoke test = run notebook 01/02 on Kaggle with small STAGE_A_EPOCHS /
STAGE_B_EPOCHS. Usage on Kaggle:  `python verify_decoupled.py`
"""

from __future__ import annotations


def _check_model(name: str) -> None:
    import timm
    import torch
    import torch.nn as nn

    from training_utils import (
        apply_logit_adjustment,
        assert_frozen_unchanged,
        class_priors_from_counts,
        freeze_all_but_classifier,
        freeze_backbone_bn,
        snapshot_frozen_params,
        LogitAdjustedLoss,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(name, pretrained=False, num_classes=8).to(device)

    # --- Stage B freeze (params + BN running stats) ---
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_train, n_total = freeze_all_but_classifier(model)
    n_bn = freeze_backbone_bn(model)
    head_params = sum(p.numel() for p in model.get_classifier().parameters())
    assert n_train == head_params, f"{name}: trainable ({n_train}) != head ({head_params})"
    assert n_train < trainable_before, f"{name}: freeze did not reduce trainable params"
    frozen_snapshot = snapshot_frozen_params(model)
    buffer_snapshot = {k: b.detach().clone() for k, b in model.named_buffers()}
    assert len(frozen_snapshot) > 0

    # --- A few real optimizer steps on the head only ---
    from losses import FocalLoss

    counts = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
    priors = class_priors_from_counts(counts)
    crit = LogitAdjustedLoss(FocalLoss(gamma=2.0, alpha=None), priors, tau=1.0).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    for _ in range(3):
        x = torch.randn(4, 3, 224, 224, device=device)
        y = torch.randint(0, 8, (4,), device=device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    # --- Prove the backbone is bit-identical: params AND BN buffers ---
    n_verified = assert_frozen_unchanged(model, frozen_snapshot)
    drifted = [
        k for k, b in model.named_buffers()
        if not torch.equal(b.detach(), buffer_snapshot[k])
    ]
    assert not drifted, f"{name}: backbone BUFFERS drifted (BN stats not frozen): {drifted[:5]}"

    # --- Logit adjustment direction sanity ---
    raw = torch.zeros(1, 8, device=device)
    adj = apply_logit_adjustment(raw, priors, tau=1.0)
    assert adj[0, 1] > adj[0, 5], "majority logit must exceed rare logit after adjustment"

    print(f"[PASS] {name}: head-only trainable={n_train:,} (of {n_total:,}); "
          f"{n_verified:,} backbone params + {len(buffer_snapshot):,} buffers bit-identical "
          f"after 3 Stage-B steps (BN modules frozen={n_bn}); logit adjustment OK.")


def main() -> None:
    for name in ("efficientnet_b4", "vit_base_patch16_224"):
        _check_model(name)
    print("\nALL PHASE-4 FREEZE / LOGIT CHECKS PASSED on the real architectures.")


if __name__ == "__main__":
    main()
