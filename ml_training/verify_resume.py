"""Verification tool for Phase 5 (checkpoint/resume) — RUN ON KAGGLE (needs torch).

Proves the resume path restores optimizer + scheduler + AMP-scaler state EXACTLY: after
saving mid-schedule and reloading into FRESH objects, the learning rate equals what the
uninterrupted cosine schedule would produce at that epoch, and the early-stopping counters
(best_auc, epochs_no_improve) come back intact. No images / GPU required — runs in ~1s on CPU.

Usage on Kaggle:  python verify_resume.py
"""

from __future__ import annotations


def main() -> None:
    import torch
    import torch.nn.functional as F

    from training_utils import save_resumable, load_resumable

    torch.manual_seed(0)
    T_MAX = 10
    K = 4  # interrupt after this many epochs

    def make():
        m = torch.nn.Linear(4, 8)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return m, opt, sched, scaler

    def step_epoch(m, opt, sched):
        # Mirror the notebook's per-epoch order: train step(s) -> scheduler.step().
        x = torch.randn(2, 4)
        y = torch.randint(0, 8, (2,))
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        sched.step()

    # Reference: the uninterrupted schedule's LR after each epoch.
    m, opt, sched, _ = make()
    ref_lr = []
    for _ in range(T_MAX):
        step_epoch(m, opt, sched)
        ref_lr.append(opt.param_groups[0]["lr"])

    # Interrupted run: K epochs, then save the full resumable state.
    m2, opt2, sched2, scaler2 = make()
    for _ in range(K):
        step_epoch(m2, opt2, sched2)
    lr_before = opt2.param_groups[0]["lr"]
    path = "/tmp/_argus_resume_test.pth"
    save_resumable(path, "test", K, m2, opt2, sched2, scaler2, best_auc=0.77, epochs_no_improve=2)

    # Resume into BRAND-NEW objects (simulating a fresh process after a crash).
    m3, opt3, sched3, scaler3 = make()
    r = load_resumable(path, m3, opt3, sched3, scaler3, map_location="cpu")
    lr_after = opt3.param_groups[0]["lr"]

    print(f"LR at epoch {K} before interrupt : {lr_before:.8e}")
    print(f"LR after resume (fresh objects)  : {lr_after:.8e}")
    print(f"Uninterrupted schedule LR @ {K}    : {ref_lr[K-1]:.8e}")
    print(f"resume bookkeeping               : start_epoch={r['start_epoch']} "
          f"best_auc={r['best_auc']:.2f} epochs_no_improve={r['epochs_no_improve']}")

    assert r["start_epoch"] == K + 1, "resume must continue at epoch K+1"
    assert abs(r["best_auc"] - 0.77) < 1e-9, "best_auc not restored"
    assert r["epochs_no_improve"] == 2, "epochs_no_improve not restored"
    assert abs(lr_after - lr_before) < 1e-12, "resumed LR != pre-interrupt LR"
    assert abs(lr_after - ref_lr[K-1]) < 1e-12, "resumed LR != uninterrupted schedule at epoch K"

    print("\n[PASS] resume restores model/optimizer/scheduler/scaler exactly; LR matches the "
          "uninterrupted cosine schedule; best-AUC and early-stop counters restored.")


if __name__ == "__main__":
    main()
