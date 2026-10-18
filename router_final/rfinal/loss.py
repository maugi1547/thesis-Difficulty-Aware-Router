"""
Loss training detektor dual-branch (router_final).

    L = L_A + w_B · L_B,    L_X = (box·7.5 + cls·0.5 + dfl·1.5)_X · batch_size     (v8DetectionLoss STOK, per head)

Tidak ada gate loss, sparsity penalty, maupun proxy loss. Kedua head punya TaskAlignedAssigner sendiri dan
dilatih penuh di setiap batch. w_B (default 0.7, sama dengan M0) dibaca dari model.branch_b_loss_weight.
"""
from __future__ import annotations

import torch
from ultralytics.utils.loss import v8DetectionLoss


class BranchLoss(v8DetectionLoss):
    """v8DetectionLoss stok untuk SATU Detect head (bawaan Ultralytics selalu membaca model.model[-1])."""

    def __init__(self, model, head, tal_topk: int = 10):
        super().__init__(model, tal_topk=tal_topk)
        assert head.nc == self.nc and head.reg_max == self.reg_max, "kedua head harus punya nc & reg_max sama"
        self.stride = head.stride


class DualBranchLoss:
    """Dipanggil DualBranchDetector.loss(): (out_A, out_B), batch -> (total skalar, 6 item detached)."""

    names = ("box_A", "cls_A", "dfl_A", "box_B", "cls_B", "dfl_B")

    def __init__(self, model):
        raw = model.module if hasattr(model, "module") else model
        self.loss_A = BranchLoss(raw, raw.detect_A)
        self.loss_B = BranchLoss(raw, raw.detect_B)
        self.w_B = float(getattr(raw, "branch_b_loss_weight", 0.7))

    def __call__(self, preds, batch):
        out_A, out_B = preds
        la, items_a = self.loss_A(out_A, batch)      # la = vektor 3 (sudah × batch_size)
        lb, items_b = self.loss_B(out_B, batch)
        total = la.sum() + self.w_B * lb.sum()
        return total, torch.cat([items_a, items_b])
